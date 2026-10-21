"""Local runtime load profile: real catalog/FAISS, explicitly controlled encoders and judge."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import statistics
from threading import Lock
import time
import tracemalloc
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.run import manifest
from evaluation.memory import process_memory, gpu_memory


def distribution(values):
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[min(len(ordered)-1, int(.95*(len(ordered)-1)))],
                stdev=statistics.stdev(values) if len(values) > 1 else None)


def compact(result):
    cache_hits = cache_misses = 0
    measures = {}
    for attempt in result["retrieval_history"]:
        for branch in attempt["branches"].values():
            cache_hits += int(branch.get("request_cache_hit", False))
            for name, value in (branch.get("worker_profile") or {}).items():
                measures[name] = measures.get(name, 0) + value
            cache_misses += int(branch.get("request_cache_hit") is False)
    stable = dict(status=result["judge_status"], plan=result["plan"], exclusions=result["exclusions"],
                  results=[(row["song_id"], row["score"], row["validation_status"]) for row in result["results"]])
    return dict(timings=result["timings"], worker_measures=measures,
                request_cache_hits=cache_hits, request_cache_misses=cache_misses,
                digest=hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest(),
                status=result["judge_status"])


def summarize(rows, wall_seconds):
    return dict(requests=len(rows), wall_seconds=wall_seconds, requests_per_second=len(rows)/wall_seconds,
        stages_ms={key: distribution([row["timings"][key] for row in rows]) for key in rows[0]["timings"]},
        request_cache_hits=sum(row["request_cache_hits"] for row in rows),
        request_cache_misses=sum(row["request_cache_misses"] for row in rows),
        worker_measures_totals={key: sum(row["worker_measures"].get(key, 0) for row in rows)
                                for key in {key for row in rows for key in row["worker_measures"]}},
        consistent_results=len({row["digest"] for row in rows}) == 1,
        status_counts={status: sum(row["status"] == status for row in rows) for status in {row["status"] for row in rows}})


def run(iterations=20):
    import app
    import faiss
    import numpy as np
    import torch
    import semantic_search
    import clap_search
    from resources import get_song_id_map

    catalog_started = time.perf_counter()
    app.engine.load_indices()
    if not app.engine.ready:
        raise RuntimeError("Catalog is not ready")
    catalog_ms = (time.perf_counter()-catalog_started)*1000
    lyric_index, audio_index = app.engine.idx_lyrics, app.engine.idx_audio
    snapshot = app.engine.pin_snapshot()
    mapping = snapshot.song_id_map

    def vector(text, index):
        position = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % index.ntotal
        return index.reconstruct(position).copy()

    class ControlledLyrics:
        def encode(self, text, **kwargs):
            return vector(text, lyric_index)

    class Batch(dict):
        def to(self, device):
            return self

    class ControlledProcessor:
        def __call__(self, text, **kwargs):
            return Batch(text=text)

    class ControlledAudio:
        def get_text_features(self, text):
            return torch.from_numpy(vector(text, audio_index)[np.newaxis, :])

    seen, decision_lock = {}, Lock()

    def evaluate(**kwargs):
        with decision_lock:
            seen[kwargs["request_id"]] = seen.get(kwargs["request_id"], 0) + 1
            count = seen[kwargs["request_id"]]
        status = "rejected" if count == 1 else "approved"
        return dict(status=status, reason="CONTROL: not a musical judgment", format="control", model_calls=0)

    def plan(inputs):
        text = inputs["query"]
        return dict(lyric_positive=["home-control:" + text], lyric_negative=[],
                    clap_positive=["piano-control:" + text], clap_negative=[])

    lyric_resources = dict(embedder=ControlledLyrics(), index=lyric_index, song_id_map=mapping)
    audio_resources = dict(model=ControlledAudio(), processor=ControlledProcessor(), index=audio_index, song_id_map=mapping)
    # A separate benchmark process owns these caches; no service cache is cleared.
    semantic_search._encode_cached.cache_clear(); clap_search._encode_cached.cache_clear()
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        evidence="local_runtime_real_catalog_faiss_controlled_models", manifest=manifest(), iterations=iterations,
        catalog_songs=len(app.engine.df_meta), catalog_load_validate_ms=catalog_ms,
        real_model_encoding_ms=None, real_provider_latency_ms=None,
        limitations=["Encoders reconstruct existing vectors; no language/audio inference is measured.",
                     "Planner and judge are controls. Results do not measure musical relevance.",
                     "Process-first means after imports/catalog setup; disk-cache cold is not guaranteed.",
                     "Repeated single-query workload; diverse inputs, eviction pressure and real model contention are not established.",
                     "One FAISS OpenMP thread per benchmark task; this does not tune production settings.",
                     "Python tracemalloc runs in a separate memory phase, excludes native tensors/FAISS and GPU; harness overhead is included.",
                     "RSS is measured for this controlled workload; real model loading and encoding remain unmeasured.",
                     "GPU counters, if available, cover Torch allocations only, not total driver memory."])
    with ExitStack() as stack:
        stack.enter_context(patch.object(semantic_search, "_load_resources", return_value=lyric_resources))
        stack.enter_context(patch.object(clap_search, "_load_resources", return_value=audio_resources))
        stack.enter_context(patch.object(app, "_import_router", return_value=SimpleNamespace(invoke=plan)))
        stack.enter_context(patch.object(app, "_import_semantic_search", return_value=semantic_search.search_lyrics))
        stack.enter_context(patch.object(app, "_import_clap_search", return_value=clap_search.search_audio))
        stack.enter_context(patch.object(app, "_import_judge", return_value=(evaluate, None)))
        stack.enter_context(patch.object(app, "REFINEMENT_ENABLED", False))
        stack.enter_context(redirect_stdout(io.StringIO()))

        def request(_=None):
            faiss.omp_set_num_threads(1)
            return app.get_top_k_results("Perfil local repetido", search_engine=snapshot)

        cold = request()
        report["process_first"] = compact(cold)
        reference = compact(cold)["digest"]
        gc.collect()
        started = time.perf_counter()
        warm = [compact(request()) for _ in range(iterations)]
        report["warm_sequential"] = summarize(warm, time.perf_counter()-started)
        report["concurrent"] = {}
        all_digests = [reference] + [row["digest"] for row in warm]
        for workers in (2, 4):
            started = time.perf_counter()
            with ThreadPoolExecutor(workers) as pool:
                rows = list(pool.map(lambda _: compact(request()), range(iterations)))
            report["concurrent"][str(workers)] = summarize(rows, time.perf_counter()-started)
            all_digests += [row["digest"] for row in rows]
        # Do not compare untraced cold latency against traced warm latency.
        # Memory tracing has its own phase and does not affect the timing samples.
        gc.collect()
        rss_samples = [process_memory()]
        tracemalloc.start()
        for _ in range(5):
            request()
            rss_samples.append(process_memory())
        gc.collect()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["memory"] = dict(python_traced_current_bytes=current, python_traced_peak_bytes=peak,
                                **process_memory(), **gpu_memory(),
                                process_sample_peak_rss_bytes=max((row['process_rss_bytes'] for row in rss_samples
                                    if row['process_rss_bytes'] is not None), default=None),
                                rss_sample_count=len(rss_samples),
                                sample_requests=5, measured_in_separate_phase=True,
                                index_vector_payload_bytes=(lyric_index.ntotal*lyric_index.d + audio_index.ntotal*audio_index.d)*4)
        report["result_invariants_preserved"] = len(set(all_digests)) == 1
        report["embedding_cache_final"] = dict(lyrics=semantic_search._encode_cached.cache_info()._asdict(),
                                               audio=clap_search._encode_cached.cache_info()._asdict())
        report["total_control_requests"] = 1 + iterations*3 + 5
    report["decision"] = "Keep request-local graphs and bounded existing caches; no real-model evidence justifies a new shared graph/cache or batching."
    report["status"] = "completed" if report["result_invariants_preserved"] else "failed_invariants"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("At least two iterations are required")
    report = run(args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "catalog_songs", "result_invariants_preserved", "memory")}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
