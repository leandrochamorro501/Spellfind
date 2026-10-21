"""Profile the real pipeline, recording blockers instead of substituting controls."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from threading import Event, Thread
import time

from evaluation.memory import process_memory, gpu_memory


def run(repeats=1, allow_model_downloads=False, qrels_path=None):
    if not allow_model_downloads:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from evaluation.run import run as evaluate
    samples, stop = [], Event()

    def sample():
        while not stop.is_set():
            samples.append(process_memory())
            stop.wait(.1)

    sampler = Thread(target=sample, daemon=True)
    started = time.perf_counter()
    sampler.start()
    try:
        with redirect_stdout(io.StringIO()):
            report = evaluate(mode="real", repeats=repeats, baseline_id="pending-fixes-real", qrels_path=qrels_path)
    finally:
        stop.set()
        sampler.join(timeout=2)
    report["runtime_profile"] = dict(elapsed_seconds=time.perf_counter()-started,
        **process_memory(), **gpu_memory(),
        process_sample_peak_rss_bytes=max((row["process_rss_bytes"] for row in samples
            if row["process_rss_bytes"] is not None), default=None),
        rss_samples=len(samples), sampling_interval_seconds=.1,
        allow_model_downloads=allow_model_downloads,
        real_inference_completed=report["status"] == "completed" and bool(report["runs"]),
        limitations=["Sample peaks can miss short spikes; lifetime peaks include initialization.",
                     "Blocked runs measure startup only, not real model performance.",
                     "RSS includes the interpreter, caches and native allocations."])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--qrels", type=Path)
    args = parser.parse_args()
    if args.repeats < 1: parser.error("repeats must be positive")
    report = run(args.repeats, args.allow_model_downloads, args.qrels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], reason=report.get("blocked_reason"),
                         memory=report["runtime_profile"])))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
