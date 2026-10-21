"""Real catalog transactions/HTTP/queue with controlled network and model adapters."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import shutil
import sys
import tempfile
from uuid import uuid4
from threading import Event
import time
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from catalog import CatalogManager, FILES
from ingestion import IngestionService, JobStore, QueueFull, youtube_url, ingest_payload
from ingestion_backend import NativeBackend, subtitle_text
from resources import use_snapshot, search_resources, get_index, get_song_id_map


URL = "https://www.youtube.com/watch?v=abcdefghijk"


def bundle(path, count=2):
    path.mkdir(parents=True)
    pd.DataFrame([dict(song_id=i+1, title=f"Song {i+1}", artist="Original", tag="rock", year=2020,
                       lyrics="Returning home to my family") for i in range(count)]).to_csv(path / FILES[0], index=False)
    (path / FILES[1]).write_text(json.dumps({str(i): i+1 for i in range(count)}), encoding="utf-8")
    for name, dimension in zip(FILES[2:], (256, 512)):
        index = faiss.IndexFlatIP(dimension); index.add(np.eye(dimension, dtype=np.float32)[:count])
        faiss.write_index(index, str(path / name))


def metadata():
    return dict(title="New song", artist="New artist", tag="pop", year="2021", lyrics="New lyrics about home", source_url=URL)


class ControlledBackend:
    def __init__(self):
        self.entered, self.release = Event(), Event()
        self.calls = 0

    def capabilities(self): return dict(ready=True, tools={"control": True}, model_weights_verified=False)

    def run(self, job, work, progress):
        self.calls += 1; self.entered.set()
        progress("downloading", 10)
        if not self.release.wait(5): raise RuntimeError("Control timed out")
        audio = work / "audio.mp3"; audio.write_bytes(b"ID3CONTROL-NOT-REAL-AUDIO")
        progress("embedding_lyrics", 65)
        return {**metadata(), **job["metadata"], "lyrics_source": "controlled_fixture"}, np.ones(256), np.ones(512), audio


class Block5Tests(unittest.TestCase):
    def setUp(self):
        # Windows restricted tokens can create a mode-0700 TemporaryDirectory
        # but cannot traverse it. Inherit the workspace ACL for disposable tests.
        base = Path(__file__).resolve().parents[1] / "tmp" / "block5-tests"
        self.root = base / uuid4().hex
        self.root.mkdir(parents=True)
        def cleanup():
            target = self.root.resolve()
            if target.parent != base.resolve(): raise RuntimeError("Unexpected cleanup target")
            shutil.rmtree(target)
        self.addCleanup(cleanup)
        self.legacy = self.root / "legacy"; bundle(self.legacy)
        self.manager = CatalogManager(self.root / "catalog", self.legacy)
        self.initial = self.manager.current()

    def append(self, **kwargs):
        return self.manager.append_song(metadata(), np.ones(256), np.ones(512), job_id="a"*32, **kwargs)

    def service(self):
        backend = ControlledBackend()
        service = IngestionService(self.manager, self.root / "ingest", backend)
        def close(): backend.release.set(); service.close()
        self.addCleanup(close)
        return service, backend

    def client(self, service=None, enabled=True):
        self.engine = app.SearchEngine(self.manager); self.engine.load_indices()
        return app.create_app({"TESTING": True, "INGEST_ENABLED": enabled, "INGEST_DIR": str(self.root / "ingest")},
                              search_engine=self.engine, initialize=False, ingest_service=service).test_client()

    def await_job(self, service, job_id):
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            job = service.store.get(job_id)
            if job["status"] in {"completed", "failed", "interrupted"}: return job
            time.sleep(.01)
        self.fail("Controlled job did not terminate")

    def test_bootstrap_preserves_legacy_files_and_survives_restart(self):
        before = {name: (self.legacy/name).read_bytes() for name in FILES}
        again = CatalogManager(self.manager.root, self.legacy).current()
        self.assertEqual(again.version, self.initial.version)
        self.assertEqual({name: (self.legacy/name).read_bytes() for name in FILES}, before)

    def test_bad_publication_keeps_current_generation(self):
        bad = self.root / "bad"; bundle(bad)
        (bad / FILES[1]).write_text('{"0": 999, "1": 2}', encoding="utf-8")
        with self.assertRaises(ValueError): self.manager.publish_directory(bad)
        self.assertEqual(self.manager.current().version, self.initial.version)

    def test_new_reader_rejects_tampered_snapshot(self):
        path = self.manager.root / "versions" / self.initial.version / FILES[1]
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError): CatalogManager(self.manager.root, self.legacy).current()
        # The previously loaded in-memory snapshot is still a consistent value.
        self.assertEqual(self.initial.idx_lyrics.ntotal, 2)

    def test_snapshot_consumers_cannot_mutate_canonical_data(self):
        row = self.initial.get_song_by_id(1); row["lyrics"] = "changed"
        frame = self.initial.df_meta; frame.loc[0, "lyrics"] = "changed"
        mapping = self.initial.song_id_map; mapping["0"] = 100
        manifest = self.initial.manifest; manifest["sha256"].clear()
        self.assertNotEqual(self.initial.get_song_by_id(1)["lyrics"], "changed")
        self.assertEqual(self.initial.song_id_map["0"], 1)
        self.assertTrue(self.initial.manifest["sha256"])
        self.assertFalse(hasattr(self.initial.idx_lyrics, "add"))

    def test_concurrent_publishers_do_not_lose_updates(self):
        other = CatalogManager(self.manager.root, self.legacy)
        def add(which):
            manager = self.manager if which == 0 else other
            return manager.append_song(dict(metadata(), source_url=URL+str(which)), np.ones(256), np.ones(512), job_id=str(which)*32)
        with ThreadPoolExecutor(2) as pool: results = list(pool.map(add, [0, 1]))
        latest = self.manager.current()
        self.assertEqual(latest.idx_lyrics.ntotal, 4)
        self.assertEqual(len({sid for _, sid in results}), 2)

    def test_duplicate_source_or_job_is_idempotent(self):
        first, sid = self.append()
        again, second = self.append()
        self.assertEqual((first.version, sid), (again.version, second))
        self.assertEqual(again.idx_audio.ntotal, 3)

    def test_lost_owner_guard_prevents_publication(self):
        with self.assertRaises(RuntimeError): self.append(guard=Mock(side_effect=RuntimeError("lost session")))
        self.assertEqual(self.manager.current().version, self.initial.version)
        orphans = [path.name for path in (self.manager.root / "versions").iterdir() if path.name != self.initial.version]
        self.assertTrue(orphans)
        with self.assertRaises(ValueError): self.manager.at_version(orphans[0])

    def test_rebuild_compare_and_swap_rejects_stale_base(self):
        self.append()
        with self.assertRaises(ValueError): self.manager.publish_directory(self.legacy, expected_version=self.initial.version)
        self.assertEqual(self.manager.current().idx_audio.ntotal, 3)

    def test_old_snapshot_and_new_snapshot_keep_separate_resources(self):
        latest, _ = self.append()
        model = {"embedder": object(), "index": self.initial.idx_lyrics, "song_id_map": self.initial.song_id_map}
        def count(snapshot):
            with use_snapshot(snapshot):
                resources = search_resources(model)
                return resources["index"].ntotal, len(resources["song_id_map"]), get_index("lyrics").ntotal
        with ThreadPoolExecutor(2) as pool: values = list(pool.map(count, [self.initial, latest]))
        self.assertEqual(values, [(2, 2, 2), (3, 3, 3)])

    def test_inflight_search_remains_on_old_generation(self):
        engine = app.SearchEngine(self.manager); engine.load_indices()
        entered, release = Event(), Event()
        def plan(_):
            entered.set()
            if not release.wait(5): raise RuntimeError("test timeout")
            return dict(lyric_positive=["home"], lyric_negative=[], clap_positive=[], clap_negative=[])
        def worker(**kw):
            index, mapping = get_index("lyrics"), get_song_id_map()
            self.assertEqual(index.ntotal, len(mapping))
            return [(sid, 1/(i+1)) for i, sid in enumerate(mapping.values())]
        with patch.object(app, "_import_router", return_value=SimpleNamespace(invoke=plan)), \
             patch.object(app, "_import_judge", return_value=(lambda **_: dict(status="approved", reason="control"), None)), \
             patch.object(app, "_import_semantic_search", return_value=worker):
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(app.get_top_k_results, "Letra sobre casa", search_engine=engine)
                self.assertTrue(entered.wait(3))
                try: latest, _ = self.append()
                finally: release.set()
                result = future.result(timeout=5)
            second = app.get_top_k_results("Letra sobre casa", search_engine=engine)
        self.assertEqual(result["catalog_version"], self.initial.version)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(second["catalog_version"], latest.version)
        self.assertEqual(len(second["results"]), 3)

    def test_canonical_youtube_validation_excludes_other_hosts(self):
        self.assertEqual(youtube_url("https://youtu.be/abcdefghijk?t=12"), URL)
        self.assertEqual(youtube_url("https://www.youtube.com/shorts/abcdefghijk"), URL)
        for bad in ("http://youtube.com/watch?v=abcdefghijk", "https://youtube.com.evil/watch?v=abcdefghijk",
                    "https://user@youtube.com/watch?v=abcdefghijk", "file:///tmp/song", "https://youtu.be/../../x",
                    "https://youtube.com/watch?v=abcdefghijk&v=ABCDEFGHIJK", "https://youtube.com:444/watch?v=abcdefghijk"):
            with self.subTest(url=bad), self.assertRaises(ValueError): youtube_url(bad)

    def test_ingest_metadata_does_not_guess_year(self):
        url, record = ingest_payload(dict(url=URL, genre="pop", year=2020))
        self.assertEqual(record, dict(tag="pop", year="2020"))
        for bad in (None, {}, dict(url=URL, year=True), dict(url=URL, cookie="secret")):
            with self.assertRaises(ValueError): ingest_payload(bad)

    def test_jobs_persist_and_claim_once_across_workers(self):
        store = JobStore(self.root / "jobs")
        job = store.enqueue(URL, {})
        other = JobStore(self.root / "jobs")
        self.assertEqual(other.get(job["id"])["status"], "queued")
        with ThreadPoolExecutor(2) as pool: claims = list(pool.map(lambda x: (store if x == 0 else other).claim(str(x)), [0, 1]))
        self.assertEqual(sum(row is not None for row in claims), 1)

    def test_expired_running_job_is_interrupted_not_automatically_restarted(self):
        store = JobStore(self.root / "jobs")
        job = store.enqueue(URL, {}); store.claim("dead")
        with store.db() as db: db.execute("UPDATE jobs SET lease=0 WHERE id=?", (job["id"],))
        self.assertEqual(JobStore(self.root / "jobs").get(job["id"])["status"], "interrupted")
        with self.assertRaises(RuntimeError): store.assert_owned(job["id"], "dead")
        self.assertIsNone(store.claim("new"))

    def test_queue_duplicate_and_capacity_are_bounded(self):
        store = JobStore(self.root / "jobs")
        first = store.enqueue(URL, {})
        self.assertEqual(store.enqueue(URL, {})["id"], first["id"])
        for i in range(31): store.enqueue(f"https://www.youtube.com/watch?v={i:011d}", {})
        with self.assertRaises(QueueFull): store.enqueue("https://www.youtube.com/watch?v=ABCDEFGHIJK", {})

    def test_http_job_is_async_then_publishes_and_audio_is_versioned(self):
        service, backend = self.service(); client = self.client(service)
        response = client.post("/ingest", json=dict(url=URL, genre="pop", year=2021))
        self.assertEqual(response.status_code, 202)
        job_id = response.json["job"]["id"]
        self.assertTrue(backend.entered.wait(2))
        self.assertEqual(client.get("/health").json["songs"], 2)
        self.assertIn(client.get(f"/ingest/jobs/{job_id}").json["job"]["status"], ("queued", "running"))
        backend.release.set()
        job = self.await_job(service, job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(client.get("/health").json["songs"], 3)
        version, sid = job["result"]["catalog_version"], job["result"]["song_id"]
        audio = client.get(f"/catalog/{version}/songs/{sid}/audio", headers={"Range": "bytes=0-2"})
        self.assertEqual(audio.status_code, 206)
        self.assertEqual(audio.data, b"ID3"); audio.close()
        repeated = client.post("/ingest", json=dict(url=URL)).json["job"]
        done = self.await_job(service, repeated["id"])
        self.assertTrue(done["result"]["duplicate"])
        self.assertEqual(backend.calls, 1)

    def test_failed_processing_does_not_publish_partial_data(self):
        service, backend = self.service(); backend.run = Mock(side_effect=RuntimeError("controlled decode failure"))
        client = self.client(service)
        job = client.post("/ingest", json=dict(url=URL)).json["job"]
        self.assertEqual(self.await_job(service, job["id"])["status"], "failed")
        self.assertEqual(self.manager.current().version, self.initial.version)

    def test_disabled_and_cross_origin_requests_are_controlled(self):
        client = self.client(enabled=False)
        self.assertFalse(client.get("/ingest/status").json["enabled"])
        self.assertEqual(client.post("/ingest", json=dict(url=URL)).status_code, 503)
        self.assertEqual(client.post("/ingest", json=dict(url=URL), headers={"Origin": "https://another.invalid"}).status_code, 403)
        self.assertEqual(client.post("/ingest", json=dict(url="https://another.invalid")).status_code, 400)
        self.assertEqual(client.get("/ingest/jobs/../../anything").status_code, 404)

    def test_official_subtitles_remove_timestamps_tags_and_duplicate_cues(self):
        text = "WEBVTT\nKind: captions\n\n1\n00:00:00.000 --> 00:00:03.000\n<c>Returning home</c>\nReturning home\n[Music]\n"
        self.assertEqual(subtitle_text(text), "Returning home")

    def native(self, *, fail_subtitles=False, whisper_text="Fallback words about returning home", duration=30, instrumental=False):
        import semantic_search, clap_search, torch
        work = self.root / "native-work"; work.mkdir()
        info = dict(duration=duration, title="Control", uploader="Uploader", upload_date="20200101", subtitles={"en": []})
        class Downloader:
            def __init__(self, options): self.options = options
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def extract_info(self, url, download=False):
                if download and self.options.get("writesubtitles"):
                    if fail_subtitles: raise RuntimeError("CONTROL subtitles unavailable")
                    (work / "audio.en.vtt").write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nReturning home in this control\n", encoding="utf-8")
                elif download: (work / "audio.mp3").write_bytes(b"ID3-CONTROL")
                return info
        ytdlp = ModuleType("yt_dlp"); ytdlp.YoutubeDL = Downloader
        whisper = ModuleType("whisper"); whisper.load_model = Mock(return_value=SimpleNamespace(transcribe=lambda *_args, **_kw: {"text": whisper_text}))
        class Batch(dict):
            def to(self, device): return self
        def process(**kwargs):
            self.assertIn("audio", kwargs); self.assertEqual(kwargs["sampling_rate"], 48000)
            return Batch(input_features=torch.ones(1, 2))
        embedder = SimpleNamespace(encode=Mock(return_value=np.ones(768)))
        audio_resources = dict(processor=Mock(side_effect=process), model=SimpleNamespace(get_audio_features=Mock(return_value=torch.ones(1, 512))))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"yt_dlp": ytdlp, "whisper": whisper}))
            stack.enter_context(patch.object(semantic_search, "_load_resources", return_value={"embedder": embedder}))
            stack.enter_context(patch.object(clap_search, "_load_resources", return_value=audio_resources))
            ffmpeg = stack.enter_context(patch("ingestion_backend.subprocess.run", return_value=SimpleNamespace(stdout=np.ones(4800, dtype="<f4").tobytes())))
            output = NativeBackend().run(dict(url=URL, metadata={"instrumental": True} if instrumental else {}), work, lambda *_: None)
        return output, whisper, embedder, ffmpeg

    def test_native_adapter_prefers_official_text_and_builds_both_vectors(self):
        (record, lv, av, audio), whisper, embedder, ffmpeg = self.native()
        self.assertEqual(record["lyrics_source"], "official_subtitles")
        self.assertEqual(record["year"], "")  # Upload date is not release year.
        self.assertEqual(record["transcript_coverage"], "unverified")
        self.assertTrue(audio.is_file())
        self.assertEqual((lv.shape, av.shape), ((256,), (512,)))
        self.assertAlmostEqual(float(np.linalg.norm(lv)), 1, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(av)), 1, places=5)
        whisper.load_model.assert_not_called()
        self.assertTrue(embedder.encode.call_args.args[0].startswith("search_document: "))
        self.assertEqual(ffmpeg.call_count, 3)
        self.assertTrue(all(isinstance(call.args[0], list) and not call.kwargs.get("shell") for call in ffmpeg.call_args_list))

    def test_native_subtitle_failure_uses_whisper_alternative(self):
        (record, *_), whisper, *_rest = self.native(fail_subtitles=True)
        self.assertEqual(record["lyrics_source"], "whisper")
        from config import BASE_DIR
        whisper.load_model.assert_called_once_with("base", device="cpu", download_root=str(BASE_DIR / ".tools/whisper"))

    def test_empty_transcript_is_not_fabricated_or_published(self):
        with self.assertRaises(ValueError): self.native(fail_subtitles=True, whisper_text="")
        self.assertEqual(self.manager.current().version, self.initial.version)

    def test_declared_instrumental_skips_lyrics_models_and_keeps_audio(self):
        (record, lv, av, audio), whisper, embedder, ffmpeg = self.native(instrumental=True)
        self.assertIsNone(lv)
        self.assertEqual(record['lyrics'], '')
        self.assertEqual(record['instrumental_source'], 'user_declared_not_acoustically_verified')
        self.assertEqual(av.shape, (512,))
        whisper.load_model.assert_not_called()
        embedder.encode.assert_not_called()

    def test_unknown_or_excessive_duration_is_rejected(self):
        with self.assertRaises(ValueError): self.native(duration=float("inf"))

    def test_engine_keeps_loaded_generation_when_new_pointer_is_invalid(self):
        client = self.client(enabled=False)
        with self.manager.transaction() as db:
            db.execute("UPDATE current SET version=? WHERE singleton=1", ("f"*32,))
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["catalog_version"], self.initial.version)
        self.assertTrue(response.json["catalog_warning"])

    def test_cli_and_web_use_the_same_pinned_filtered_catalog(self):
        import cli
        latest, sid = self.append()
        engine = app.SearchEngine(self.manager)
        def worker(**kwargs): return [(sid, 1.)]
        with patch.object(app, "engine", engine), \
             patch.object(app, "_import_router", return_value=SimpleNamespace(invoke=lambda _:
                dict(lyric_positive=["home"], lyric_negative=[], clap_positive=[], clap_negative=[]))), \
             patch.object(app, "_import_judge", return_value=(lambda **_: dict(status="approved", reason="control"), None)), \
             patch.object(app, "_import_semantic_search", return_value=worker), \
             patch.object(sys, "argv", ["cli.py", "Letra sobre casa", "--artist", "New artist", "--genre", "pop", "--year-min", "2021"]):
            with redirect_stdout(io.StringIO()) as output: status = cli.main()
        result = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(result["catalog_version"], latest.version)
        self.assertEqual(result["results"][0]["song_id"], sid)


if __name__ == "__main__": unittest.main()
