"""Durable local ingestion queue. One owned job per worker; publication is idempotent."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from threading import Event, Thread, Lock
import time
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4


def youtube_url(value):
    if not isinstance(value, str) or len(value) > 2048: raise ValueError("Se requiere una URL de YouTube")
    try:
        url = urlsplit(value.strip())
        if url.scheme != "https" or url.username or url.password or url.port not in (None, 443): raise ValueError()
        host = url.hostname
        if host == "youtu.be": video = url.path.strip("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if url.path == "/watch":
                values = parse_qs(url.query).get("v", [])
                if len(values) != 1: raise ValueError()
                video = values[0]
            elif url.path.startswith(("/shorts/", "/embed/")): video = url.path.split("/")[2]
            else: raise ValueError()
        else: raise ValueError()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video): raise ValueError()
        return f"https://www.youtube.com/watch?v={video}"
    except ValueError:
        raise ValueError("Usa un enlace HTTPS de un vídeo individual de YouTube") from None


def ingest_payload(payload):
    if not isinstance(payload, dict) or set(payload)-{"url", "artist", "genre", "year", "instrumental"}: raise ValueError("Solicitud de ingesta inválida")
    url = youtube_url(payload.get("url"))
    metadata = {}
    if "instrumental" in payload:
        if type(payload["instrumental"]) is not bool: raise ValueError("instrumental debe ser booleano")
        metadata["instrumental"] = payload["instrumental"]
    for name in ("artist", "genre"):
        if name in payload:
            value = payload[name]
            if not isinstance(value, str) or not value.strip() or len(value) > 300: raise ValueError(f"{name} inválido")
            metadata["tag" if name == "genre" else name] = value.strip()
    if "year" in payload:
        year = payload["year"]
        if isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999: raise ValueError("Año inválido")
        metadata["year"] = str(year)
    return url, metadata


def capabilities():
    from media_tools import ffmpeg_path
    tools = {"yt-dlp": importlib.util.find_spec("yt_dlp") is not None,
             "FFmpeg": ffmpeg_path() is not None,
             "Whisper": importlib.util.find_spec("whisper") is not None}
    return dict(ready=tools["yt-dlp"] and tools["FFmpeg"], tools=tools,
                model_weights_verified=False, missing=[name for name in ("yt-dlp", "FFmpeg") if not tools[name]])


class QueueFull(ValueError): pass


class JobStore:
    def __init__(self, root, lease_seconds=60):
        self.root, self.lease_seconds = Path(root), lease_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, url TEXT NOT NULL, metadata TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, progress INTEGER NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, owner TEXT, lease REAL, result TEXT, error TEXT)")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)
        try:
            db.row_factory = sqlite3.Row
            yield db
            db.commit()
        except Exception:
            db.rollback(); raise
        finally:
            db.close()

    def expire(self, db):
        db.execute("UPDATE jobs SET status='interrupted', stage='interrupted', error='El trabajador perdió su sesión; vuelve a enviar el enlace para reintentar.', updated=? WHERE status='running' AND lease < ?", (time.time(), time.time()))

    def enqueue(self, url, metadata):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE"); self.expire(db)
            existing = db.execute("SELECT id FROM jobs WHERE url=? AND status IN ('queued','running') ORDER BY created LIMIT 1", (url,)).fetchone()
            if existing: job_id = existing[0]
            else:
                from maintenance import check_quota
                active = db.execute("SELECT count(*) FROM jobs WHERE status='running'").fetchone()[0]
                check_quota(self.root, int(os.getenv("SPELLFIND_INGEST_QUOTA_BYTES", str(2*1024**3))),
                            reserve=(active+1)*300*1024**2)
                if db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 32:
                    raise QueueFull("La cola está llena; espera a que terminen los trabajos")
                job_id, now = uuid4().hex, time.time()
                db.execute("INSERT INTO jobs VALUES (?,?,?,'queued','queued',0,?,?,NULL,NULL,NULL,NULL)", (job_id, url, json.dumps(metadata), now, now))
        return self.get(job_id)

    def get(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id): return None
        with self.db() as db:
            self.expire(db)
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None: return None
        data = dict(row)
        data["metadata"] = json.loads(data["metadata"])
        data["result"] = json.loads(data["result"]) if data["result"] else None
        data.pop("owner"); data.pop("lease")
        return data

    def claim(self, owner):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE"); self.expire(db)
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            if not row: return None
            from maintenance import check_quota, StorageLimit
            active = db.execute("SELECT count(*) FROM jobs WHERE status='running'").fetchone()[0]
            try:
                check_quota(self.root, int(os.getenv("SPELLFIND_INGEST_QUOTA_BYTES", str(2*1024**3))),
                            reserve=(active+1)*300*1024**2)
            except StorageLimit as exc:
                db.execute("UPDATE jobs SET status='failed',stage='failed',error=?,updated=? WHERE id=?",
                           (str(exc), time.time(), row[0]))
                return None
            db.execute("UPDATE jobs SET status='running',stage='starting',owner=?,lease=?,updated=? WHERE id=?", (owner, time.time()+self.lease_seconds, time.time(), row[0]))
        return self.get(row[0])

    def assert_owned(self, job_id, owner):
        with self.db() as db:
            row = db.execute("SELECT 1 FROM jobs WHERE id=? AND status='running' AND owner=? AND lease>=?", (job_id, owner, time.time())).fetchone()
        if row is None: raise RuntimeError("La sesión del trabajo ya no es válida")

    def heartbeat(self, job_id, owner):
        with self.db() as db:
            db.execute("UPDATE jobs SET lease=? WHERE id=? AND status='running' AND owner=? AND lease>=?", (time.time()+self.lease_seconds, job_id, owner, time.time()))

    def progress(self, job_id, owner, stage, percent):
        self.assert_owned(job_id, owner)
        with self.db() as db:
            db.execute("UPDATE jobs SET stage=?,progress=?,updated=? WHERE id=? AND owner=? AND status='running'", (stage, percent, time.time(), job_id, owner))

    def finish(self, job_id, owner, status, result=None, error=None):
        with self.db() as db:
            db.execute("UPDATE jobs SET status=?,stage=?,progress=?,result=?,error=?,updated=? WHERE id=? AND owner=? AND status='running'", (status, status, 100 if status=='completed' else 0, json.dumps(result) if result else None, error, time.time(), job_id, owner))


class IngestionService:
    def __init__(self, manager, root, backend=None):
        from ingestion_backend import NativeBackend
        self.manager, self.store = manager, JobStore(root)
        self.backend = backend or NativeBackend()
        self.stop, self.wake = Event(), Event()
        self.owner, self.thread = uuid4().hex, None
        self._start_lock = Lock()

    def status(self): return self.backend.capabilities()

    def start(self):
        with self._start_lock:
            if self.thread is None and self.status()["ready"]:
                self.thread = Thread(target=self._loop, daemon=True, name="spellfind-ingestion")
                self.thread.start()

    def enqueue(self, url, metadata):
        if not self.status()["ready"]: raise RuntimeError("Faltan herramientas de ingesta")
        job = self.store.enqueue(url, metadata)
        self.start(); self.wake.set()
        return job

    def close(self):
        self.stop.set(); self.wake.set()
        if self.thread: self.thread.join(timeout=2)

    def _loop(self):
        while not self.stop.is_set():
            try: job = self.store.claim(self.owner)
            except sqlite3.Error:
                self.wake.wait(1); self.wake.clear(); continue
            if job is None:
                self.wake.wait(1); self.wake.clear(); continue
            self.process(job)

    def process(self, job):
        from resources import use_snapshot
        done = Event()
        def heartbeat():
            while not done.wait(min(10, self.store.lease_seconds/3)):
                try: self.store.heartbeat(job["id"], self.owner)
                except Exception: return
        keeper = Thread(target=heartbeat, daemon=True); keeper.start()
        def guard():
            if self.stop.is_set(): raise RuntimeError("El servicio se está deteniendo; vuelve a intentar")
            self.store.assert_owned(job["id"], self.owner)
            from maintenance import check_quota
            check_quota(self.store.root, int(os.getenv("SPELLFIND_INGEST_QUOTA_BYTES", str(2*1024**3))))
        def progress(stage, percent):
            guard(); self.store.progress(job["id"], self.owner, stage, percent)
        try:
            snapshot = self.manager.current()
            existing = snapshot.find_source(job["url"])
            if existing is not None:
                guard()
                self.store.finish(job["id"], self.owner, "completed", dict(song_id=existing["song_id"], catalog_version=snapshot.version,
                    duplicate=True, note="El vídeo ya existía; no se modificaron sus metadatos."))
                return
            work = self.store.root / "work" / job["id"]
            work.mkdir(parents=True, exist_ok=True)
            with use_snapshot(self.manager.current()):
                metadata, lv, av, audio = self.backend.run(job, work, progress)
            progress("publishing", 90)
            media = self.store.root / "media"; media.mkdir(exist_ok=True)
            target = media / (job["id"] + ".mp3")
            if not audio.is_file(): raise ValueError("No se produjo audio completo")
            shutil.copyfile(audio, target)
            metadata.update(audio_file=target.name, source_url=job["url"])
            snapshot, sid = self.manager.append_song(metadata, lv, av, job_id=job["id"], guard=guard)
            self.store.finish(job["id"], self.owner, "completed", dict(song_id=sid, catalog_version=snapshot.version))
        except Exception as exc:
            self.store.finish(job["id"], self.owner, "failed", error=str(exc)[:500])
        finally:
            done.set(); keeper.join(timeout=1)
