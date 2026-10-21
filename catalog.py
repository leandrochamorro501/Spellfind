"""Immutable, verified catalog generations; SQLite serializes atomic publication."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
from threading import RLock
from types import MappingProxyType
from uuid import uuid4

import numpy as np
from vector_validation import validate_alignment, validate_index, unit_vector

FILES = ("datos_alineados.csv", "song_id_map.json", "faiss_lyrics.index", "faiss_audio.index")
RECIPE = {"lyrics": "nomic-ai/nomic-embed-text-v1.5:search_document:256:normalized",
          "audio": "laion/clap-htsat-unfused:48000:normalized"}
_UNSET = object()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): result.update(chunk)
    return result.hexdigest()


class ReadIndex:
    """Read-only public surface; stored indexes cannot be add/reset by consumers."""
    __slots__ = ("__index",)

    def __init__(self, index):
        self.__index = index

    @property
    def d(self): return self.__index.d
    @property
    def ntotal(self): return self.__index.ntotal
    @property
    def metric_type(self): return self.__index.metric_type
    @property
    def is_trained(self): return self.__index.is_trained
    def search(self, vectors, k): return self.__index.search(vectors, k)
    def reconstruct(self, pos): return self.__index.reconstruct(pos).copy()
    def reconstruct_n(self, pos, n): return self.__index.reconstruct_n(pos, n).copy()


@dataclass(frozen=True)
class Snapshot:
    version: str
    _manifest_json: str
    _rows: tuple
    _lookup: object
    _mapping: object
    idx_lyrics: ReadIndex
    idx_audio: ReadIndex
    ready: bool = True
    error: object = None
    _modality_maps: object = None

    @property
    def manifest(self): return json.loads(self._manifest_json)

    @property
    def df_meta(self):
        import pandas as pd
        return pd.DataFrame([dict(row) for row in self._rows])

    @property
    def song_id_map(self): return dict(self._mapping)

    def mapping_for(self, modality):
        if modality not in {"lyrics", "audio"}: raise ValueError("Unknown modality")
        return dict(self._modality_maps[modality]) if self._modality_maps is not None else self.song_id_map

    def get_song_by_id(self, sid):
        row = self._lookup.get(sid)
        return dict(row) if row is not None else None

    def metadata_rows(self):
        return [{key: row.get(key, "") for key in ("song_id", "artist", "tag", "year")} for row in self._rows]

    def find_source(self, url):
        return next((dict(row) for row in self._rows if row.get("source_url") == url), None)


def read_bundle(path, version="legacy", manifest=None):
    import faiss
    import pandas as pd
    path = Path(path)
    frame = pd.read_csv(path / FILES[0], dtype=str).fillna("")
    if not {"song_id", "title", "artist", "lyrics"}.issubset(frame.columns):
        raise ValueError("Catalog missing required columns")
    ids = pd.to_numeric(frame["song_id"], errors="raise")
    if not np.isfinite(ids).all() or (ids < 0).any() or (ids % 1 != 0).any():
        raise ValueError("Catalog IDs must be nonnegative finite integers")
    frame["song_id"] = ids.astype("int64")
    mapping = json.loads((path / FILES[1]).read_text(encoding="utf-8"))
    indexes = [faiss.read_index(str(path / name)) for name in FILES[2:]]
    separate = isinstance(mapping, dict) and mapping.get("schema_version") == 2
    for mode, index in zip(("lyrics", "audio"), indexes): validate_index(index, mode, allow_empty=separate)
    modality_maps = None
    if separate:
        all_ids = frame["song_id"].tolist()
        if len(set(all_ids)) != len(all_ids) or mapping.get("catalog") != {str(i): sid for i, sid in enumerate(all_ids)}:
            raise ValueError("Invalid catalog map")
        modality_maps = {}
        for mode, index in zip(("lyrics", "audio"), indexes):
            positions = mapping.get(mode)
            if not isinstance(positions, dict) or set(positions) != {str(i) for i in range(index.ntotal)}:
                raise ValueError("Invalid modality positions")
            values = list(positions.values())
            if any(type(sid) is not int or sid not in all_ids for sid in values) or len(set(values)) != len(values):
                raise ValueError("Invalid modality song IDs")
            modality_maps[mode] = MappingProxyType(positions)
        if set(modality_maps["audio"].values()) != set(all_ids): raise ValueError("Every song requires audio")
        for row in frame.to_dict(orient="records"):
            has_lyrics = row["song_id"] in modality_maps["lyrics"].values()
            if has_lyrics != bool(str(row["lyrics"]).strip()): raise ValueError("Lyrics availability/map mismatch")
        mapping = mapping["catalog"]
    else:
        validate_alignment(frame, mapping, *indexes)
    rows = tuple(MappingProxyType({key: int(value) if key == "song_id" else str(value)
                                  for key, value in row.items()}) for row in frame.to_dict(orient="records"))
    return Snapshot(version, json.dumps(manifest or {}), rows,
                    MappingProxyType({row["song_id"]: row for row in rows}), MappingProxyType(mapping),
                    *(ReadIndex(index) for index in indexes), _modality_maps=modality_maps)


class CatalogManager:
    def __init__(self, root, legacy):
        self.root, self.legacy = Path(root), Path(legacy)
        self._lock, self._cached = RLock(), None

    @contextmanager
    def transaction(self):
        self.root.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.root / "registry.sqlite3", timeout=60)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS current (singleton INTEGER PRIMARY KEY CHECK(singleton=1), version TEXT NOT NULL)")
            history_existed = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='history'").fetchone() is not None
            db.execute("CREATE TABLE IF NOT EXISTS history (version TEXT PRIMARY KEY, created REAL NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            if not history_existed:
                db.execute("INSERT OR IGNORE INTO history SELECT version, ? FROM current", (time.time(),))
            yield db
            db.commit()
        except Exception:
            db.rollback(); raise
        finally:
            db.close()

    def _load(self, version):
        if not isinstance(version, str) or len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
            raise ValueError("Invalid catalog generation")
        directory = self.root / "versions" / version
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != version or set(manifest.get("sha256", {})) != set(FILES):
            raise ValueError("Invalid catalog manifest")
        if any(digest(directory / name) != sha for name, sha in manifest["sha256"].items()):
            raise ValueError("Catalog generation hash mismatch")
        if manifest.get("recipe") != RECIPE:
            raise ValueError("Incompatible embedding recipe")
        return read_bundle(directory, version, manifest)

    def _publish(self, db, writer, guard=None):
        from maintenance import check_quota
        quota = int(os.getenv("SPELLFIND_CATALOG_QUOTA_BYTES", str(512*1024**2)))
        check_quota(self.root, quota)
        version = uuid4().hex
        stage = self.root / "staging" / version
        stage.mkdir(parents=True)
        writer(stage)
        check_quota(self.root, quota)
        read_bundle(stage)  # validation before changing the registry
        manifest = dict(schema_version=1, version=version, recipe=RECIPE,
                        local_ingestion_aggregation="lyrics_chunks2000_mean;audio_start_middle_end10s_mean",
                        provenance="Imported legacy embeddings have unverified upstream revisions; recipe describes local ingestion.",
                        sha256={name: digest(stage / name) for name in FILES})
        from config import MODEL_REVISIONS
        manifest["local_model_revisions"] = MODEL_REVISIONS
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        final = self.root / "versions" / version
        final.parent.mkdir(exist_ok=True)
        stage.rename(final)
        candidate = self._load(version)
        if guard is not None: guard()
        db.execute("INSERT INTO history VALUES (?, ?)", (version, time.time()))
        db.execute("INSERT OR REPLACE INTO current VALUES (1, ?)", (version,))
        return candidate

    def current(self):
        with self._lock:
            with self.transaction() as db:
                row = db.execute("SELECT version FROM current WHERE singleton=1").fetchone()
                if row is None:
                    def copy(stage):
                        before = {name: digest(self.legacy / name) for name in FILES}
                        for name in FILES: shutil.copyfile(self.legacy / name, stage / name)
                        if any(digest(stage / name) != before[name] or digest(self.legacy / name) != before[name] for name in FILES):
                            raise ValueError("Legacy files changed while importing")
                    candidate = self._publish(db, copy)
                elif self._cached is not None and self._cached.version == row[0]:
                    return self._cached
                else:
                    candidate = self._load(row[0])
            self._cached = candidate  # only after commit
            return candidate

    def at_version(self, version):
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM history WHERE version=?", (version,)).fetchone() is None:
                raise ValueError("Catalog generation was not published")
        if self._cached is not None and self._cached.version == version: return self._cached
        return self._load(version)

    def published_version(self):
        with self.transaction() as db:
            row = db.execute("SELECT version FROM current WHERE singleton=1").fetchone()
        return row[0] if row else None

    def publish_directory(self, source, expected_version=_UNSET):
        source = Path(source)
        with self._lock:
            with self.transaction() as db:
                row = db.execute("SELECT version FROM current WHERE singleton=1").fetchone()
                if expected_version is not _UNSET and expected_version != (row[0] if row else None):
                    raise ValueError("Catalog changed during rebuild; publication refused")
                candidate = self._publish(db, lambda stage: [shutil.copyfile(source / name, stage / name) for name in FILES])
            self._cached = candidate
            return candidate

    def append_song(self, metadata, lyrics_vector, audio_vector, *, job_id, guard=None):
        import faiss
        import pandas as pd
        has_lyrics = bool(metadata.get("lyrics", "").strip())
        if has_lyrics != (lyrics_vector is not None): raise ValueError("Lyrics text/vector availability mismatch")
        lv, av = unit_vector(lyrics_vector, 256) if has_lyrics else None, unit_vector(audio_vector, 512)
        if not metadata.get("title", "").strip(): raise ValueError("A complete title is required")
        self.current()
        with self._lock:
            with self.transaction() as db:
                version = db.execute("SELECT version FROM current WHERE singleton=1").fetchone()[0]
                previous = self._load(version)
                duplicate = next((row for row in previous._rows if row.get("ingest_job") == job_id or
                    metadata.get("source_url") and row.get("source_url") == metadata["source_url"]), None)
                if duplicate:
                    return previous, int(duplicate["song_id"])
                sid = max(previous._lookup) + 1
                record = {**{key: str(value) for key, value in metadata.items()}, "song_id": sid, "ingest_job": job_id}
                def write(stage):
                    rows = [dict(row) for row in previous._rows] + [record]
                    pd.DataFrame(rows).to_csv(stage / FILES[0], index=False)
                    maps = dict(schema_version=2, catalog={str(i): row["song_id"] for i, row in enumerate(rows)})
                    for mode, name, old, new in zip(("lyrics", "audio"), FILES[2:], (previous.idx_lyrics, previous.idx_audio), (lv, av)):
                        positions = previous.mapping_for(mode)
                        if new is not None: positions[str(len(positions))] = sid
                        maps[mode] = positions
                        index = faiss.IndexFlatIP(old.d)
                        vectors = old.reconstruct_n(0, old.ntotal)
                        if new is not None: vectors = np.vstack([vectors, new])
                        if len(vectors): index.add(vectors)
                        faiss.write_index(index, str(stage / name))
                    (stage / FILES[1]).write_text(json.dumps(maps), encoding="utf-8")
                candidate = self._publish(db, write, guard)
            self._cached = candidate
            return candidate, sid


_default = None
_default_lock = RLock()


def default_manager():
    global _default
    from config import CATALOG_DIR, OUTPUT_DIR
    with _default_lock:
        if _default is None: _default = CatalogManager(CATALOG_DIR, OUTPUT_DIR)
        return _default
