"""Pinned per-request snapshots; legacy unpinned helpers keep static lazy caches."""
import json
from functools import lru_cache
from threading import RLock
from contextvars import ContextVar
from contextlib import contextmanager
from config import OUTPUT_DIR
from vector_validation import validate_index, validate_alignment

_lock = RLock()
_snapshot = ContextVar("spellfind_catalog_snapshot", default=None)


@contextmanager
def use_snapshot(snapshot):
    token = _snapshot.set(snapshot)
    try:
        yield snapshot
    finally:
        _snapshot.reset(token)


def search_resources(model_resources):
    snapshot = _snapshot.get()
    if snapshot is None: return model_resources
    index = snapshot.idx_lyrics if "embedder" in model_resources else snapshot.idx_audio
    mode = "lyrics" if "embedder" in model_resources else "audio"
    return {**model_resources, "index": index, "song_id_map": snapshot.mapping_for(mode)}


@lru_cache(maxsize=1)
def _catalog():
    import pandas as pd
    frame = pd.read_csv(OUTPUT_DIR / "datos_alineados.csv").fillna("")
    required = {"song_id", "title", "artist", "lyrics"}
    if not required.issubset(frame.columns):
        raise ValueError("Catalog missing required metadata columns")
    import numpy as np
    ids = pd.to_numeric(frame["song_id"], errors="raise")
    if not np.isfinite(ids).all() or (ids < 0).any() or (ids % 1 != 0).any():
        raise ValueError("Catalog IDs must be finite non-negative integers")
    frame["song_id"] = ids.astype("int64")
    if frame["song_id"].duplicated().any():
        raise ValueError("Duplicate song IDs in catalog")
    return frame


def get_catalog():
    if _snapshot.get() is not None: return _snapshot.get().df_meta
    with _lock:
        return _catalog()


@lru_cache(maxsize=1)
def _song_id_map():
    with (OUTPUT_DIR / "song_id_map.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def get_song_id_map(modality=None):
    if _snapshot.get() is not None:
        return _snapshot.get().mapping_for(modality) if modality else _snapshot.get().song_id_map
    with _lock:
        return _song_id_map()


@lru_cache(maxsize=2)
def _index(modality):
    import faiss
    if modality not in {"lyrics", "audio"}:
        raise ValueError("Unknown index modality")
    index = faiss.read_index(str(OUTPUT_DIR / f"faiss_{modality}.index"))
    validate_index(index, modality)
    return index


def get_index(modality):
    if _snapshot.get() is not None:
        if modality not in {"lyrics", "audio"}: raise ValueError("Unknown modality")
        return _snapshot.get().idx_lyrics if modality == "lyrics" else _snapshot.get().idx_audio
    with _lock:
        return _index(modality)


def validate_bundle():
    if _snapshot.get() is not None: return
    with _lock:
        validate_alignment(get_catalog(), get_song_id_map(),
                           get_index("lyrics"), get_index("audio"))


@lru_cache(maxsize=1)
def _metadata():
    return get_catalog().set_index("song_id").to_dict(orient="index")


def get_song(song_id):
    if _snapshot.get() is not None: return _snapshot.get().get_song_by_id(song_id)
    with _lock:
        row = _metadata().get(song_id)
    if row is None:
        return None
    return {"song_id": int(song_id), **{
        key: str(row.get(key, "")) for key in
        ("title", "artist", "tag", "year", "lyrics")
    }}
