"""
semantic_search.py — SPELLFIND · Worker Lírico
=====================================================
Búsqueda semántica sobre el índice FAISS de letras (Nomic embeddings).
Implementa el motor ortogonal: embedding positivo − negativos, renormalizado.
Aplica multi-query + RRF interno antes de devolver resultados a app.py.

Interfaz pública:
    search_lyrics(positive: list[str], negative: list[str], top_k: int)
        → [(song_id, rrf_score), ...]
"""

from console import safe_print as print
from profiling import phase, add
from time import perf_counter
import numpy as np
from vector_validation import unit_vector

from collections import defaultdict
from functools import lru_cache
from threading import RLock
from resources import get_index, get_song_id_map, validate_bundle, search_resources
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

NOMIC_MODEL  = "nomic-ai/nomic-embed-text-v1.5"
RRF_K        = 60

# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON — carga pesada una sola vez
# ──────────────────────────────────────────────────────────────────────────────

_resources = None
_model_lock = RLock()

def _load_resources() -> dict:
    with _model_lock:
        return _load_resources_unlocked()


def _load_resources_unlocked() -> dict:
    global _resources
    if _resources is not None:
        return _resources

    validate_bundle()
    index = get_index("lyrics")
    song_id_map = get_song_id_map("lyrics")
    print("[SEMANTIC] Cargando modelo Nomic...")
    from config import MODEL_REVISIONS
    embedder = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True, revision=MODEL_REVISIONS["lyrics"],
        model_kwargs={"code_revision": MODEL_REVISIONS["lyrics_code"]},
        config_kwargs={"code_revision": MODEL_REVISIONS["lyrics_code"]})

    _resources = {
        "embedder":    embedder,
        "index":       index,
        "song_id_map": song_id_map,
    }
    print("[SEMANTIC] ✅ Worker lírico listo")
    return _resources


# ──────────────────────────────────────────────────────────────────────────────
# MOTOR ORTOGONAL
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _encode_cached(text, embedder):
    return unit_vector(embedder.encode(f"search_query: {text}", normalize_embeddings=True))


def _encode_text(text, embedder):
    started = perf_counter()
    with _model_lock:
        add("encoder_lock_wait_ms", (perf_counter()-started)*1000)
        hits = _encode_cached.cache_info().hits
        # Query arithmetic must never mutate a shared cached embedding.
        with phase("encoding_ms"):
            vector = _encode_cached(text, embedder).copy()
        add("embedding_cache_hits" if _encode_cached.cache_info().hits > hits else "embedding_cache_misses")
        return vector


def _build_query_vector(
    positive_text:  str,
    negative_texts: list[str],
    embedder:       SentenceTransformer,
) -> np.ndarray:
    vec = _encode_text(positive_text, embedder)
    for neg_text in negative_texts:
        vec = vec - _encode_text(neg_text, embedder)

    # ── Truncar a 256-dim (Matryoshka) y renormalizar ──
    if vec.ndim != 1 or vec.size < 256 or not np.isfinite(vec).all():
        raise ValueError("Invalid Nomic embedding shape or values")
    vec = vec[:256]
    return unit_vector(vec, 256)

# ──────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA MULTI-QUERY RAW
# ──────────────────────────────────────────────────────────────────────────────

def _multi_query_search_raw(
    lyric_queries:   list[str],
    lyric_negatives: list[str],
    top_k:           int,
    resources:       dict,
) -> list[dict]:
    """
    Ejecuta una búsqueda FAISS por cada query positiva y acumula
    los resultados crudos con su origen para el RRF posterior.
    """
    embedder    = resources["embedder"]
    index       = resources["index"]
    song_id_map = resources["song_id_map"]

    all_raw = []

    for i, query_text in enumerate(lyric_queries):
        q_vec = _build_query_vector(query_text, lyric_negatives, embedder)
        
        #print(f"[SEMANTIC DEBUG] q_vec.shape={q_vec.shape}  index.d={index.d}")
        q_vec = unit_vector(q_vec, index.d)
        with phase("index_search_ms"):
            scores, indices = index.search(q_vec[np.newaxis, :], top_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            # Traducir posición FAISS → song_id
            song_id = int(song_id_map.get(str(idx), -1))
            if song_id == -1:
                continue

            all_raw.append({
                "song_id":      song_id,
                "score":        float(score),
                "query_origin": f"lyric_{i + 1}",
            })

    return all_raw


# ──────────────────────────────────────────────────────────────────────────────
# RRF INTERNO (entre múltiples queries de la misma rama)
# ──────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    all_raw: list[dict],
    k:       int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fusiona los resultados de múltiples queries líricas en un único ranking.
    Devuelve [(song_id, rrf_score), ...] ordenado descendente.
    """
    # Agrupar por query de origen
    by_query: dict[str, list[dict]] = defaultdict(list)
    for item in all_raw:
        by_query[item["query_origin"]].append(item)

    # Ordenar cada grupo por score coseno descendente
    for q in by_query:
        by_query[q].sort(key=lambda x: x["score"], reverse=True)

    # Acumular score RRF por song_id
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked_list in by_query.values():
        for rank, item in enumerate(ranked_list, start=1):
            rrf_scores[item["song_id"]] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def search_lyrics(
    positive: list[str],
    negative: list[str],
    top_k:    int = 10,
) -> list[tuple[int, float]]:
    """
    Punto de entrada para app.py.

    Args:
        positive: lista de queries líricas positivas (del router)
        negative: lista de conceptos a sustraer ortogonalmente
        top_k:    candidatos por query antes de RRF

    Returns:
        [(song_id, rrf_score), ...] ordenado por relevancia descendente
    """
    if not positive:
        return []

    # Filtrar strings vacíos
    positive = [q.strip() for q in positive if q.strip()]
    negative = [n.strip() for n in negative if n.strip()]

    if not positive:
        return []

    with phase("resources_ms"):
        resources = search_resources(_load_resources())

    print(f"[SEMANTIC] Buscando: {len(positive)} queries positivas, "
          f"{len(negative)} negativas, top_k={top_k}")

    all_raw = _multi_query_search_raw(
        lyric_queries=positive,
        lyric_negatives=negative,
        top_k=top_k * 3,   # margen amplio antes del RRF
        resources=resources,
    )

    with phase("internal_rrf_ms"):
        fused = _reciprocal_rank_fusion(all_raw)
    print(f"[SEMANTIC] → {len(fused)} candidatos tras RRF interno")

    return fused[:top_k]
