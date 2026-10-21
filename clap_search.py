"""
clap_search.py — SPELLFIND · Worker Acústico
==================================================
Búsqueda semántica sobre el índice FAISS de audio (CLAP embeddings).
Usa la rama texto→audio de CLAP: convierte descripciones textuales
al mismo espacio latente que los embeddings de audio del corpus.
Aplica el mismo motor ortogonal y RRF interno que el worker lírico.

Modelo: laion/clap-htsat-unfused  (mismo que generó los embeddings)

Interfaz pública:
    search_audio(positive: list[str], negative: list[str], top_k: int)
        → [(song_id, rrf_score), ...]
"""

from console import safe_print as print
from profiling import phase, add
from time import perf_counter
import numpy as np
from vector_validation import unit_vector
import torch

from collections import defaultdict
from functools import lru_cache
from threading import RLock
from resources import get_index, get_song_id_map, validate_bundle, search_resources
from transformers import ClapModel, ClapProcessor

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

CLAP_MODEL  = "laion/clap-htsat-unfused"
RRF_K       = 60
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

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
    index = get_index("audio")
    song_id_map = get_song_id_map("audio")
    print(f"[CLAP] Cargando modelo CLAP en {DEVICE}...")
    from config import MODEL_REVISIONS
    model = ClapModel.from_pretrained(CLAP_MODEL, revision=MODEL_REVISIONS["audio"]).to(DEVICE)
    model.eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL, revision=MODEL_REVISIONS["audio"])
    print("[CLAP] Modelo cargado")

    _resources = {
        "model":       model,
        "processor":   processor,
        "index":       index,
        "song_id_map": song_id_map,
    }
    print("[CLAP] ✅ Worker acústico listo")
    return _resources


# ──────────────────────────────────────────────────────────────────────────────
# ENCODER DE TEXTO → ESPACIO CLAP
# ──────────────────────────────────────────────────────────────────────────────
def _encode_text(text: str, model: ClapModel, processor: ClapProcessor) -> np.ndarray:
    started = perf_counter()
    with _model_lock:
        add("encoder_lock_wait_ms", (perf_counter()-started)*1000)
        hits = _encode_cached.cache_info().hits
        with phase("encoding_ms"):
            vector = _encode_cached(text, model, processor).copy()
        add("embedding_cache_hits" if _encode_cached.cache_info().hits > hits else "embedding_cache_misses")
        return vector


@lru_cache(maxsize=512)
def _encode_cached(text: str, model: ClapModel, processor: ClapProcessor) -> np.ndarray:
    inputs = processor(text=text, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        output = model.get_text_features(**inputs)

    # Debug: mostrar tipo y atributos disponibles
    #print(f"[CLAP DEBUG] type(output): {type(output)}")

    # Extraer tensor según lo que retorne
    if isinstance(output, torch.Tensor):
        vec_tensor = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        vec_tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        vec_tensor = output.last_hidden_state[:, 0, :]  # CLS token
    else:
        # Último recurso: primer elemento si es iterable
        vec_tensor = output[0]
        #print(f"[CLAP DEBUG] Usando output[0], shape: {vec_tensor.shape}")

    #print(f"[CLAP DEBUG] vec_tensor.shape: {vec_tensor.shape}")

    vec = vec_tensor.squeeze().cpu().float().numpy()
    return unit_vector(vec, 512)

# ──────────────────────────────────────────────────────────────────────────────
# MOTOR ORTOGONAL CLAP
# ──────────────────────────────────────────────────────────────────────────────

def _build_query_vector(
    positive_text:  str,
    negative_texts: list[str],
    model:          ClapModel,
    processor:      ClapProcessor,
) -> np.ndarray:
    """
    Vector de consulta con rechazo ortogonal en el espacio CLAP:
        vec = encode(positive) − Σ encode(negative_i)
    Renormalizado a norma unitaria.
    """
    vec = _encode_text(positive_text, model, processor)

    for neg_text in negative_texts:
        neg_vec = _encode_text(neg_text, model, processor)
        vec = vec - neg_vec

    return unit_vector(vec, 512)


# ──────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA MULTI-QUERY RAW
# ──────────────────────────────────────────────────────────────────────────────

def _multi_query_search_raw(
    clap_queries:   list[str],
    clap_negatives: list[str],
    top_k:          int,
    resources:      dict,
) -> list[dict]:
    """
    Ejecuta una búsqueda FAISS por cada query acústica positiva.
    """
    model       = resources["model"]
    processor   = resources["processor"]
    index       = resources["index"]
    song_id_map = resources["song_id_map"]

    all_raw = []

    for i, query_text in enumerate(clap_queries):
        q_vec = _build_query_vector(query_text, clap_negatives, model, processor)
        q_vec = unit_vector(q_vec, index.d)
        with phase("index_search_ms"):
            scores, indices = index.search(q_vec[np.newaxis, :], top_k)

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            song_id = int(song_id_map.get(str(idx), -1))
            if song_id == -1:
                continue

            all_raw.append({
                "song_id":      song_id,
                "score":        float(score),
                "query_origin": f"clap_{i + 1}",
            })

    return all_raw


# ──────────────────────────────────────────────────────────────────────────────
# RRF INTERNO
# ──────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    all_raw: list[dict],
    k:       int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fusiona resultados de múltiples queries acústicas en un único ranking.
    Devuelve [(song_id, rrf_score), ...] ordenado descendente.
    """
    by_query: dict[str, list[dict]] = defaultdict(list)
    for item in all_raw:
        by_query[item["query_origin"]].append(item)

    for q in by_query:
        by_query[q].sort(key=lambda x: x["score"], reverse=True)

    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked_list in by_query.values():
        for rank, item in enumerate(ranked_list, start=1):
            rrf_scores[item["song_id"]] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ PÚBLICA
# ──────────────────────────────────────────────────────────────────────────────

def search_audio(
    positive: list[str],
    negative: list[str],
    top_k:    int = 10,
) -> list[tuple[int, float]]:
    """
    Punto de entrada para app.py.

    Args:
        positive: lista de descripciones acústicas positivas (del router)
        negative: lista de conceptos acústicos a sustraer
        top_k:    candidatos por query antes de RRF

    Returns:
        [(song_id, rrf_score), ...] ordenado por relevancia descendente
    """
    if not positive:
        return []

    positive = [q.strip() for q in positive if q.strip()]
    negative = [n.strip() for n in negative if n.strip()]

    if not positive:
        return []

    with phase("resources_ms"):
        resources = search_resources(_load_resources())

    print(f"[CLAP] Buscando: {len(positive)} queries positivas, "
          f"{len(negative)} negativas, top_k={top_k}")

    all_raw = _multi_query_search_raw(
        clap_queries=positive,
        clap_negatives=negative,
        top_k=top_k * 3,
        resources=resources,
    )

    with phase("internal_rrf_ms"):
        fused = _reciprocal_rank_fusion(all_raw)
    print(f"[CLAP] → {len(fused)} candidatos tras RRF interno")

    return fused[:top_k]
