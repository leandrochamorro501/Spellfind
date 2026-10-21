"""Numerical invariants for cosine retrieval, independent of model libraries."""
import numpy as np

DIMENSIONS = {"lyrics": 256, "audio": 512}


def unit_vector(value, dimension=None):
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or (dimension is not None and vector.size != dimension):
        raise ValueError(f"Expected a one-dimensional vector of dimension {dimension}")
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("Embedding is empty or contains non-finite values")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding has zero or degenerate norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def validate_index(index, modality, allow_empty=False):
    dimension = DIMENSIONS[modality]
    if index.d != dimension or index.ntotal < (0 if allow_empty else 1):
        raise ValueError(f"Invalid {modality} index dimension or empty index")
    # FAISS METRIC_INNER_PRODUCT == 0. Distances cannot be treated as cosine ranks.
    if index.metric_type != 0 or not index.is_trained:
        raise ValueError(f"{modality} index must be trained and use inner product")
    for start in range(0, index.ntotal, 4096):
        vectors = np.asarray(index.reconstruct_n(start, min(4096, index.ntotal - start)))
        if vectors.ndim != 2 or vectors.shape[1] != dimension or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid vectors in {modality} index")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.allclose(norms, 1, rtol=1e-3, atol=1e-3):
            raise ValueError(f"{modality} index contains zero or non-unit vectors")


def validate_alignment(frame, mapping, lyrics, audio):
    count = len(frame)
    if not count or lyrics.ntotal != count or audio.ntotal != count:
        raise ValueError("Catalog and index counts do not match")
    if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(count)}:
        raise ValueError("Song map must cover every index position exactly")
    ids = frame["song_id"].tolist()
    if len(set(ids)) != count:
        raise ValueError("Duplicate catalog song IDs")
    for position, expected in enumerate(ids):
        actual = mapping[str(position)]
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            raise ValueError(f"Song map/catalog mismatch at index position {position}")
