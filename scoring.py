"""Bounded weighted reciprocal ranks, with additive per-song explanations."""
import math

METHOD = "weighted_normalized_rrf_v2"
RRF_K = 60


def hybrid_scores(lyrics, audio, lyric_weight=.6, audio_weight=.4):
    weights = {"lyrics": lyric_weight, "audio": audio_weight}
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in weights.values()) or sum(weights.values()) > 1 + 1e-9:
        raise ValueError("Effective fusion weights must be finite, nonnegative and sum to at most 1")
    parts = {}
    for name, rows in (("lyrics", lyrics), ("audio", audio)):
        seen = set()
        for position, (sid, raw) in enumerate(rows, start=1):
            if sid in seen:
                raise ValueError("Duplicate candidate in retrieval branch")
            seen.add(sid)
            if not math.isfinite(float(raw)):
                raise ValueError("Non-finite worker score")
            if weights[name] == 0:
                continue
            part = dict(rank=position, raw_worker_score=float(raw),
                        rank_score=(RRF_K + 1) / (RRF_K + position), weight=weights[name])
            part["contribution"] = part["weight"] * part["rank_score"]
            parts.setdefault(str(sid), {})[name] = part
    for components in parts.values():
        for name in weights:
            components.setdefault(name, dict(rank=None, raw_worker_score=None, rank_score=0.,
                                              weight=weights[name], contribution=0.))
    scores = [(int(sid), sum(part["contribution"] for part in components.values()))
              for sid, components in parts.items()]
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores, parts
