"""JSON-safe contracts shared by media modules and their agents.

Retrieval is a lead, not evidence that a condition is true.  A module must
declare which aspects it can search and independently validate.
"""
from dataclasses import dataclass
from typing import Protocol


STATUSES = frozenset({"meets", "contradicts", "unverified"})
ORIGINS = frozenset({"user_explicit", "model_hypothesis"})


@dataclass(frozen=True)
class Capability:
    aspect: str
    searchable: bool
    verifiable: bool
    evidence_kind: str

    def as_dict(self):
        return dict(aspect=self.aspect, searchable=self.searchable,
                    verifiable=self.verifiable, evidence_kind=self.evidence_kind)


def condition_record(item):
    """Normalize a model-supplied condition without upgrading its provenance."""
    if not isinstance(item, dict):
        raise ValueError("Condition must be an object")
    aspect = item.get("aspect", item.get("modality"))
    polarity = item.get("polarity")
    text = item.get("text")
    origin = item.get("origin", "model_hypothesis")
    quote = item.get("source_quote", "")
    if not isinstance(aspect, str) or not aspect.strip() or polarity not in {"positive", "negative"}:
        raise ValueError("Invalid condition aspect or polarity")
    if not isinstance(text, str) or not text.strip() or origin not in ORIGINS or not isinstance(quote, str):
        raise ValueError("Invalid condition text or origin")
    if origin == "user_explicit" and not quote.strip():
        raise ValueError("An explicit condition needs a source quote")
    return dict(id=str(item.get("id", "")), aspect=aspect.strip(), polarity=polarity,
                text=text.strip(), origin=origin, source_quote=quote)


def evidence_record(*, kind, source, excerpt=None, scope=None, catalog_version=None):
    if not isinstance(kind, str) or not kind or not isinstance(source, str) or not source:
        raise ValueError("Evidence requires kind and source")
    if excerpt is not None and not isinstance(excerpt, str):
        raise ValueError("Evidence excerpt must be text")
    return dict(kind=kind, source=source, excerpt=excerpt, scope=scope,
                catalog_version=catalog_version)


def verdict_record(condition_id, status, reason, evidence=(), *, scope=None):
    if status not in STATUSES or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Invalid condition verdict")
    if status != "unverified" and not evidence:
        raise ValueError("A decision requires evidence")
    return dict(condition_id=condition_id, status=status, reason=reason.strip(),
                evidence=list(evidence), scope=scope)


class MediaModule(Protocol):
    media_type: str
    capabilities: tuple[Capability, ...]

    def plan(self, query: str) -> dict: ...
    def prepare_plan(self, raw: dict, query: str) -> dict: ...
    def fallback_plan(self, query: str) -> dict: ...
    def retrieve_plan(self, plan: dict, weights: tuple, cache: dict, top_k: int): ...
    def requested_aspects(self, plan: dict) -> dict: ...
    def evaluate_candidate(self, *, query: str, plan: dict, candidate: dict,
                           item_id: int, request_id: str) -> dict: ...
    def refine(self, **kwargs) -> dict: ...
    def lookup(self, item_id: int): ...
    def infer_weights(self, plan: dict) -> tuple[float, float]: ...
    def plan_fingerprint(self, plan: dict) -> str: ...
    def validate_refinement(self, current: dict, result: dict) -> dict: ...
    def condition_verdicts(self, plan: dict, validation: dict, *, catalog_version=None) -> list[dict]: ...
