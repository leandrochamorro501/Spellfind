"""Music-specific agent tools behind the common media boundary.

The legacy lyric/CLAP fields stay inside this adapter while existing clients
continue to receive the v2 music response.
"""
from media_contracts import Capability, condition_record, evidence_record, verdict_record


class MusicModule:
    media_type = "music"
    capabilities = (
        Capability("lyrics", True, True, "verbatim_lyric_excerpt"),
        Capability("audio", True, False, "retrieval_similarity"),
        Capability("metadata", True, True, "catalog_record"),
    )

    def __init__(self, *, planner, search, judge, refiner, lookup, infer_weights):
        self._planner, self._search, self._judge = planner, search, judge
        self._refiner, self._lookup, self._infer_weights = refiner, lookup, infer_weights

    def plan(self, query):
        return self._planner(query)

    def prepare_plan(self, raw, query):
        from intent import prepare_plan
        return prepare_plan(raw, query)

    def fallback_plan(self, query):
        from search_contract import build_fallback_plan
        return build_fallback_plan(query)

    def search(self, **kwargs):
        return self._search(**kwargs)

    def retrieve_plan(self, plan, weights, cache, top_k):
        return self._search(lyric_positive=plan["lyric_positive"],
            lyric_negative=plan["lyric_negative"], clap_positive=plan["clap_positive"],
            clap_negative=plan["clap_negative"], lyric_weight=weights[0],
            audio_weight=weights[1], cache=cache, top_k=top_k)

    def evaluate(self, **kwargs):
        return self._judge(**kwargs)

    def refine(self, **kwargs):
        return self._refiner(**kwargs)

    def lookup(self, item_id):
        return self._lookup(item_id)

    def infer_weights(self, plan):
        return self._infer_weights(plan["lyric_positive"], plan["clap_positive"])

    def requested_aspects(self, plan):
        return dict(primary=bool(plan["lyric_positive"] or plan["lyric_negative"]),
                    secondary=bool(plan["clap_positive"] or plan["clap_negative"]))

    def evaluate_candidate(self, *, query, plan, candidate, item_id, request_id):
        needs = self.requested_aspects(plan)
        validation = dict(lyrics=dict(status="not_evaluated" if needs["primary"] else "not_requested"),
                          audio=dict(status="not_evaluated" if needs["secondary"] else "not_requested",
                                     reason="CLAP retrieval is not independent acoustic validation"))
        if not candidate:
            return dict(verdict=dict(status="rejected", reason="Metadata unavailable", refinable=False),
                        validation=validation, judge_called=False, metadata_skip=True, scope="request")
        if not needs["primary"]:
            return dict(verdict=dict(status="not_evaluated", reason="Acoustic retrieval only", refinable=False),
                        validation=validation, judge_called=False, metadata_skip=False, scope="plan")
        if not candidate.get("lyrics", "").strip():
            reason = "No lyric evidence available"
            validation["lyrics"] = dict(status="abstained", reason=reason)
            return dict(verdict=dict(status="abstained", reason=reason, refinable=False),
                        validation=validation, judge_called=False, metadata_skip=True, scope="plan")
        try:
            verdict = self._judge(query=query, lyrics=candidate["lyrics"],
                lyric_positive=plan["lyric_positive"], lyric_negative=plan["lyric_negative"],
                song_id=item_id, song_title=candidate["title"], request_id=request_id)
            if verdict.get("status") not in {"approved", "rejected", "abstained", "error"}:
                raise ValueError("Invalid judge status")
            if verdict.get("format") == "legacy" and verdict["status"] in {"approved", "rejected"}:
                verdict = dict(verdict, status="abstained", approved=False, refinable=False,
                               reason="Legacy decision cannot validate or semantically reject a candidate")
            validation["lyrics"] = {key: verdict[key] for key in
                ("status", "reason", "evidence", "coverage", "format") if key in verdict}
        except Exception as exc:
            verdict = dict(status="error", reason=str(exc), refinable=False)
            validation["lyrics"] = dict(status="error", reason=str(exc))
        return dict(verdict=verdict, validation=validation, judge_called=True,
                    metadata_skip=False, scope="plan")

    def plan_fingerprint(self, plan):
        from intent import plan_fingerprint
        return plan_fingerprint(plan)

    def validate_refinement(self, current, result):
        from refiner import validate_refinement
        return validate_refinement(current, result)

    def public_plan(self, plan):
        if not plan:
            return None
        conditions = [condition_record(item) for item in plan.get("constraints", [])]
        tools = []
        if plan.get("lyric_positive") or plan.get("lyric_negative"):
            tools.append("lyrics")
        if plan.get("clap_positive") or plan.get("clap_negative"):
            tools.append("audio")
        return dict(media_type=self.media_type, status=plan.get("planning_status", "ready"),
                    conditions=conditions, tools=tools)

    def condition_verdicts(self, plan, validation, *, catalog_version=None):
        """Project the existing holistic lyric judgment without inventing per-condition proof.

        A quote validates a single positive lyric condition only when there is
        exactly one lyric condition. Multiple/negative conditions need a future
        condition-level judge; they remain unverified in this projection.
        """
        if not plan:
            return []
        conditions = [condition_record(item) for item in plan.get("constraints", [])]
        lyric_conditions = [item for item in conditions if item["aspect"] == "lyrics"]
        lyric = validation.get("lyrics", {})
        quotes = lyric.get("evidence", []) if lyric.get("format") == "structured" else []
        result = []
        for item in conditions:
            reason = "La recuperación por audio indica parecido; no hay comprobación acústica independiente."
            evidence = []
            status = "unverified"
            if item["aspect"] == "lyrics":
                reason = "El juez actual evalúa la letra en conjunto; esta condición no tiene una prueba individual suficiente."
                if (len(lyric_conditions) == 1 and item["polarity"] == "positive" and
                        lyric.get("status") == "approved" and quotes and
                        lyric.get("coverage", {}).get("complete") is True):
                    evidence = [evidence_record(kind="lyric_quote", source="judge",
                        excerpt=quote, scope="complete_lyrics", catalog_version=catalog_version) for quote in quotes]
                    status, reason = "meets", "El juez aprobó la única condición lírica y citó la letra completa."
            result.append(dict(verdict_record(item["id"], status, reason, evidence,
                                              scope=item["aspect"]),
                               condition=item["text"], origin=item["origin"], polarity=item["polarity"]))
        return result


def registry(*modules):
    result = {}
    for module in modules:
        if module.media_type in result:
            raise ValueError("Duplicate media type")
        result[module.media_type] = module
    return result
