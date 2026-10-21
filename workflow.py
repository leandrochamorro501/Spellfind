"""Request-local workflow with explicit degradation and independent budgets."""
from copy import deepcopy
from time import perf_counter
from typing import TypedDict
from search_contract import build_fallback_plan
from intent import prepare_plan, plan_fingerprint


class SearchState(TypedDict, total=False):
    query: str
    request_id: str
    plan: dict
    initial_plan: dict
    plan_version: int
    exclusions: list
    ranked: list
    candidates: list
    candidate: dict
    iterations: int
    judge_log: list
    refinement_log: list
    trace: list
    status: str
    error_stage: str
    error_reason: str
    verdict: dict
    approved_song: int | None
    selected_song: int | None
    approved_item: int | None
    selected_item: int | None
    validation: dict
    lyric_weight: float
    audio_weight: float
    n_lyrics: int
    n_audio: int
    retrieval: dict
    degradations: list
    counters: dict
    budgets: dict
    budget_reason: str
    requested_top_k: int
    last_success: dict
    refinement_disabled: bool
    node_timings: list
    retrieval_history: list


def active_exclusions(state):
    return {row["song_id"] for row in state["exclusions"]
            if row["scope"] == "request" or row["plan_version"] == state["plan_version"]}


def build_workflow(*, planner=None, search=None, judge=None, refiner=None, lookup=None, infer_weights=None,
                   module=None,
                   max_iterations=3, max_candidates=20, max_refinements=2,
                   max_expansions=1, initial_top_k=10, enable_refinement=True):
    from langgraph.graph import StateGraph, START, END

    if module is not None:
        if any(value is not None for value in (planner, search, judge, refiner, lookup, infer_weights)):
            raise ValueError("Pass either a media module or legacy callbacks")
        planner, search, judge, refiner, lookup = (
            module.plan, module.retrieve_plan, module.evaluate_candidate, module.refine, module.lookup)
        infer_weights = lambda lyrics, audio: module.infer_weights(
            dict(lyric_positive=lyrics, clap_positive=audio))
    if any(value is None for value in (planner, search, judge, refiner, lookup, infer_weights)):
        raise ValueError("Missing workflow agent or tool")
    for value in (max_iterations, max_candidates, initial_top_k):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Budgets must be positive integers")
    for value in (max_refinements, max_expansions):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Optional budgets must be nonnegative integers")
    retrieval_cache = {}
    budgets = dict(judge_evaluations=max_iterations, candidate_visits=max_candidates,
                   refinements=max_refinements, expansions=max_expansions,
                   judge_model_calls=max_iterations * 2,
                   max_top_k=initial_top_k * 2 ** max_expansions)

    def event(state, node, **details):
        return state.get("trace", []) + [dict(node=node, plan_version=state.get("plan_version", 1), **details)]

    def degradation(state, stage, message, **details):
        return state.get("degradations", []) + [dict(stage=stage, message=message, **details)]

    def plan_node(state):
        try:
            raw = planner(state["query"])
            plan = module.prepare_plan(raw, state["query"]) if module is not None else prepare_plan(raw, state["query"])
        except Exception as exc:
            plan = module.fallback_plan(state["query"]) if module is not None else build_fallback_plan(state["query"])
            plan["_fallback_reason"] = str(exc)
        status = "clarification_needed" if plan["planning_status"] == "needs_clarification" else "running"
        return dict(plan=plan, initial_plan=deepcopy(plan), status=status, budgets=budgets,
                    requested_top_k=initial_top_k,
                    trace=event(state, "plan", fallback=bool(plan.get("_fallback")), status=status))

    def search_node(state):
        p = state["plan"]
        lw, aw = module.infer_weights(p) if module is not None else infer_weights(p["lyric_positive"], p["clap_positive"])
        counts = dict(state["counters"])
        counts["retrieval_rounds"] += 1
        updates = dict(counters=counts)
        try:
            batch = (module.retrieve_plan(p, (lw, aw), retrieval_cache, state["requested_top_k"])
                     if module is not None else
                     search(lyric_positive=p["lyric_positive"], lyric_negative=p["lyric_negative"],
                            clap_positive=p["clap_positive"], clap_negative=p["clap_negative"],
                            lyric_weight=lw, audio_weight=aw, cache=retrieval_cache,
                            top_k=state["requested_top_k"]))
            lyrics, audio, ranked = batch
            info = getattr(batch, "details", {})
            updates["_attempt_diagnostics"] = info
            weights = info.get("effective_weights", dict(lyrics=lw, audio=aw))
            updates.update(retrieval=info, lyric_weight=weights["lyrics"], audio_weight=weights["audio"],
                           n_lyrics=len(lyrics), n_audio=len(audio))
            if info.get("degraded"):
                updates["degradations"] = degradation(state, "search",
                    "Recuperación parcial: una modalidad falló; pesos efectivos ajustados.", details=info)
            ranked = [[int(sid), float(score)] for sid, score in ranked]
            updates["last_success"] = dict(plan=deepcopy(p), plan_version=state["plan_version"],
                ranked=ranked, retrieval=info, lyric_weight=weights["lyrics"], audio_weight=weights["audio"],
                n_lyrics=len(lyrics), n_audio=len(audio))
        except Exception as exc:
            updates["_attempt_diagnostics"] = getattr(exc, "details", {"error": str(exc)})
            previous = state.get("last_success")
            if not previous:
                return dict(updates, status="error", error_stage="search", error_reason=str(exc),
                            ranked=[], candidates=[], retrieval=getattr(exc, "details", {}),
                            trace=event(state, "search", status="error", reason=str(exc)))
            # A ranking can be reused only together with the plan that produced it.
            updates.update(deepcopy(previous), refinement_disabled=True)
            updates["degradations"] = degradation(state, "search",
                "Falló la nueva recuperación; se conserva el último plan y ranking válidos.", reason=str(exc),
                from_version=state["plan_version"], to_version=previous["plan_version"])
            ranked = previous["ranked"]
        effective = {**state, **updates}
        blocked = active_exclusions(effective)
        candidates = [row for row in ranked if row[0] not in blocked]
        saturated = max(effective.get("n_lyrics", 0), effective.get("n_audio", 0)) >= state["requested_top_k"]
        status = "running" if candidates else "no_candidates"
        if not candidates and saturated and counts["expansions"] < max_expansions:
            counts["expansions"] += 1
            updates["requested_top_k"] = state["requested_top_k"] * 2
            status = "expand"
        updates.update(ranked=ranked, candidates=candidates, status=status,
                       trace=event(state, "search", candidates=len(candidates), status=status))
        if status == "no_candidates":
            updates["budget_reason"] = "retrieved_window_exhausted"
        return updates

    def evaluate_node(state):
        counts = dict(state["counters"])
        if counts["candidate_visits"] >= max_candidates or counts["judge_evaluations"] >= max_iterations:
            reason = "candidate_budget" if counts["candidate_visits"] >= max_candidates else "judge_budget"
            return dict(status="limit_reached", budget_reason=reason)
        song_id = state["candidates"][0][0]
        iteration = state["iterations"] + 1
        counts["candidate_visits"] += 1
        p = state["plan"]
        requested = module.requested_aspects(p) if module is not None else None
        needs_lyrics = requested["primary"] if requested is not None else bool(p["lyric_positive"] or p["lyric_negative"])
        needs_audio = requested["secondary"] if requested is not None else bool(p["clap_positive"] or p["clap_negative"])
        validation = dict(lyrics=dict(status="not_evaluated" if needs_lyrics else "not_requested"),
                          audio=dict(status="not_evaluated" if needs_audio else "not_requested",
                                     reason="CLAP retrieval is not independent acoustic validation"))
        candidate, scope, judge_called = None, "plan", False
        try:
            candidate = lookup(song_id)
            if module is not None:
                review = module.evaluate_candidate(query=state["query"], plan=p, candidate=candidate,
                                                   item_id=song_id, request_id=state["request_id"])
                verdict, validation = review["verdict"], review["validation"]
                judge_called, scope = review["judge_called"], review["scope"]
                if review["metadata_skip"]:
                    counts["metadata_skips"] += 1
                if judge_called:
                    counts["judge_evaluations"] += 1
                    counts["judge_model_calls"] += verdict.get("model_calls", 1)
            elif not candidate:
                scope = "request"
                counts["metadata_skips"] += 1
                verdict = dict(status="rejected", reason="Metadata unavailable", refinable=False)
            elif not needs_lyrics:
                verdict = dict(status="not_evaluated", reason="Acoustic retrieval only", refinable=False)
            elif not candidate.get("lyrics", "").strip():
                counts["metadata_skips"] += 1
                verdict = dict(status="abstained", reason="No lyric evidence available", refinable=False)
                validation["lyrics"] = dict(status="abstained", reason=verdict["reason"])
            else:
                judge_called = True
                counts["judge_evaluations"] += 1
                verdict = judge(query=state["query"], lyrics=candidate["lyrics"],
                    lyric_positive=p["lyric_positive"], lyric_negative=p["lyric_negative"],
                    song_id=song_id, song_title=candidate["title"], request_id=state["request_id"])
                counts["judge_model_calls"] += verdict.get("model_calls", 1)
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
        verdict = dict(verdict, song_id=song_id, iteration=iteration, plan_version=state["plan_version"],
                       title=(candidate or {}).get("title", ""), validation=validation, judge_called=judge_called)
        updates = dict(iterations=iteration, counters=counts, verdict=verdict, candidate=candidate,
                       judge_log=state["judge_log"] + [verdict], validation=validation,
                       trace=event(state, "evaluate", status=verdict["status"], song_id=song_id))
        if verdict["status"] == "error":
            updates.update(status="error", error_stage="judge", error_reason=verdict["reason"])
        elif candidate and not needs_lyrics:
            updates.update(status="retrieved", selected_song=song_id, selected_item=song_id)
        elif verdict["status"] == "approved":
            updates.update(status="partially_validated" if needs_audio else "approved",
                           selected_song=song_id, approved_song=song_id,
                           selected_item=song_id, approved_item=song_id)
        else:
            ledger = state["exclusions"] + [dict(song_id=song_id, reason=verdict["reason"],
                decision=verdict["status"], scope=scope, plan_version=state["plan_version"], iteration=iteration)]
            blocked = active_exclusions({**state, "exclusions": ledger})
            remaining = [row for row in state["ranked"] if row[0] not in blocked]
            exhausted = counts["candidate_visits"] >= max_candidates or counts["judge_evaluations"] >= max_iterations
            updates.update(exclusions=ledger, candidates=remaining, status="limit_reached" if exhausted else "running")
            if exhausted:
                updates["budget_reason"] = "candidate_budget" if counts["candidate_visits"] >= max_candidates else "judge_budget"
        return updates

    def refine_node(state):
        counts = dict(state["counters"])
        eligible = (enable_refinement and not state.get("refinement_disabled")
                    and state["verdict"]["status"] == "rejected" and state["verdict"].get("refinable", True)
                    and counts["refinements"] < max_refinements)
        if not eligible:
            return dict(trace=event(state, "refine", modalities=[], reason="policy, evidence or budget"))
        counts["refinements"] += 1
        try:
            result = refiner(query=state["query"], plan=deepcopy(state["plan"]),
                            verdict=deepcopy(state["verdict"]), candidate=deepcopy(state["candidate"]),
                            request_id=state["request_id"])
            if module is not None:
                result = module.validate_refinement(state["plan"], result)
            else:
                from refiner import validate_refinement
                result = validate_refinement(state["plan"], result)
        except Exception as exc:
            return dict(counters=counts, refinement_disabled=True,
                refinement_log=state["refinement_log"] + [dict(iteration=state["iterations"], status="error", reason=str(exc))],
                degradations=degradation(state, "refine", "Falló el refinador; se continúa con el plan válido.", reason=str(exc)),
                trace=event(state, "refine", status="degraded", reason=str(exc)))
        fingerprint = module.plan_fingerprint if module is not None else plan_fingerprint
        changed = fingerprint(result["plan"]) != fingerprint(state["plan"])
        version = state["plan_version"] + int(changed)
        log = dict(iteration=state["iterations"], from_version=state["plan_version"], to_version=version, **result)
        return dict(plan=result["plan"], plan_version=version, counters=counts,
                    refinement_log=state["refinement_log"] + [log],
                    trace=event(state, "refine", modalities=result["modalities"], to_version=version))

    def observed(name, node):
        def invoke(state):
            started = perf_counter()
            updates = node(state)
            if name == "search":
                info = updates.pop("_attempt_diagnostics", {})
                updates["retrieval_history"] = state.get("retrieval_history", []) + [
                    dict(plan_version=state["plan_version"], requested_top_k=state["requested_top_k"],
                         branches=info.get("branches", {}), effective_weights=info.get("effective_weights"),
                         error=info.get("error"))]
            updates["node_timings"] = state.get("node_timings", []) + [
                dict(node=name, elapsed_ms=(perf_counter()-started)*1000)]
            return updates
        return invoke

    graph = StateGraph(SearchState)
    for name, node in (("plan", plan_node), ("search", search_node), ("evaluate", evaluate_node), ("refine", refine_node)):
        graph.add_node(name, observed(name, node))
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda s: "search" if s["status"] == "running" else "end", {"search": "search", "end": END})
    graph.add_conditional_edges("search", lambda s: "evaluate" if s["status"] == "running" else "search" if s["status"] == "expand" else "end",
                               {"evaluate": "evaluate", "search": "search", "end": END})
    graph.add_conditional_edges("evaluate", lambda s: "refine" if s["status"] == "running" else "end", {"refine": "refine", "end": END})
    graph.add_edge("refine", "search")
    return graph.compile()


def initial_state(query, request_id, pool=None):
    """The optional legacy pool is ignored; only JSON values enter state."""
    return SearchState(query=query, request_id=request_id, plan_version=1, exclusions=[],
        iterations=0, judge_log=[], refinement_log=[], trace=[], status="running", ranked=[],
        candidates=[], approved_song=None, selected_song=None, approved_item=None, selected_item=None,
        validation={}, degradations=[], node_timings=[], retrieval_history=[],
        counters=dict(candidate_visits=0, metadata_skips=0, judge_evaluations=0,
                      judge_model_calls=0, refinements=0, expansions=0, retrieval_rounds=0),
        lyric_weight=.6, audio_weight=.4, n_lyrics=0, n_audio=0)
