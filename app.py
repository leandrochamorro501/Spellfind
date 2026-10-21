"""SPELLFIND web adapter for the shared LangGraph search workflow."""
import traceback
import os
from time import perf_counter
from profiling import worker_profile
from scoring import hybrid_scores, METHOD as SCORE_METHOD
from search_options import validate_weights, validate_filters, matches_metadata, catalog_options
from result_evidence import lyric_excerpt
from retrieval import SearchBatch, RetrievalError
from uuid import uuid4
from threading import RLock
from config import web_config
from resources import get_catalog, get_song_id_map, get_index, get_song, validate_bundle, use_snapshot
from workflow import build_workflow, initial_state
from music_module import MusicModule, registry

from flask import Flask, Blueprint, render_template, request, jsonify, send_from_directory

# ── Imports lazy ──────────────────────────────────────────────────────────────
def _import_router():
    from router import get_router_chain
    return get_router_chain()

def _import_semantic_search():
    from semantic_search import search_lyrics
    return search_lyrics

def _import_clap_search():
    from clap_search import search_audio
    return search_audio

def _import_refiner():
    from refiner import refine_plan
    return refine_plan


def _import_judge():
    from judge import evaluate_song, ExclusionPool
    return evaluate_song, ExclusionPool

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

TOP_K           = 10
FINAL_TOP_K     = 5
MAX_JUDGE_ITERS = 3
REFINEMENT_ENABLED = os.getenv("SPELLFIND_ENABLE_REFINEMENT", "false").lower() in {"true", "1", "yes"}

blueprint = Blueprint("spellfind", __name__)

# ──────────────────────────────────────────────────────────────────────────────
# SEARCH ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class SearchEngine:
    def __init__(self, catalog_manager=None):
        self.catalog_manager, self._snapshot, self.catalog_warning = catalog_manager, None, None
        self.df_meta     = None
        self.song_id_map = None
        self.idx_lyrics  = None
        self.idx_audio   = None
        self.ready       = False
        self.error       = None
        self._lock = RLock()
        self.initialized = False

    def load_indices(self):
        with self._lock:
            self._load_indices()

    def _load_indices(self):
        self.initialized = True
        self.ready = self._snapshot is not None
        self.error = None
        try:
            self.pin_snapshot()
            self.ready = True
        except Exception as exc:
            self.error = str(exc)
            traceback.print_exc()

    def get_song_by_id(self, song_id):
        return self._snapshot.get_song_by_id(song_id) if self._snapshot is not None else get_song(song_id)

    def pin_snapshot(self):
        from catalog import default_manager
        with self._lock:
            manager = self.catalog_manager or default_manager()
            try:
                snapshot = manager.current()
                self.catalog_warning = None
            except Exception:
                if self._snapshot is None: raise
                self.catalog_warning = "No se pudo cargar la nueva generación; se conserva el catálogo anterior."
                return self._snapshot
            if self._snapshot is None or snapshot.version != self._snapshot.version:
                self._snapshot = snapshot
                self.df_meta, self.song_id_map = snapshot.df_meta, snapshot.song_id_map
                self.idx_lyrics, self.idx_audio = snapshot.idx_lyrics, snapshot.idx_audio
            return snapshot

    def metadata_rows(self):
        if self._snapshot is not None: return self._snapshot.metadata_rows()
        return self.df_meta.reindex(columns=["song_id", "artist", "tag", "year"], fill_value="").to_dict(orient="records")


engine = SearchEngine()

# ──────────────────────────────────────────────────────────────────────────────
# PESOS DINÁMICOS
# ──────────────────────────────────────────────────────────────────────────────

def _infer_weights(lyric_positive: list, clap_positive: list) -> tuple[float, float]:
    has_lyric = bool(lyric_positive)
    has_audio = bool(clap_positive)
    if has_lyric and has_audio:
        return 0.6, 0.4
    elif has_lyric:
        return 1., 0.
    elif has_audio:
        return 0., 1.
    else:
        return 0., 0.

# ──────────────────────────────────────────────────────────────────────────────
# FUSIÓN HÍBRIDA (RRF ponderado por modalidad)
# ──────────────────────────────────────────────────────────────────────────────

def compute_hybrid_score(
    lyrics_results: list[tuple[int, float]],
    audio_results:  list[tuple[int, float]],
    lyric_weight:   float = 0.6,
    audio_weight:   float = 0.4,
) -> list[tuple[int, float]]:
    return hybrid_scores(lyrics_results, audio_results, lyric_weight, audio_weight)[0]

# ──────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA
# ──────────────────────────────────────────────────────────────────────────────

def _execute_search(
    lyric_positive: list,
    lyric_negative: list,
    clap_positive:  list,
    clap_negative:  list,
    lyric_weight:   float,
    audio_weight:   float,
    excluded_ids:   set | None = None,
    cache:          dict | None = None,
    top_k:          int | None = None,
    allowed_ids:    set | None = None,
    catalog_size:   int | None = None,
) -> tuple[list, list, list]:
    """
    Ejecuta las ramas con peso positivo y fusiona por rangos ponderados.
    Devuelve (lyrics_results, audio_results, fused).
    """
    excluded_ids = excluded_ids or set()

    cache = {} if cache is None else cache
    top_k = TOP_K if top_k is None else top_k
    details = {"branches": {}, "requested_weights": {"lyrics": lyric_weight, "audio": audio_weight},
               "weight_policy": "renormalize_nonempty_successful_positive_weight_branches", "top_k": top_k,
               "score_method": SCORE_METHOD,
               "filter_scope": "full_catalog_then_exact_filter" if allowed_ids is not None else "retrieved_window"}
    fetch_k = max(top_k, catalog_size or 0) if allowed_ids is not None else top_k

    def retrieve(modality, positive, negative, importer):
        start = perf_counter()
        if not positive:
            details["branches"][modality] = {"status": "not_requested"}
            return []
        weight = lyric_weight if modality == "lyrics" else audio_weight
        if weight == 0 or allowed_ids == set():
            details["branches"][modality] = {"status": "zero_weight" if weight == 0 else "no_filter_matches", "count": 0}
            return []
        key = (modality, tuple(sorted(positive)), tuple(sorted(negative)), fetch_k)
        try:
            hit = key in cache
            profile = None
            if not hit:
                with worker_profile() as profile:
                    cache[key] = list(importer()(positive=positive, negative=negative, top_k=fetch_k))
            rows = cache[key]
            if allowed_ids is not None:
                rows = [row for row in rows if row[0] in allowed_ids][:top_k]
            details["branches"][modality] = {"status": "ok" if rows else "empty", "count": len(rows),
                "request_cache_hit": hit, "elapsed_ms": (perf_counter() - start) * 1000,
                "worker_profile": profile, "fetched_top_k": fetch_k}
            return rows
        except Exception as exc:
            details["branches"][modality] = {"status": "failed", "reason": str(exc),
                                            "elapsed_ms": (perf_counter() - start) * 1000}
            return []

    lyrics_results = retrieve("lyrics", lyric_positive, lyric_negative, _import_semantic_search)
    audio_results = retrieve("audio", clap_positive, clap_negative, _import_clap_search)

    failed = [name for name, branch in details["branches"].items() if branch["status"] == "failed"]
    lw = lyric_weight if lyrics_results else 0.0
    aw = audio_weight if audio_results else 0.0
    total = lw + aw
    if total:
        lw, aw = lw / total, aw / total
    details.update(effective_weights={"lyrics": lw, "audio": aw}, degraded=bool(failed))
    if not lyrics_results and not audio_results:
        if failed:
            raise RetrievalError(details)
        return SearchBatch([], [], [], details)

    fused, components = hybrid_scores(lyrics_results, audio_results, lw, aw)
    details["score_components"] = components
    if excluded_ids:
        fused = [(sid, sc) for sid, sc in fused if sid not in excluded_ids]

    return SearchBatch(lyrics_results, audio_results, fused, details)

# ──────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def get_top_k_results(query: str, k: int = FINAL_TOP_K, search_engine=None, *, weights=None, filters=None, media_type="music") -> dict:
    from catalog import Snapshot
    active = engine if search_engine is None else search_engine
    if isinstance(active, (SearchEngine, Snapshot)) and active.ready:
        started = perf_counter()
        snapshot = active.pin_snapshot() if isinstance(active, SearchEngine) else active
        pinned_ms = (perf_counter()-started)*1000
        with use_snapshot(snapshot):
            result = _get_top_k_results(query, k, snapshot, weights=weights, filters=filters, media_type=media_type)
            result["catalog_version"] = snapshot.version
            result["catalog_warning"] = getattr(active, "catalog_warning", None)
            if result["catalog_warning"]:
                result["warning"] = " ".join(filter(None, (result.get("warning"), result["catalog_warning"])))
            for song in result["results"]:
                if song.pop("audio_file", None): song["audio_url"] = f"/catalog/{snapshot.version}/songs/{song['song_id']}/audio"
            result["timings"]["catalog_pin_ms"] = pinned_ms
            result["timings"]["total_ms"] = (perf_counter()-started)*1000
            return result
    return _get_top_k_results(query, k, active, weights=weights, filters=filters, media_type=media_type)


def _get_top_k_results(query: str, k: int = FINAL_TOP_K, search_engine=None, *, weights=None, filters=None, media_type="music") -> dict:
    started = perf_counter()
    if media_type != "music":
        raise ValueError("Este medio aún no está disponible; por ahora solo se puede buscar música.")
    weights, filters = validate_weights(weights), validate_filters(filters)
    active_engine = engine if search_engine is None else search_engine
    if not isinstance(query, str) or not query.strip():
        raise ValueError("La consulta debe ser un texto no vacío")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k debe ser un entero positivo")
    if not active_engine.ready:
        return {"error": f"Motor no inicializado: {active_engine.error}"}
    request_id = uuid4().hex
    allowed_ids, size = None, None
    if filters:
        if not callable(getattr(active_engine, "metadata_rows", None)):
            raise ValueError("El motor no admite filtros de catálogo")
        rows = active_engine.metadata_rows()
        size = len(rows)
        allowed_ids = {int(row["song_id"]) for row in rows if matches_metadata(row, filters)}
    filter_summary = dict(applied=filters, matched_catalog_songs=len(allowed_ids) if allowed_ids is not None else None,
                          catalog_songs=size, semantics="exact_nfc_casefold_trim_and_inclusive_years")
    if allowed_ids == set():
        # No candidate can satisfy the explicit filters. Avoid router/model calls.
        return dict(api_version=2, media_type="music", common_plan=None, media_capabilities=[],
            request_id=request_id, query=query, results=[], plan=None, initial_plan=None,
            judge_status="no_candidates", termination_reason="no_filter_matches", error_stage=None,
            warning="No hay canciones que cumplan todos los filtros del catálogo.", filters=filters,
            filter_summary=filter_summary, requested_weights=weights, lyric_weight=0., audio_weight=0.,
            score_method=SCORE_METHOD, score_is_probability=False, score_range=[0, 1],
            n_lyrics=0, n_audio=0, judge_iterations=0, candidate_iterations=0, constraints=[], requested_modalities=[],
            exclusions=[], retrieval={}, degradations=[], counters=initial_state(query, request_id)["counters"],
            budgets={}, budget_reason="no_filter_matches", catalog_exhausted=False, validation_scope="none",
            plan_version=0, judge_verdict=None, judge_log=[], refinement_log=[], workflow_trace=[], node_timings=[],
            retrieval_history=[],
            timings=dict(total_ms=(perf_counter()-started)*1000))
    judge_fn, _ = _import_judge()
    music = MusicModule(planner=lambda text: _import_router().invoke({"query": text}),
        search=(lambda **kw: _execute_search(**kw, allowed_ids=allowed_ids, catalog_size=size)) if filters else _execute_search,
        judge=judge_fn, refiner=lambda **kwargs: _import_refiner()(**kwargs),
        lookup=active_engine.get_song_by_id,
        infer_weights=(lambda *_: (weights["lyrics"], weights["audio"])) if weights is not None else _infer_weights)
    module = registry(music)[media_type]
    build_started = perf_counter()
    graph = build_workflow(module=module,
        max_iterations=MAX_JUDGE_ITERS, enable_refinement=REFINEMENT_ENABLED,
        initial_top_k=TOP_K)
    build_ms = (perf_counter() - build_started) * 1000
    invoke_started = perf_counter()
    state = graph.invoke(initial_state(query, request_id),
                         config={"recursion_limit": 160})
    workflow_ms = (perf_counter() - invoke_started) * 1000
    format_started = perf_counter()
    selected = state["selected_song"]
    historical = {row["song_id"] for row in state["exclusions"]}
    # A historical rejection can return only after this plan explicitly reevaluates it.
    ranked = [(sid, score) for sid, score in state["ranked"] if sid not in historical or sid == selected]
    if selected is not None:
        ranked.sort(key=lambda item: item[0] != selected)
    results = []
    for song_id, score in ranked:
        meta = active_engine.get_song_by_id(song_id)
        if meta:
            decision = next((entry for entry in reversed(state["judge_log"])
                             if entry["song_id"] == song_id and entry["plan_version"] == state["plan_version"]), None)
            validation = decision["validation"] if decision else {
                "lyrics": {"status": "not_evaluated" if state["plan"]["lyric_positive"] or state["plan"]["lyric_negative"] else "not_requested"},
                "audio": {"status": "not_evaluated" if state["plan"]["clap_positive"] or state["plan"]["clap_negative"] else "not_requested"}}
            snippet = lyric_excerpt(meta.get("lyrics", ""), validation, state["plan"]["lyric_positive"],
                                    bool(state["plan"]["lyric_positive"] or state["plan"]["lyric_negative"]))
            components = state.get("retrieval", {}).get("score_components", {}).get(str(song_id), {})
            meta.update(media_type=module.media_type,
                        condition_verdicts=module.condition_verdicts(state["plan"], validation,
                            catalog_version=getattr(active_engine, "version", None)),
                        score=score, score_components=components, lyric_excerpt=snippet, validation=validation,
                        selection_reason="validated_recommendation" if song_id == selected and state["status"] in
                                         {"approved", "partially_validated"} else "retrieval_alternative",
                        validation_status=(state["status"] if song_id == selected and state["status"] in
                                           {"approved", "partially_validated"} else decision["status"] if decision else "not_evaluated"))
            meta.pop("lyrics", None)
            results.append(meta)
        if len(results) >= k:
            break
    warnings = {
        "clarification_needed": state["plan"].get("planning_warning", "Reformula la consulta para indicar letra, sonido y restricciones."),
        "retrieved": "Resultados acústicos recuperados; el sonido no ha sido validado de forma independiente.",
        "partially_validated": "Letra validada; el sonido permanece sin evaluación independiente.",
        "error": "No se pudo completar la validación. Los resultados no están validados.",
        "limit_reached": "Se alcanzó el límite de intentos sin una recomendación aprobada.",
        "no_candidates": "No quedan candidatos utilizables en la ventana recuperada; no se ha agotado todo el catálogo.",
        "abstained": "El juez se abstuvo; no hay una recomendación validada.",
    }
    if state["status"] == "no_candidates" and any(branch.get("status") == "zero_weight"
                                                  for branch in state.get("retrieval", {}).get("branches", {}).values()):
        warnings["no_candidates"] = "No hay resultados de modalidades solicitadas con peso positivo. Revisa los pesos."
    return {
        "api_version": 2, "media_type": module.media_type,
        "media_capabilities": [item.as_dict() for item in module.capabilities],
        "common_plan": module.public_plan(state["plan"]),
        "score_method": SCORE_METHOD, "score_range": [0, 1], "score_is_probability": False,
        "selection_policy": "validated_candidate_first_then_retrieval_score",
        "requested_weights": weights if weights is not None else state.get("retrieval", {}).get("requested_weights"),
        "filters": filters, "filter_summary": filter_summary,
        "timings": dict(graph_build_ms=build_ms, workflow_ms=workflow_ms,
                        format_ms=(perf_counter()-format_started)*1000, total_ms=(perf_counter()-started)*1000),
        "node_timings": state.get("node_timings", []),
        "retrieval_history": state.get("retrieval_history", []),
        "request_id": request_id, "judge_status": state["status"],
        "termination_reason": state["status"], "error_stage": state.get("error_stage"),
        "warning": " ".join(filter(None, [warnings.get(state["status"])] +
                    [row["message"] for row in state.get("degradations", [])])) or None,
        "exclusions": state["exclusions"],
        "retrieval": state.get("retrieval", {}), "degradations": state.get("degradations", []),
        "budgets": state.get("budgets", {}), "counters": state.get("counters", {}),
        "budget_reason": state.get("budget_reason"),
        "catalog_exhausted": False,
        "query": query, "plan": state["plan"], "initial_plan": state["initial_plan"],
        "results": results, "lyric_weight": state["lyric_weight"],
        "audio_weight": state["audio_weight"], "n_lyrics": state["n_lyrics"],
        "n_audio": state["n_audio"], "judge_verdict": state.get("verdict"),
        "validation_scope": "lyrics" if state["plan"]["lyric_positive"] or state["plan"]["lyric_negative"] else "none",
        "requested_modalities": [name for name,prefix in (("lyrics","lyric"),("audio","clap"))
                                 if state["plan"][prefix+"_positive"] or state["plan"][prefix+"_negative"]],
        "plan_version": state["plan_version"], "constraints": state["plan"].get("constraints", []),
        "candidate_iterations": state["iterations"],
        "judge_iterations": sum(row.get("judge_called",False) for row in state["judge_log"]), "judge_log": state["judge_log"],
        "refinement_log": state["refinement_log"], "workflow_trace": state["trace"],
    }

# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS FLASK
# ──────────────────────────────────────────────────────────────────────────────

@blueprint.route("/")
def index():
    return render_template("index.html")


@blueprint.route("/search", methods=["POST"])
def search():
    from flask import current_app
    if not request.is_json:
        return jsonify({"error": "Se requiere un cuerpo JSON."}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "El cuerpo debe ser un objeto JSON válido."}), 400
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "El campo 'query' debe ser un texto no vacío."}), 400
    query = query.strip()
    if len(query) > current_app.config["MAX_QUERY_LENGTH"]:
        return jsonify({"error": "La consulta excede el límite de longitud."}), 400
    try:
        weights, filters = validate_weights(body.get("weights")), validate_filters(body.get("filters"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    active_engine = current_app.extensions["spellfind_engine"]
    if not active_engine.ready:
        return jsonify({"error": "El motor no está disponible.", "ready": False}), 503
    try:
        options = {}
        if "media_type" in body:
            options["media_type"] = body["media_type"]
        if weights is not None:
            options["weights"] = weights
        if filters:
            options["filters"] = filters
        result = get_top_k_results(query, k=FINAL_TOP_K, search_engine=active_engine, **options)
        status = 503 if result.get("judge_status") == "error" else 200
        return jsonify(result), status
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Unexpected search failure")
        return jsonify({"error": "No se pudo completar la búsqueda."}), 500


@blueprint.route("/health")
def health():
    from flask import current_app
    active_engine = current_app.extensions["spellfind_engine"]
    owner = active_engine
    if isinstance(active_engine, SearchEngine) and active_engine.ready: active_engine = active_engine.pin_snapshot()
    ready = active_engine.ready
    return jsonify({
        "ready": ready,
        "catalog_version": getattr(active_engine, "version", None), "catalog_warning": getattr(owner, "catalog_warning", None),
        "error": None if ready else "Catalog unavailable or invalid",
        "songs": len(active_engine.df_meta) if active_engine.df_meta is not None else 0,
        "lyrics_vectors": active_engine.idx_lyrics.ntotal if active_engine.idx_lyrics is not None else 0,
        "audio_vectors": active_engine.idx_audio.ntotal if active_engine.idx_audio is not None else 0,
    }), 200 if ready else 503


@blueprint.route("/catalog/options")
def filter_options():
    from flask import current_app
    active_engine = current_app.extensions["spellfind_engine"]
    if isinstance(active_engine, SearchEngine) and active_engine.ready: active_engine = active_engine.pin_snapshot()
    if not active_engine.ready:
        return jsonify({"error": "El catálogo no está disponible."}), 503
    return jsonify(catalog_options(active_engine.metadata_rows()))


@blueprint.route("/media/capabilities")
def media_capabilities():
    """List actual modules; planned media are deliberately absent."""
    return jsonify(media=[dict(media_type="music", label="Música",
        capabilities=[item.as_dict() for item in MusicModule.capabilities])])


@blueprint.route("/ingest/status")
def ingest_status():
    from flask import current_app
    from ingestion import capabilities
    service = current_app.extensions.get("spellfind_ingestion")
    state = service.status() if service is not None else capabilities()
    return jsonify(enabled=current_app.config["INGEST_ENABLED"], **state)


@blueprint.route("/ingest", methods=["POST"])
def ingest():
    from flask import current_app
    from ingestion import ingest_payload, QueueFull
    from maintenance import StorageLimit
    if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")) or request.headers.get("Sec-Fetch-Site") == "cross-site":
        return jsonify(error="La ingesta debe enviarse desde esta aplicación local."), 403
    if not request.is_json: return jsonify(error="Se requiere JSON."), 400
    try: url, metadata = ingest_payload(request.get_json(silent=True))
    except ValueError as exc: return jsonify(error=str(exc)), 400
    service = current_app.extensions.get("spellfind_ingestion")
    if not current_app.config["INGEST_ENABLED"] or service is None:
        return jsonify(error="La ingesta está desactivada en esta instalación."), 503
    if not service.status()["ready"]:
        return jsonify(error="Faltan herramientas de ingesta.", capabilities=service.status()), 503
    try: job = service.enqueue(url, metadata)
    except (QueueFull, StorageLimit) as exc: return jsonify(error=str(exc)), 429
    except RuntimeError as exc: return jsonify(error=str(exc)), 503
    return jsonify(job=job, status_url=f"/ingest/jobs/{job['id']}"), 202


@blueprint.route("/ingest/jobs/<job_id>")
def ingest_job(job_id):
    from flask import current_app
    from pathlib import Path
    from ingestion import JobStore
    service = current_app.extensions.get("spellfind_ingestion")
    root = Path(current_app.config["INGEST_DIR"])
    store = service.store if service is not None else JobStore(root) if (root / "jobs.sqlite3").is_file() else None
    job = store.get(job_id) if store is not None else None
    return (jsonify(job=job), 200) if job is not None else (jsonify(error="Trabajo no encontrado."), 404)


@blueprint.route("/catalog/<version>/songs/<int:song_id>/audio")
def song_audio(version, song_id):
    from flask import current_app
    from pathlib import Path
    import re
    from catalog import default_manager
    active = current_app.extensions["spellfind_engine"]
    manager = getattr(active, "catalog_manager", None) or default_manager()
    try: song = manager.at_version(version).get_song_by_id(song_id)
    except (ValueError, OSError): return jsonify(error="Versión no disponible."), 404
    name = song.get("audio_file", "") if song else ""
    if not re.fullmatch(r"[a-f0-9]{32}\.mp3", name): return jsonify(error="No hay audio local para esta canción."), 404
    service = current_app.extensions.get("spellfind_ingestion")
    media = (service.store.root if service is not None else Path(current_app.config["INGEST_DIR"])) / "media"
    if (media / name).resolve().parent != media.resolve(): return jsonify(error="Audio no disponible."), 404
    return send_from_directory(media, name, mimetype="audio/mpeg", conditional=True)


def create_app(config=None, search_engine=None, initialize=True, ingest_service=None):
    application = Flask(__name__)
    application.config.update(web_config())
    from config import INGEST_DIR
    application.config["INGEST_DIR"] = str(INGEST_DIR)
    if config:
        application.config.update(config)
    active_engine = engine if search_engine is None else search_engine
    application.extensions["spellfind_engine"] = active_engine
    if initialize:
        active_engine.load_indices()
    if application.config["INGEST_ENABLED"] and ingest_service is None:
        from ingestion import IngestionService
        from catalog import default_manager
        ingest_service = IngestionService(getattr(active_engine, "catalog_manager", None) or default_manager(), application.config["INGEST_DIR"])
    application.extensions["spellfind_ingestion"] = ingest_service
    if ingest_service is not None and application.config["INGEST_ENABLED"]:
        import atexit
        ingest_service.start()
        atexit.register(ingest_service.close)
    application.register_blueprint(blueprint)

    @application.errorhandler(413)
    def too_large(error):
        return jsonify({"error": "El cuerpo de la solicitud es demasiado grande."}), 413

    return application


if __name__ == "__main__":
    application = create_app()
    application.run(host=application.config["HOST"], port=application.config["PORT"],
                    debug=application.config["DEBUG"], use_reloader=False)
