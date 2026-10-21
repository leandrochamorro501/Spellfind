"""Explicit local UI fixture. Never use this controlled planner/judge for recommendations."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import app
    parser = argparse.ArgumentParser(description="UI QA only: controlled rankings and judgments, real catalog metadata")
    parser.add_argument("--port", type=int, default=5074)
    args = parser.parse_args()
    application = app.create_app()
    application.jinja_env.auto_reload = True
    if not app.engine.ready:
        raise RuntimeError("Catalog unavailable")
    ids = sorted(int(value) for value in app.engine.df_meta["song_id"])

    def search(**kw):
        return [(sid, 1 / (i+1)) for i, sid in enumerate(ids[:kw["top_k"]])]

    def judge(**kw):
        return dict(status="approved", reason="CONTROL LOCAL: no es una evaluación musical real.",
                    format="structured", evidence=[kw["lyrics"][:80]], model_calls=0, coverage={"complete": True})

    @application.after_request
    def mark_control(response):
        if response.mimetype == "text/html":
            response.set_data(response.get_data(as_text=True).replace("<body>",
                '<body><aside style="padding:1rem;background:#493816;color:#fff">PRUEBA LOCAL: rankings y juez simulados; no mide calidad musical.</aside>', 1))
        return response

    with ExitStack() as stack:
        stack.enter_context(patch.object(app, "_import_router", return_value=SimpleNamespace(invoke=lambda _:
            dict(lyric_positive=["returning home"], lyric_negative=[], clap_positive=["soft piano"], clap_negative=[]))))
        stack.enter_context(patch.object(app, "_import_semantic_search", return_value=search))
        stack.enter_context(patch.object(app, "_import_clap_search", return_value=search))
        stack.enter_context(patch.object(app, "_import_judge", return_value=(judge, None)))
        stack.enter_context(patch.object(app, "REFINEMENT_ENABLED", False))
        application.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
