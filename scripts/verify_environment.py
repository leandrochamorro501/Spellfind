"""Import real runtime libraries and validate local data without downloading models."""
import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--indexing", action="store_true")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use an isolated virtual environment")
    names = ["flask", "waitress", "numpy", "pandas", "faiss", "langchain_core",
             "langchain_groq", "langgraph.graph", "dotenv", "requests"]
    if args.models:
        names += ["torch", "transformers", "sentence_transformers", "einops",
                  "semantic_search", "clap_search"]
    if args.indexing:
        names += ["datasets", "huggingface_hub", "align_and_index"]
    for name in names:
        import_module(name)
    from resources import validate_bundle, get_catalog
    validate_bundle()
    from app import create_app
    application = create_app({"TESTING": True})
    health = application.test_client().get("/health")
    if health.status_code != 200:
        raise RuntimeError("Application is not ready")
    versions = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in versions:
            raise RuntimeError(f"Duplicate distribution metadata: {name}")
        versions[name] = distribution.version
    result = {"status": "passed", "python": platform.python_version(),
              "platform": platform.platform(), "isolated": True,
              "model_weights_downloaded": False, "imports": names,
              "songs": len(get_catalog()), "health": health.json,
              "dependencies": dict(sorted(versions.items()))}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "songs": result["songs"],
                      "imports": len(names), "model_weights_downloaded": False}))


if __name__ == "__main__":
    main()
