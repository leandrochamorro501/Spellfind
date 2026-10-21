"""Project-relative paths and explicit runtime configuration."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("HF_HOME", str(BASE_DIR / ".tools/huggingface"))
MODEL_REVISIONS = json.loads((BASE_DIR / "requirements/model-revisions.json").read_text(encoding="utf-8"))


def project_path(name, default):
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else BASE_DIR / value


OUTPUT_DIR = project_path("SPELLFIND_OUTPUT_DIR", "output")
PROMPT_PATH = project_path("SPELLFIND_PROMPT_PATH", "api_prompt.txt")
CATALOG_DIR = project_path("SPELLFIND_CATALOG_DIR", "data/catalog")
INGEST_DIR = project_path("SPELLFIND_INGEST_DIR", "data/ingestion")


def web_config():
    length = int(os.getenv("SPELLFIND_MAX_QUERY_LENGTH", "4000"))
    port = int(os.getenv("SPELLFIND_PORT", "5000"))
    if length < 1 or not 1 <= port <= 65535:
        raise ValueError("Invalid query limit or port configuration")
    return {
        "MAX_QUERY_LENGTH": length,
        "MAX_CONTENT_LENGTH": max(16384, length * 8),
        "HOST": os.getenv("SPELLFIND_HOST", "127.0.0.1"),
        "PORT": port,
        "DEBUG": os.getenv("SPELLFIND_DEBUG", "false").lower() in {"true", "1", "yes"},
        "INGEST_ENABLED": os.getenv("SPELLFIND_INGEST_ENABLED", "false").lower() in {"true", "1", "yes"},
    }
