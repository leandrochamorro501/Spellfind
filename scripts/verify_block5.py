"""Save local test and source/reference checks without overwriting earlier blocks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
target = root / "validation/block5"
target.mkdir(parents=True, exist_ok=True)
summary = {}
sys.path.insert(0, str(root))
from ingestion import capabilities
from catalog import default_manager
catalog = default_manager().current()
readiness = dict(optional_ingestion=capabilities(), catalog_version=catalog.version,
    catalog_songs=len(catalog.song_id_map), real_ingestion_verified=False,
    real_model_quality_verified=False, browser_ingestion_verified=False,
    pdf_compilation_verified=False,
    closure={"P17": "local_regressions_pass_required",
             "P18": "decision_and_durable_queue_implemented",
             "P19": "adapter_control_tests_only_real_integration_pending",
             "P20": "local_regressions_only_external_validation_pending"},
    pending=["E17 retention_quotas_backups", "E18 optional_lock_real_integration", "E19 instrumental_optional_modalities"])
(target / "readiness.json").write_text(json.dumps(readiness, indent=2)+"\n", encoding="utf-8")
for encoding in ("cp1252", "utf-8"):
    env = dict(os.environ, PYTHONIOENCODING=encoding, PYTHONUTF8="1" if encoding == "utf-8" else "0")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=root, env=env, capture_output=True)
    output = (result.stdout + result.stderr).decode(encoding, errors="replace")
    (target / f"tests-{encoding}.txt").write_text(output, encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", output)
    summary[encoding] = dict(exit_code=result.returncode, tests=int(count[1]) if count else None)
(target / "tests-summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
files = list((root / "documentacion").glob("*.tex"))
source = "\n".join(path.read_text(encoding="utf-8") for path in files)
labels = re.findall(r"\\label\{([^}]+)\}", source)
refs = re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", source)
inputs = re.findall(r"\\input\{([^}]+)\}", source)
checks = dict(tex_files=len(files), missing_references=sorted(set(refs)-set(labels)),
    duplicate_labels=sorted({label for label in labels if labels.count(label)>1}),
    missing_inputs=[name for name in inputs if not (root / "documentacion" / (name+".tex")).is_file()],
    pdf_compilation_verified=False)
baseline = target / "baseline.json"
checks["changed_manifest_files"] = ([name for name, sha in json.loads(baseline.read_text(encoding="utf-8"))["manifest"]["sha256"].items()
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != sha] if baseline.is_file() else None)
(target / "source-checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
print(json.dumps(dict(tests=summary, sources=checks)))
sys.exit(int(any(item["exit_code"] for item in summary.values()) or any(checks[key] for key in
    ("missing_references", "duplicate_labels", "missing_inputs", "changed_manifest_files"))))

