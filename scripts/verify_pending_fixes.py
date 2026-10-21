"""Capture current regressions and source checks without replacing older blocks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "validation/pending_fixes"
TARGET.mkdir(parents=True, exist_ok=True)
summary = {}
for encoding in ("cp1252", "utf-8"):
    env = dict(os.environ, PYTHONIOENCODING=encoding, PYTHONUTF8="1" if encoding == "utf-8" else "0")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                            cwd=ROOT, env=env, capture_output=True)
    output = (result.stdout + result.stderr).decode(encoding, errors="replace")
    (TARGET / f"tests-{encoding}.txt").write_text(output, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests", output)
    summary[encoding] = dict(exit_code=result.returncode, tests=int(match[1]) if match else None)
(TARGET / "tests-summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
files = list((ROOT / "documentacion").glob("*.tex"))
source = "\n".join(path.read_text(encoding="utf-8") for path in files)
labels = re.findall(r"\\label\{([^}]+)\}", source)
refs = re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", source)
inputs = re.findall(r"\\input\{([^}]+)\}", source)
checks = dict(tex_files=len(files), missing_references=sorted(set(refs)-set(labels)),
    duplicate_labels=sorted({label for label in labels if labels.count(label)>1}),
    missing_inputs=[name for name in inputs if not (ROOT / "documentacion" / (name+".tex")).is_file()],
    pdf_compilation_verified=False)
baseline = TARGET / "baseline.json"
checks["changed_manifest_files"] = ([name for name, sha in json.loads(baseline.read_text(encoding="utf-8"))["manifest"]["sha256"].items()
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha] if baseline.is_file() else ["missing baseline report"])
old = json.loads((ROOT / "validation/block5/baseline.json").read_text(encoding="utf-8"))["manifest"]["sha256"]
checks["changed_legacy_catalog_files"] = [name for name in (
    "datos_alineados.csv", "song_id_map.json", "faiss_lyrics.index", "faiss_audio.index")
    if hashlib.sha256((ROOT / "output" / name).read_bytes()).hexdigest() != old.get("output/"+name, old.get("output\\"+name))]
(TARGET / "source-checks.json").write_text(json.dumps(checks, indent=2)+"\n", encoding="utf-8")
print(json.dumps(dict(tests=summary, sources=checks)))
raise SystemExit(int(any(item["exit_code"] or item["tests"] is None for item in summary.values())
    or any(checks[key] for key in ("missing_references", "duplicate_labels", "missing_inputs",
                                 "changed_manifest_files", "changed_legacy_catalog_files"))))
