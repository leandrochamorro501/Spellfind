"""Capture identical tests under both Windows output encodings."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
target = root / "validation/block3"
target.mkdir(parents=True, exist_ok=True)
summary = {}
for encoding in ("cp1252", "utf-8"):
    env = dict(os.environ, PYTHONIOENCODING=encoding, PYTHONUTF8="1" if encoding == "utf-8" else "0")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                            cwd=root, env=env, capture_output=True)
    output = (result.stdout + result.stderr).decode(encoding, errors="replace")
    (target / f"tests-{encoding}.txt").write_text(output, encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", output)
    summary[encoding] = dict(exit_code=result.returncode, tests=int(count[1]) if count else None)
(target / "tests-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary))
sys.exit(int(any(item["exit_code"] for item in summary.values())))
