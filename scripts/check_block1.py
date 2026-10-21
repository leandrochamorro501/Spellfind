"""Reproduce offline Block I acceptance in the active virtual environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

BASE = Path(__file__).resolve().parents[1]
DEST = BASE / "validation/block1"


def execute(arguments, name, allowed=(0,), env=None):
    result = subprocess.run(arguments, cwd=BASE, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    (DEST / name).write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode not in allowed:
        raise RuntimeError(f"Acceptance step failed: {name}. See validation/block1/{name}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reinstall-lock", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-X", "utf8"]
    execute(py + ["-m", "pip", "check"], "pip_check.txt")
    execute(py + ["scripts/verify_environment.py", "--models", "--indexing",
                   "--output", str(DEST / "environment.json")], "imports.txt")
    execute(py + ["-m", "unittest", "discover", "-s", "tests", "-v"], "tests.txt")
    execute(py + ["-m", "evaluation.run", "--mode", "simulated", "--repeats", "3",
                   "--output", str(DEST / "baseline_simulated.json")], "baseline_simulated.txt")
    execute(py + ["-m", "evaluation.run", "--mode", "simulated", "--repeats", "3", "--refinement",
                   "--output", str(DEST / "baseline_simulated_refined.json")], "baseline_simulated_refined.txt")
    if not os.getenv("GROQ_API_KEY"):
        execute(py + ["-m", "evaluation.run", "--mode", "real", "--repeats", "3",
                       "--output", str(DEST / "baseline_real_status.json")], "baseline_real_status.txt", allowed=(1,),
                env={**os.environ, "GROQ_API_KEY": ""})
    else:
        (DEST / "baseline_real_status.json").write_text(json.dumps({
            "status": "not_run", "reason": "Offline acceptance never calls an external model"}), encoding="utf-8")
    if args.reinstall_lock:
        target = BASE / ".venv-lockcheck"
        if target.exists():
            raise RuntimeError("Lock-check environment already exists; refusing to mix installations")
        venv.EnvBuilder(with_pip=True).create(target)
        new_py = str(target / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        execute([new_py, "-m", "pip", "install", "-r", "requirements.txt"], "lock_install.txt")
        execute([new_py, "-m", "pip", "check"], "lock_pip_check.txt")
        execute([new_py, "-X", "utf8", "scripts/verify_environment.py", "--models",
                 "--output", str(DEST / "lock_environment.json")], "lock_imports.txt")
    print(json.dumps({"status": "passed", "evidence": str(DEST),
                      "real_musical_quality": "pending credentials, weights and human labels"}))


if __name__ == "__main__":
    main()
