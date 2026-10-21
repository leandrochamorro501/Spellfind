"""Install the checked-in lock using the current virtual environment."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indexing", action="store_true", help="Install dataset rebuilding tools")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Activate a virtual environment before installing dependencies")
    base = Path(__file__).resolve().parent
    lock = base / ("requirements/indexing.lock" if args.indexing else "requirements.txt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(lock)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
