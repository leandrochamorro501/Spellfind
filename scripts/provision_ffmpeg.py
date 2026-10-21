"""Provision FFmpeg from the installed pinned wheel inside the project."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
target=ROOT/'.tools/ffmpeg/ffmpeg.exe'
target.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(imageio_ffmpeg.get_ffmpeg_exe(),target)
version=subprocess.check_output([str(target),'-version'],text=True).splitlines()[0]
print(json.dumps(dict(path=str(target),version=version,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
