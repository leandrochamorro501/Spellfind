"""Resolve the project's optional FFmpeg binary without modifying global PATH."""
import os
from pathlib import Path
import shutil
from config import BASE_DIR


def ffmpeg_path():
    configured = os.getenv("SPELLFIND_FFMPEG")
    if configured:
        path = Path(configured)
        if not path.is_absolute(): path = BASE_DIR / path
        return str(path) if path.is_file() else None
    return shutil.which("ffmpeg") or (str(BASE_DIR / ".tools/ffmpeg/ffmpeg.exe")
        if (BASE_DIR / ".tools/ffmpeg/ffmpeg.exe").is_file() else None)
