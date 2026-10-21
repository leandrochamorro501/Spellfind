"""Optional YouTube/FFmpeg/Whisper and shared embedding adapters, lazy loaded."""
import html
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from vector_validation import unit_vector


def subtitle_text(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "-->" in line or line.isdigit() or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")): continue
        line = html.unescape(re.sub(r"<[^>]*>", "", line)).strip()
        if line.casefold() in {"[music]", "[música]", "♪", "♫"}: continue
        if line and (not lines or line != lines[-1]): lines.append(line)
    return "\n".join(lines)


class NativeBackend:
    def capabilities(self):
        from ingestion import capabilities
        return capabilities()

    def run(self, job, work, progress):
        from yt_dlp import YoutubeDL
        from media_tools import ffmpeg_path
        ffmpeg = ffmpeg_path() or "ffmpeg"
        progress("downloading", 10)
        options = dict(quiet=True, no_warnings=True, noplaylist=True, socket_timeout=15, retries=1, extractor_retries=1,
                       ffmpeg_location=ffmpeg)
        with YoutubeDL({**options, "skip_download": True}) as downloader:
            info = downloader.extract_info(job["url"], download=False)
        duration = info.get("duration")
        if info.get("is_live") or isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 900:
            raise ValueError("La ingesta admite vídeos de duración conocida de hasta 15 minutos")
        languages = [lang for lang in info.get("subtitles", {}) if lang != "live_chat" and re.fullmatch(r"[a-zA-Z0-9_-]{1,30}", lang)]
        languages.sort(key=lambda lang: (lang != info.get("language"), not lang.startswith("es"), not lang.startswith("en"), lang))
        download_options = {**options, "format": "bestaudio/best", "outtmpl": str(work / "audio.%(ext)s"),
            "max_filesize": 100*1024*1024, "writesubtitles": False, "writeautomaticsub": False,
            "subtitleslangs": languages[:1], "subtitlesformat": "vtt",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}]}
        with YoutubeDL(download_options) as downloader:
            downloader.extract_info(job["url"], download=True)
        audio = work / "audio.mp3"
        if not audio.is_file() or audio.stat().st_size > 100*1024*1024: raise ValueError("Audio ausente o mayor de 100 MiB")
        progress("transcribing", 40)
        instrumental = job["metadata"].get("instrumental") is True
        if languages and not instrumental:
            try:
                with YoutubeDL({**options, "skip_download": True, "outtmpl": str(work / "audio.%(ext)s"),
                    "writesubtitles": True, "writeautomaticsub": False, "subtitleslangs": languages[:1], "subtitlesformat": "vtt"}) as downloader:
                    downloader.extract_info(job["url"], download=True)
            except Exception:
                pass  # Subtitle failure must still allow the Whisper alternative.
        lyrics, source = "", "official_subtitles"
        for path in sorted(work.glob("audio.*.vtt")):
            if path.stat().st_size <= 2*1024*1024:
                try: lyrics = subtitle_text(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError): continue
                if lyrics.strip(): break
        if not lyrics.strip() and not instrumental:
            try: import whisper
            except ImportError: raise RuntimeError("No hay subtítulos oficiales utilizables y falta Whisper") from None
            source = "whisper"
            # Lazy loading keeps Whisper absent from ordinary search startup.
            from threading import RLock
            if not hasattr(self, "_whisper_lock"): self._whisper_lock = RLock()
            with self._whisper_lock:
                if not hasattr(self, "_whisper"):
                    from config import BASE_DIR
                    self._whisper = whisper.load_model("base", device="cpu", download_root=str(BASE_DIR / ".tools/whisper"))
                decoded = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(audio),
                    "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
                    check=True, capture_output=True, timeout=120).stdout
                lyrics = self._whisper.transcribe(np.frombuffer(decoded, dtype="<f4").copy(), fp16=False).get("text", "").strip()
        if not lyrics.strip() and not instrumental: raise ValueError("No hay transcripción; no se publica una letra inventada")
        if len(lyrics) > 100000: raise ValueError("La transcripción supera el límite permitido")
        metadata = dict(title=str(info.get("track") or info.get("title") or "")[:500],
            artist=str(info.get("artist") or info.get("uploader") or "")[:300], tag="", year="", lyrics=lyrics,
            lyrics_source=source, artist_source="video_metadata", year_source="unknown",
            transcript_coverage="unverified",
            embedding_coverage="lyrics_chunks_2000_mean;audio_start_middle_end_10s_mean")
        metadata.update(job["metadata"])
        if "year" in job["metadata"]: metadata["year_source"] = "user_provided"
        if "artist" in job["metadata"]: metadata["artist_source"] = "user_provided"
        progress("embedding_lyrics", 65)
        lv = None
        metadata["lyrics_availability"] = "available" if lyrics.strip() else "not_requested_user_declared_instrumental"
        if instrumental:
            metadata["lyrics_source"] = "none_user_declared_instrumental"
            metadata["instrumental_source"] = "user_declared_not_acoustically_verified"
            metadata["transcript_coverage"] = "not_requested"
            metadata["embedding_coverage"] = "audio_start_middle_end_10s_mean"
        if lyrics.strip():
            import semantic_search
            resources = semantic_search._load_resources()
            vectors = []
            for start in range(0, len(lyrics), 2000):
                with semantic_search._model_lock:
                    vectors.append(unit_vector(resources["embedder"].encode("search_document: " + lyrics[start:start+2000],
                        normalize_embeddings=True)[:256], 256))
            lv = unit_vector(np.mean(vectors, axis=0), 256)
        progress("embedding_audio", 80)
        import clap_search
        import torch
        resources = clap_search._load_resources()
        vectors = []
        for offset in sorted({0., max(0., duration/2-5), max(0., duration-10)}):
            raw = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(audio),
                "-t", "10", "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1"],
                check=True, capture_output=True, timeout=60).stdout
            samples = np.frombuffer(raw, dtype="<f4").copy()
            if not samples.size: raise ValueError("No se pudo decodificar el audio")
            with clap_search._model_lock, torch.no_grad():
                inputs = resources["processor"](audio=samples, sampling_rate=48000, return_tensors="pt").to(clap_search.DEVICE)
                output = resources["model"].get_audio_features(**inputs)
                tensor = output if isinstance(output, torch.Tensor) else output.pooler_output if hasattr(output, "pooler_output") else output[0]
                vectors.append(unit_vector(tensor.squeeze().cpu().float().numpy(), 512))
        return metadata, lv, unit_vector(np.mean(vectors, axis=0), 512), audio
