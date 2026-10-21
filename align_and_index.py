"""
align_and_index.py — SPELLFIND
====================================
Descarga los 4 datasets desde Hugging Face, calcula la intersección de
song_ids válidos (canciones con metadata + lyrics_embedding + clap_embedding
+ archivo .mp3 disponible), exporta datos_alineados.csv y construye los dos
índices FAISS listos para ser cargados por el servidor Flask.

Uso:
    pip install datasets pandas faiss-cpu numpy huggingface_hub
    python align_and_index.py

    Para datasets privados, el token se lee automáticamente desde la variable
    de entorno HF_TOKEN, o bien puedes autenticarte una vez con:
        huggingface-cli login
    y el token queda guardado en ~/.cache/huggingface/token.

Salidas generadas en ./output/:
    datos_alineados.csv          — DataFrame limpio con las N canciones alineadas
    faiss_lyrics.index           — Índice FAISS de embeddings de letras (Nomic)
    faiss_audio.index            — Índice FAISS de embeddings de audio (CLAP)
    song_id_map.json             — Mapeo posición_en_índice → song_id (para lookup)
"""

import json
import os
import ast
from console import safe_print as print
import numpy as np
import pandas as pd
import faiss
from datasets import load_dataset
from huggingface_hub import login, HfApi

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

HF_REPOS = {
    "metadata":          "aleshitajaja/sonic-search-data",
    "lyrics_embeddings": "aleshitajaja/sonic-search-lyrics-embeddings",
    "audio_embeddings":  "aleshitajaja/sonic-search-audio-embeddings",
    "audio_previews":    "aleshitajaja/sonic-search-audio-previews",
}

from config import OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# AUTENTICACIÓN — DATASETS PRIVADOS
# ──────────────────────────────────────────────────────────────────────────────

def authenticate():
    """
    Autentica con Hugging Face para acceder a repositorios privados.
    Prioridad:
      1. Variable de entorno HF_TOKEN  (ideal para CI/scripts desatendidos)
      2. Token guardado por `huggingface-cli login`  (ideal para uso local)
    Si ninguno está disponible, solicita el token por consola.
    """
    token = os.environ.get("HF_TOKEN")

    if token:
        print("[AUTH] Usando token desde variable de entorno HF_TOKEN.")
        login(token=token)
        return

    # Verificar si ya hay un token guardado en cache (~/.cache/huggingface/token)
    try:
        api = HfApi()
        user = api.whoami()          # lanza excepción si no hay token en cache
        print(f"[AUTH] Sesión activa como: {user['name']}")
        return
    except Exception:
        pass

    # Último recurso: pedirlo por consola (no se muestra al escribir)
    import getpass
    print("[AUTH] No se encontró token de Hugging Face.")
    print("       Opciones:")
    print("         a) Ejecuta:  huggingface-cli login")
    print("         b) Define:   export HF_TOKEN=hf_...")
    print("         c) Ingrésalo ahora (no se mostrará en pantalla):")
    token = getpass.getpass("  HF Token: ").strip()
    if token:
        login(token=token)
    else:
        raise EnvironmentError(
            "Token requerido para acceder a datasets privados. "
            "Ejecuta `huggingface-cli login` o define la variable HF_TOKEN."
        )

# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 — DESCARGA DE DATASETS DESDE HUGGING FACE
# ──────────────────────────────────────────────────────────────────────────────

import time

def _load_with_retry(repo: str, split: str = "train", retries: int = 4, wait: int = 5):
    """
    Envuelve load_dataset con reintentos automáticos para tolerar
    interrupciones de red transitorias (WinError 10054, timeouts, etc.).
    """
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(repo, split=split)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"    ⚠ Intento {attempt}/{retries} falló: {e}")
            print(f"    Reintentando en {wait}s...")
            time.sleep(wait)


def download_datasets() -> dict[str, pd.DataFrame | set]:
    """
    Descarga los 4 datasets y los devuelve como DataFrames o sets según el caso.
    El dataset de audio_previews solo necesita los song_ids (nombres de archivo).
    """
    print("\n[1/4] Descargando dataset de metadata (sonic-search-data)...")
    ds_meta = _load_with_retry(HF_REPOS["metadata"], split="songs")
    df_meta = ds_meta.to_pandas()

    # El índice del DataFrame es el song_id — lo normalizamos como columna explícita
    # para evitar ambigüedades al hacer merge.
    if "song_id" not in df_meta.columns:
        # Si el índice tiene nombre, lo usamos; si no, asumimos que es el índice numérico
        df_meta = df_meta.reset_index()
        df_meta.rename(columns={"index": "song_id"}, inplace=True)

    df_meta["song_id"] = df_meta["song_id"].astype(int)
    print(f"    → {len(df_meta)} canciones en metadata. Columnas: {list(df_meta.columns)}")

    print("\n[2/4] Descargando lyrics embeddings...")
    ds_lyrics = _load_with_retry(HF_REPOS["lyrics_embeddings"])
    df_lyrics = ds_lyrics.to_pandas()
    df_lyrics["song_id"] = df_lyrics["song_id"].astype(int)
    print(f"    → {len(df_lyrics)} entradas. Columnas: {list(df_lyrics.columns)}")

    print("\n[3/4] Descargando audio embeddings (CLAP)...")
    ds_audio = _load_with_retry(HF_REPOS["audio_embeddings"])
    df_audio = ds_audio.to_pandas()
    df_audio["song_id"] = df_audio["song_id"].astype(int)
    print(f"    → {len(df_audio)} entradas. Columnas: {list(df_audio.columns)}")

    print("\n[4/4] Listando archivos .mp3 disponibles (sin descargar)...")
    audio_ids = _list_audio_ids_from_repo(HF_REPOS["audio_previews"])
    print(f"    → {len(audio_ids)} archivos .mp3 disponibles")

    return {
        "meta":    df_meta,
        "lyrics":  df_lyrics,
        "audio":   df_audio,
        "mp3_ids": audio_ids,
    }


def _list_audio_ids_from_repo(repo: str) -> set[int]:
    """
    Lista los archivos del repositorio de audio en HF usando la API HTTP
    (sin descargar ningún .mp3). Extrae los song_ids desde los nombres de archivo.
    """
    from huggingface_hub import list_repo_files
    audio_ids = set()
    try:
        files = list_repo_files(repo, repo_type="dataset")
        for filename in files:
            # Los archivos están como "114.mp3" o "data/114.mp3"
            song_id = _parse_id_from_filename(filename)
            if song_id is not None:
                audio_ids.add(song_id)
    except Exception as e:
        print(f"    ⚠ Error listando archivos del repo: {e}")
        print("    Se continuará sin filtrar por .mp3 disponibles.")
    return audio_ids


def _extract_audio_ids(ds_previews) -> set[int]:
    """
    Extrae los song_ids numéricos desde el dataset de audio previews.
    Soporta dos formatos comunes en HF:
      - Dataset con columna 'file_name' o 'audio' (struct con 'path')
      - Dataset de tipo Audio con columna 'audio'
    """
    df = ds_previews.to_pandas()
    audio_ids = set()

    # Caso A: columna explícita con nombre de archivo
    for col in ["file_name", "filename", "path", "name"]:
        if col in df.columns:
            for val in df[col].dropna():
                song_id = _parse_id_from_filename(str(val))
                if song_id is not None:
                    audio_ids.add(song_id)
            if audio_ids:
                return audio_ids

    # Caso B: columna 'audio' con diccionario {'path': '...', 'bytes': ...}
    if "audio" in df.columns:
        for val in df["audio"].dropna():
            path = None
            if isinstance(val, dict):
                path = val.get("path") or val.get("file_name")
            elif isinstance(val, str):
                path = val
            if path:
                song_id = _parse_id_from_filename(path)
                if song_id is not None:
                    audio_ids.add(song_id)
        if audio_ids:
            return audio_ids

    # Caso C: columna 'song_id' directa
    if "song_id" in df.columns:
        return set(df["song_id"].dropna().astype(int).tolist())

    print("    ⚠ No se pudo extraer song_ids del dataset de previews. Verifica su estructura.")
    print(f"    Columnas disponibles: {list(df.columns)}")
    return audio_ids


def _parse_id_from_filename(filename: str) -> int | None:
    """Extrae el número de un nombre de archivo tipo '213.mp3' → 213."""
    try:
        basename = os.path.basename(filename)
        return int(os.path.splitext(basename)[0])
    except (ValueError, AttributeError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 — FILTRADO POR INTERSECCIÓN
# ──────────────────────────────────────────────────────────────────────────────

def compute_intersection(data: dict) -> pd.DataFrame:
    """
    Calcula la intersección de song_ids válidos en los 4 conjuntos de datos
    y devuelve el DataFrame alineado con metadata + embeddings.

    La canción queda incluida si y solo si:
      ✓ Existe en el DataFrame de metadata (sonic-search-data)
      ✓ Tiene un lyrics_embedding en sonic-search-lyrics-embeddings
      ✓ Tiene un clap_embedding en sonic-search-audio-embeddings
      ✓ Tiene un archivo .mp3 en sonic-search-audio-previews
    """
    print("\n[ALINEACIÓN] Calculando intersección de song_ids...")

    ids_meta   = set(data["meta"]["song_id"].tolist())
    ids_lyrics = set(data["lyrics"]["song_id"].tolist())
    ids_audio  = set(data["audio"]["song_id"].tolist())
    ids_mp3    = data["mp3_ids"]

    print(f"    song_ids en metadata:          {len(ids_meta)}")
    print(f"    song_ids en lyrics_embeddings: {len(ids_lyrics)}")
    print(f"    song_ids en audio_embeddings:  {len(ids_audio)}")
    print(f"    song_ids con .mp3:             {len(ids_mp3)}")

    # Intersección de los 4 conjuntos
    valid_ids = ids_meta & ids_lyrics & ids_audio & ids_mp3

    print(f"\n    ✓ Canciones alineadas (en los 4 conjuntos): {len(valid_ids)}")

    if len(valid_ids) == 0:
        raise ValueError(
            "La intersección está vacía. Verifica que los song_ids sean consistentes "
            "entre los 4 datasets. Revisa el formato del dataset de audio_previews."
        )

    # Diagnóstico de pérdidas (útil para debugging)
    _print_loss_report(ids_meta, ids_lyrics, ids_audio, ids_mp3, valid_ids)

    # Filtrar cada DataFrame
    df_meta   = data["meta"][data["meta"]["song_id"].isin(valid_ids)].copy()
    df_lyrics = data["lyrics"][data["lyrics"]["song_id"].isin(valid_ids)].copy()
    df_audio  = data["audio"][data["audio"]["song_id"].isin(valid_ids)].copy()

    # Merge secuencial sobre song_id
    df_aligned = df_meta.merge(df_lyrics, on="song_id", how="inner")
    df_aligned = df_aligned.merge(df_audio, on="song_id", how="inner")

    # Ordenar por song_id para garantizar consistencia con los índices FAISS
    df_aligned = df_aligned.sort_values("song_id").reset_index(drop=True)

    print(f"\n    DataFrame alineado: {len(df_aligned)} filas × {len(df_aligned.columns)} columnas")
    return df_aligned


def _print_loss_report(ids_meta, ids_lyrics, ids_audio, ids_mp3, valid_ids):
    """Imprime un diagnóstico de qué canciones se pierden y por qué."""
    lost_no_lyrics  = ids_meta - ids_lyrics
    lost_no_audio   = ids_meta - ids_audio
    lost_no_mp3     = ids_meta - ids_mp3
    lost_any        = ids_meta - valid_ids

    print(f"\n    [Diagnóstico de pérdidas desde metadata]")
    print(f"    Sin lyrics_embedding:  {len(lost_no_lyrics)} canciones")
    print(f"    Sin clap_embedding:    {len(lost_no_audio)} canciones")
    print(f"    Sin archivo .mp3:      {len(lost_no_mp3)} canciones")
    print(f"    Pérdida total:         {len(lost_any)} canciones eliminadas")


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 — EXPORTAR datos_alineados.csv
# ──────────────────────────────────────────────────────────────────────────────

def export_aligned_csv(df_aligned: pd.DataFrame, directory=OUTPUT_DIR) -> str:
    """Guarda el DataFrame alineado en CSV sin los vectores (demasiado pesados para CSV)."""
    # Columnas de metadata solamente para el CSV legible
    meta_cols = ["song_id", "title", "artist", "tag", "year", "lyrics"]
    available = [c for c in meta_cols if c in df_aligned.columns]

    csv_path = os.path.join(directory, "datos_alineados.csv")
    df_aligned[available].to_csv(csv_path, index=False)
    print(f"\n[EXPORTAR] datos_alineados.csv guardado en: {csv_path}")
    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# PASO 4 — CONSTRUCCIÓN DE ÍNDICES FAISS
# ──────────────────────────────────────────────────────────────────────────────

def _parse_embedding(value) -> np.ndarray | None:
    """
    Convierte el valor de un embedding almacenado en HF a numpy array.
    Soporta: list, np.ndarray, string JSON/Python repr.
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return np.array(parsed, dtype=np.float32)
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(value)
            return np.array(parsed, dtype=np.float32)
        except (ValueError, SyntaxError):
            pass
    return None


def build_faiss_indices(df_aligned: pd.DataFrame, directory=OUTPUT_DIR) -> dict:
    """
    Construye dos índices FAISS (IndexFlatIP = producto interno, equivalente
    a similitud coseno cuando los vectores están normalizados L2).

    Retorna las rutas de los archivos generados.
    """
    print("\n[FAISS] Extrayendo matrices de embeddings...")

    # ── Lyrics (Nomic) ────────────────────────────────────────────────────────
    lyrics_col = "lyrics_embedding"
    if lyrics_col not in df_aligned.columns:
        raise ValueError(f"Columna '{lyrics_col}' no encontrada en el DataFrame alineado.")

    lyrics_vectors = np.stack(
        [_parse_embedding(v) for v in df_aligned[lyrics_col]], axis=0
    ).astype(np.float32)

    # ── Audio (CLAP) ──────────────────────────────────────────────────────────
    audio_col = "clap_embedding"
    if audio_col not in df_aligned.columns:
        raise ValueError(f"Columna '{audio_col}' no encontrada en el DataFrame alineado.")

    audio_vectors = np.stack(
        [_parse_embedding(v) for v in df_aligned[audio_col]], axis=0
    ).astype(np.float32)

    dim_lyrics = lyrics_vectors.shape[1]
    dim_audio  = audio_vectors.shape[1]
    print(f"    Dimensiones lyrics (Nomic): {lyrics_vectors.shape}  — dim={dim_lyrics}")
    print(f"    Dimensiones audio  (CLAP):  {audio_vectors.shape}  — dim={dim_audio}")

    # Normalización L2 para que IndexFlatIP = similitud coseno
    faiss.normalize_L2(lyrics_vectors)
    faiss.normalize_L2(audio_vectors)

    # ── Crear índices ─────────────────────────────────────────────────────────
    print("\n[FAISS] Construyendo índices IndexFlatIP...")

    index_lyrics = faiss.IndexFlatIP(dim_lyrics)
    index_lyrics.add(lyrics_vectors)
    print(f"    ✓ faiss_lyrics.index — {index_lyrics.ntotal} vectores indexados")

    index_audio = faiss.IndexFlatIP(dim_audio)
    index_audio.add(audio_vectors)
    print(f"    ✓ faiss_audio.index  — {index_audio.ntotal} vectores indexados")

    # ── Guardar índices ───────────────────────────────────────────────────────
    lyrics_path = os.path.join(directory, "faiss_lyrics.index")
    audio_path  = os.path.join(directory, "faiss_audio.index")

    faiss.write_index(index_lyrics, lyrics_path)
    faiss.write_index(index_audio, audio_path)
    print(f"\n    Índice de letras guardado en:  {lyrics_path}")
    print(f"    Índice de audio guardado en:   {audio_path}")

    # ── Mapeo posición → song_id ──────────────────────────────────────────────
    # CRÍTICO: el orden del DataFrame (ya ordenado por song_id) determina la
    # posición en el índice FAISS. Este JSON es el "diccionario de traducción"
    # para saber qué canción corresponde a cada resultado de búsqueda.
    song_id_map = {
        str(i): int(song_id)
        for i, song_id in enumerate(df_aligned["song_id"].tolist())
    }
    map_path = os.path.join(directory, "song_id_map.json")
    with open(map_path, "w") as f:
        json.dump(song_id_map, f, indent=2)
    print(f"    Mapa de song_ids guardado en:  {map_path}")

    return {
        "lyrics_index_path": lyrics_path,
        "audio_index_path":  audio_path,
        "song_id_map_path":  map_path,
        "n_songs":           index_lyrics.ntotal,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────

def main():
    from catalog import default_manager
    from tempfile import TemporaryDirectory
    manager = default_manager()
    base_version = manager.published_version()
    print("=" * 60)
    print("  SPELLFIND — Pipeline de Alineación e Indexación")
    print("=" * 60)

    # 0. Autenticar (necesario para repos privados)
    authenticate()

    # 1. Descargar
    data = download_datasets()

    # 2. Alinear por intersección
    df_aligned = compute_intersection(data)

    # 3. Exportar CSV legible
    with TemporaryDirectory(prefix="spellfind-build-") as stage:
        export_aligned_csv(df_aligned, stage)
        result = build_faiss_indices(df_aligned, stage)
        snapshot = manager.publish_directory(stage, expected_version=base_version)

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  Pipeline completado exitosamente")
    print("=" * 60)
    print(f"  Canciones alineadas:     {result['n_songs']}")
    print(f"  Generación publicada:    {snapshot.version}")
    print(f"  Directorio:              {manager.root / 'versions' / snapshot.version}")
    print("  Incluye índices de letras/audio, mapa, catálogo y manifest SHA-256.")
    print("  Las nuevas consultas adoptarán la generación publicada.")
    print("=" * 60)


if __name__ == "__main__":
    main()
