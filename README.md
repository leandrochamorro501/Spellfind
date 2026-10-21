# SPELLFIND

*Proyecto universitario para asignatura de LLMs

Buscador local de música por letra y sonido. Requiere **Windows x64, Python 3.12**, conexión a Internet para obtener los modelos y una clave de **Groq**.

## Ejecutar

En PowerShell, desde esta carpeta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env` y coloca tu clave en `GROQ_API_KEY`. Para buscar, aporta tu propio catálogo en `output/` con `datos_alineados.csv`, `song_id_map.json`, `faiss_lyrics.index` y `faiss_audio.index`; los datos y letras no se publican en el repositorio.

```powershell
.\.venv\Scripts\python.exe app.py
```

Abre <http://127.0.0.1:5000/>. La primera búsqueda puede descargar pesos de modelos y tardar más. Para activar la ingesta opcional desde enlaces de YouTube, sigue [INSTALL.md](INSTALL.md). La documentación de agentes está en [documentacion/agentes_multimedia/main.tex](documentacion/agentes_multimedia/main.tex).
