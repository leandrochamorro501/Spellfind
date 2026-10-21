# SPELLFIND — instalación y operación verificables

Perfil verificado: **Windows x64, CPython 3.12**. Los locks contienen versiones
exactas de las dependencias y sus transitivas, resueltas e importadas en un
entorno nuevo. No son locks multiplataforma ni fijan los pesos remotos.

## Instalación

Desde la carpeta SPELLFIND, con Python 3.12 instalado:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
Copy-Item .env.example .env
```

Editar `.env` para configurar `GROQ_API_KEY`. No compartirlo ni añadirlo a Git.
El instalador alternativo `python install_dependencies.py` utiliza el mismo lock
y exige un entorno virtual; ya no mantiene una segunda lista de paquetes.

Perfiles:

- `requirements/core.lock`: web, grafo y validación del catálogo; no carga modelos.
- `requirements/models.lock`: ejecución completa de inferencia; lo referencia
  `requirements.txt` y también `requirements-dev.txt` (unittest viene con Python).
- `requirements/indexing.lock`: añade las herramientas para reconstruir datasets.
  Instalarlo con `python -m pip install -r requirements/indexing.lock`.
- Los archivos `.in` declaran dependencias directas y rangos para resolver futuras
  actualizaciones. No sustituyen los locks en una instalación reproducible.

No reutilizar `tmp/test-deps` de sesiones anteriores. Para regenerar locks,
resolver `requirements/indexing.in` en un entorno vacío, ejecutar `pip check`,
verificar imports y correr `python scripts/lock_environment.py`. Ese script obtiene
cierres de dependencias con sus marcadores/extras. Revisar y probar el nuevo lock
antes de sustituir el entorno que funciona. En otros sistemas o versiones de
Python se requiere otra resolución y validación; no se certifican aquí.

## Comprobación sin descargar pesos

```powershell
python scripts/verify_environment.py --models
python -m unittest discover -s tests -v
```

El verificador importa Flask, FAISS, LangGraph, Torch, Transformers, Nomic/CLAP y
valida los datos locales, sin llamar a `from_pretrained` ni al proveedor.
Para verificar el indexador, instalar su perfil y añadir `--indexing`.
La primera búsqueda real puede descargar los pesos de los modelos; la biblioteca
instalada no equivale a tenerlos en caché. Groq necesita una clave válida.

## Arranque

Desarrollo local, con debug desactivado por defecto:

```powershell
python app.py
```

Servicio WSGI en Windows:

```powershell
waitress-serve --listen=127.0.0.1:5000 wsgi:app
```

También puede usarse la fábrica con `flask --app app:create_app run`. La CLI
continúa disponible: `python cli.py "canción triste con piano"`.

La fábrica inicializa y valida el catálogo sin descargar modelos. Las rutas de
catálogo, prompt y salida del indexador se resuelven desde el proyecto, no desde
la carpeta actual. `.env` se lee desde la carpeta del proyecto y no reemplaza
variables de entorno ya configuradas.

Configuración documentada en `.env.example`: host, puerto, debug, longitud máxima
de consulta, ubicación de catálogo y prompt. Waitress utiliza los argumentos de
su comando para host/puerto; `SPELLFIND_HOST` y `SPELLFIND_PORT` corresponden al
arranque `python app.py`. Reiniciar después de cambiar configuración o catálogo.
La publicación en caliente de índices sigue pendiente en P17.

## Contrato HTTP

- `GET /`: interfaz.
- `GET /health`: 200 si el catálogo está listo y 503 si está ausente/inválido.
  No certifica credenciales, disponibilidad del proveedor ni pesos descargados.
- `POST /search`: objeto JSON con `query` textual no vacía. 400 para JSON,
  tipo o longitud inválidos; 413 para cuerpo excesivo; 503 para motor/proveedor
  no disponible o error controlado del flujo; 500 para fallo inesperado.
- El bloque II añade `retrieved` (solo audio, sin validar), `partially_validated`
  (letra aprobada, audio sin validar) y `clarification_needed` (reformular sin búsqueda).
  Estos estados devuelven 200 con evidencia y advertencia.
- Un flujo terminado sin aprobación por límite o candidatos devuelve 200 con
  estado y advertencia; no se marca como aprobado.

Los errores inesperados se registran en el servidor y devuelven un mensaje
público genérico. No hay ingesta desde YouTube en este bloque.

## Evaluación y evidencia

Ver `evaluation/README.md`. Los informes del bloque están en `validation/block1/`.
La línea base simulada no acredita relevancia musical; qrels necesita anotaciones
humanas y la ejecución real necesita clave y pesos. Las pruebas del bloque no
consumen la API de Groq.

## Continuación del bloque III

`SPELLFIND_ENABLE_REFINEMENT=false` es el valor por defecto. Activarlo requiere
reiniciar el proceso y permite refinamiento experimental tras rechazo lírico;
la calibración humana y su beneficio musical todavía no están medidos.

La API añade `retrieval`, `degradations`, `budgets`, `counters`, `budget_reason`
y `catalog_exhausted: false`. Puede responder 200 con resultados degradados y
advertencia si una rama falla; si no hay resultados utilizables y hay fallos,
responde 503. El fallo del refinador conserva el plan válido. Si falla toda una
recuperación reformulada, restaura el último plan/ranking obtenidos juntos.
`validation.lyrics.status=abstained` expresa incertidumbre, sin aprobación ni
rechazo semántico inventados. El fin de una ventana no declara agotamiento global.

Los mensajes de inicialización soportan CP1252 y UTF-8. Para conservar símbolos
Unicode en consola puede usarse `python -X utf8 ...`; las pruebas de ambas
codificaciones se reproducen con `python scripts/verify_block3.py`.
Evidencia y límites: `validation/block3/README.md`.

## Pesos, filtros y resultados — API v2

`POST /search` mantiene `query` y admite opciones:

```json
{
  "query": "Una letra sobre volver a casa con piano",
  "weights": {"lyrics": 0.7, "audio": 0.3},
  "filters": {"artist": "Manuela Tome", "genre": "pop", "year_min": 2020, "year_max": 2020}
}
```

Las opciones son independientes y opcionales. Omitir pesos usa la política
automática. Ambos pesos deben ser números finitos entre 0 y 1 y sumar 1; un peso
cero desactiva esa recuperación, sin borrar restricciones del plan. Si ninguna
modalidad solicitada tiene peso positivo, se informa ausencia de resultados.
Los pesos se reajustan entre ramas útiles con peso positivo; se informan tanto
los solicitados como los efectivos. No se fuerza una modalidad ausente del plan.

Artista y género coinciden completos, con Unicode NFC, sin distinguir mayúsculas
ni espacios exteriores; conservan acentos. El género usa el campo `tag`. Los
años son enteros 1–9999, inclusivos; campos ausentes no cumplen un filtro temporal.
Los filtros usan AND. Tipos, claves y rangos inválidos devuelven 400. Una selección
sin coincidencias devuelve 200, `termination_reason=no_filter_matches`, sin modelos.
`GET /catalog/options` devuelve artistas, géneros y extremos temporales; 503 si
el catálogo no está listo. Con filtros se recupera hasta el tamaño del catálogo
para no perder canciones fuera del top inicial; revisar esta política al escalar.

La respuesta añade `api_version=2`, `score_method`, `score_is_probability=false`,
`score_components`, `lyric_excerpt`, `filter_summary` y `timings`/`retrieval_history`.
**Se elimina `score_pct`:** migrar clientes a `score` y sus aportes. La escala
0–1 es una fusión de posiciones, no una probabilidad. Los scores de versiones
distintas no son comparables directamente. Una recomendación lírica puede ir
primero aunque otra tenga mayor score; `selection_reason` lo explica.
Los fragmentos distinguen citas del juez y extractos orientativos. Audio sin
validación independiente y alternativas no juzgadas conservan ese estado.
Los tiempos son del servicio y no sustituyen latencia de red o factura del proveedor.

La CLI utiliza los mismos contratos:

```powershell
python -X utf8 cli.py "Una letra sobre volver con piano" --lyrics-weight 0.7 --artist "Manuela Tome" --genre pop --year-min 2020 --year-max 2020
```

La interfaz incluye estos controles. Para verificarla con un fixture local
explícitamente simulado: `python scripts/preview_block4.py` (127.0.0.1:5074).
Ese servidor muestra un aviso de control y **no ofrece juicios musicales reales**;
la aplicación normal se inicia con `python app.py` o el WSGI documentado.

Pruebas: `python scripts/verify_block4.py`. Perfil local:
`python -X utf8 -m evaluation.profile --iterations 20 --output validation/block4/profile.json`.
El perfil usa índices reales y encoders de control: inferencia, RSS/GPU y modelos
reales siguen pendientes. No se necesitan dependencias adicionales para el bloque.
# Catálogo e ingesta opcional (bloque V)

**Estado actualizado:** las indicaciones experimentales de esta sección se
completan con este procedimiento verificado para Windows x64/Python 3.12:

```powershell
python -m venv .venv-ingestion
.\.venv-ingestion\Scripts\python.exe -m pip install -r requirements/ingestion.lock
.\.venv-ingestion\Scripts\python.exe -m pip check
.\.venv-ingestion\Scripts\python.exe scripts/provision_ffmpeg.py
.\.venv-ingestion\Scripts\python.exe app.py
```

`requirements/models.lock` también se actualizó a Transformers 4.57.6: Nomic falla
con Transformers 5. El entorno histórico `.venv-block1` conserva sus paquetes
anteriores; usa el entorno actualizado para inferencia real. No se modifica el
PATH global. Los pesos se cachean en `.tools/huggingface` y `.tools/whisper` y no
forman parte de las copias de catálogo. `HF_HOME` existente tiene precedencia.

Para instrumentales añade `"instrumental":true` al POST o usa la casilla de la
interfaz. No se fabrican letras ni vectores líricos; la ausencia de voces es una
declaración, no una detección automática. El mapa v2 es independiente por modalidad.

Mantenimiento, preferentemente después de completar los trabajos de ingesta:

```powershell
python -m maintenance cleanup
python -m maintenance backup --path backups/catalogo.zip
python -m maintenance cleanup --apply --days 30 --keep-versions 3
python -m maintenance restore --path backups/catalogo.zip --destination data/restauracion-nueva
python -m maintenance purge-trash
```

`cleanup` conserva siempre la generación actual y medios de versiones retenidas.
`--apply` caduca enlaces antiguos y mueve archivos a cuarentena; todavía ocupan
espacio. El período de recuperación de 30 días comienza al moverlos. El borrado
definitivo exige `purge-trash --apply` y una copia previa es recomendable.
No se ejecuta limpieza destructiva automáticamente. Las cuotas predeterminadas
son catálogo 512 MiB e ingesta/copia 2 GiB, ajustables mediante las variables
`SPELLFIND_*_QUOTA_BYTES` de `.env.example`. Son controles de admisión/publicación,
no una cuota dura del sistema operativo. Las copias repetidas en una carpeta
de backups requieren su propia política de espacio.

La restauración exige carpeta inexistente. Para adoptar una restauración, detén
el servicio, configura `SPELLFIND_CATALOG_DIR=<destino>/catalog` y
`SPELLFIND_INGEST_DIR=<destino>/ingestion`, y vuelve a arrancarlo. No se reemplaza
automáticamente el catálogo actual. La copia incluye trabajos, catálogo y medios
referenciados; excluye configuración privada y claves. Los hashes no autentican
copias de origen desconocido.

Los párrafos siguientes conservan el contexto de la implementación inicial del
bloque V; el lock y la ruta local de FFmpeg descritos arriba son el estado actual.

`SPELLFIND_CATALOG_DIR=data/catalog` y `SPELLFIND_INGEST_DIR=data/ingestion`
son rutas relativas al proyecto. El primer acceso al catálogo copia los cuatro
artefactos de `output/` a una generación validada. No borres `data/` si necesitas
conservar canciones añadidas, trabajos y enlaces de audio. La reconstrucción
`align_and_index.py` publica una generación; rechaza reemplazarla si hubo una
actualización concurrente. El módulo `maintenance` proporciona limpieza explícita y restauración verificada.

La búsqueda funciona con `SPELLFIND_INGEST_ENABLED=false` (predeterminado).
Para probar ingesta, prepara un entorno separado siguiendo los requisitos de
modelos e instala `requirements/ingestion.lock`; aprovisiona FFmpeg con el script
indicado arriba. `ingestion.in` declara dependencias y el lock fija las versiones verificadas.
Se necesitan además pesos Nomic/CLAP y Whisper para vídeos sin subtítulos útiles.
Activa `SPELLFIND_INGEST_ENABLED=true` y reinicia el servidor local.

`GET /ingest/status` indica herramientas disponibles; no verifica los pesos.
`POST /ingest` acepta JSON `{"url":"https://www.youtube.com/watch?v=VIDEO_ID_11"}`,
con `artist`, `genre` y `year` opcionales (año entero). Devuelve 202 y `status_url`.
Consulta `GET /ingest/jobs/<id>` para estado, etapa, progreso, error o resultado.
No uses el ejemplo de ID como vídeo real. Sólo se admiten vídeos individuales
HTTPS, no directos, con duración conocida hasta 15 minutos.

La interfaz guarda el último ID para consultar tras recargar. Los trabajos
pendientes pueden retomarse al iniciar el servicio habilitado; los trabajos en
ejecución cuyo lease vence quedan `interrupted`. Vuelve a enviar el enlace para
reintentar: si ya se publicó, se devuelve la canción existente. Enviar de nuevo
un enlace no actualiza metadatos de una canción ya publicada.

Los instrumentales declarados pueden añadirse sin transcripción. El año de subida
no se trata como año de lanzamiento. Conserva el host local predeterminado:
la protección de origen no proporciona autenticación para un servicio público.
La continuación y los límites vigentes están en `documentacion/trabajo_autonomo.tex`.

