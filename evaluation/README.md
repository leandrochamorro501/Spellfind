# Línea base P04

Referencia: `block1-pre-P05`, después de las validaciones de P01–P03 y antes
de cambiar la semántica en los bloques siguientes. No es una reproducción de
las presentaciones históricas ni una medida del código original A/B/C.

`queries.json` fija 12 consultas: dos por categoría (letras, audio, mixtas,
literales, negativas e instrumentales). Los planes del fixture se usan solo en
simulación. El modo real utiliza el router y los modelos configurados.

Desde la carpeta del proyecto, en el entorno verificado:

```powershell
python -m evaluation.run --mode simulated --repeats 3 --output evaluation/runs/simulated.json
python -m evaluation.run --mode simulated --repeats 3 --refinement --output evaluation/runs/simulated-refined.json
python -m evaluation.run --mode real --repeats 3 --output evaluation/runs/real.json
python -m evaluation.run --mode real --repeats 3 --refinement --output evaluation/runs/real-refined.json
```

La simulación utiliza datos locales reales para identificar canciones, rankings
fabricados de forma determinista y veredictos simulados. Solo valida el control,
estabilidad del fixture y reutilización de ramas. Sus tiempos no miden Nomic,
CLAP o Groq, y sus resultados nunca reciben métricas de relevancia musical.

El modo real necesita `GROQ_API_KEY`, acceso a los modelos y sus pesos. Si falta
la clave, escribe un informe `blocked` y no llama al proveedor. La ejecución real
puede consumir la cuota del proveedor; se elige de forma explícita con `--mode real`.

Cada informe registra hashes de fuentes, datos y locks, versiones del entorno,
planes, IDs, scores, estados, exclusiones, trazas, latencia y contadores en los
límites del servicio. Esos contadores no equivalen al número de llamadas HTTP del
proveedor: sus reintentos internos no están instrumentados. El coste monetario
queda desconocido; las revisiones remotas de los modelos aún no están fijadas.
La primera ejecución se etiqueta como primera del proceso, no como caché de disco
fría garantizada. Para comparaciones de latencia usar procesos separados con el
mismo hardware, snapshots, configuración y políticas de carga.

## Juicios humanos

`qrels.json` empieza vacío y mantiene `pending_human_annotation`. No se inventan
canciones relevantes. Para cada ID de consulta, añadir:

```json
{
  "catalog_complete": false,
  "grades": {},
  "negative_violations": {}
}
```

Las claves de ambos mapas son IDs de canciones del catálogo, como cadenas.
Grados: 0 no relevante, 1 relación débil, 2 relevante, 3 muy relevante.
`negative_violations` indica con booleanos si se viola una restricción explícita;
revisar por modalidad, escuchando el audio cuando se evalúe sonido. Registrar
identidad del anotador y motivo fuera de los campos de puntuación si se desea.

No marcar `catalog_complete` hasta que `grades` cubra el catálogo entero. Recall
y nDCG permanecen nulos sin esa cobertura y sin juicios para todo el top-K; el
protocolo rechaza una cobertura completa declarada falsamente. La tasa de
violaciones requiere juicios explícitos para cada resultado. Un dato sin anotar
es desconocido, nunca un cero automático. Comparar baseline y refinamiento con
el mismo qrels y al menos tres repeticiones. Jaccard mide coincidencia de conjuntos,
no calidad del orden ni satisfacción del usuario.

La medición real y la anotación humana siguen pendientes si el entorno no dispone
de credenciales o pesos; una simulación correcta no cierra esas verificaciones.

## Capturas después del bloque II

Las capturas históricas del bloque II usan `block2-pre-P09`; `--baseline-id`
permite nombrar una captura nueva. No sobrescribir los JSON históricos del
bloque I al ejecutar código posterior. Las consultas y sus IDs se conservaron.

Los informes actuales incluyen `status_counts` y separan
`historical_reconsiderations` de `excluded_reappearances`: un candidato rechazado
bajo un plan puede reevaluarse bajo otro; no debe reaparecer sin esa reevaluación.
Las consultas que necesitan reformulación no tienen ranking, por lo que Jaccard
queda nulo cuando ambos resultados están vacíos. Esto no cuenta como estabilidad
perfecta ni como ausencia de errores semánticos.

Dos planes del fixture antiguo carecen de citas para sus negaciones y la
interpretación local no cubre toda la frase: ahora solicitan reformulación.
Registrar esa menor cobertura al comparar bloques; no sustituirla por un resultado
musical inventado ni modificar silenciosamente el fixture histórico.

## Comparación después del bloque III

Las capturas de ese bloque usan `block3-post-P12`. Cada corrida fija explícitamente la
política de refinamiento, independientemente del `.env`: desactivada sin
`--refinement` y experimental con esa opción. La aplicación normal la mantiene
desactivada por defecto (`SPELLFIND_ENABLE_REFINEMENT=false`). No sobrescribir
las capturas de bloques anteriores.

```powershell
python -X utf8 -m evaluation.run --mode simulated --repeats 3 --output validation/block4/baseline.json
python -X utf8 -m evaluation.run --mode simulated --repeats 3 --refinement --output validation/block4/refined.json
python -X utf8 -m evaluation.compare validation/block4/baseline.json validation/block4/refined.json --output validation/block4/comparison.json
```

La comparación exige fuentes/datos/tests con los mismos hashes y pares de
consulta/repetición. Registra dispersión de diferencias de tiempo y contadores;
las métricas musicales de simulación siguen nulas. Los dos comandos de medición
se ejecutan en procesos separados. Un experimento real necesita además controlar
carga, cachés, orden y disponibilidad del proveedor: el control local no establece
una mejora de latencia ni estima precio.

`judge_metrics.calibration_metrics(rows)` recibe pares con `human_status` y
`model_status` (`approved`, `rejected`, `abstained`; el modelo también `error`).
La falsa aprobación se calcula sobre humanos que rechazaron, y el falso rechazo
sobre humanos que aprobaron; abstención y error se informan aparte. Sin pares
anotados no se producen tasas. Conservar consulta, candidato, evidencia, snapshot
y anotador al preparar los pares; los veredictos simulados no sirven de referencia
humana. Ver `validation/block3/judge-calibration.json` para el estado pendiente.

La decisión de activar refinamiento rutinario requiere relevancia y cumplimiento
negativo comparables, calibración y un presupuesto de coste/latencia aceptado.
`compare.py` mantiene la decisión pendiente sin evidencia real y pide revisión
manual aun cuando haya nDCG real. No activa funciones automáticamente a partir
de una diferencia positiva ni reemplaza anotaciones desconocidas por ceros.

## Bloque IV: puntuación, filtros y perfil

El runner vigente usa `block4-post-P15`; los comandos anteriores escriben en
la carpeta del bloque IV para conservar los informes previos. No comparar scores
numéricos de fórmulas distintas: la API v2 usa rangos ponderados, con método
declarado y sin porcentajes. Las consultas de evaluación no se cambiaron para
favorecer ese cambio. Las pruebas HTTP añaden pesos/filtros y metadatos ausentes.

```powershell
python -X utf8 -m evaluation.profile --iterations 20 --output validation/block4/profile.json
python scripts/verify_block4.py
```

El perfil mide runtime del servicio, LangGraph, FAISS y cachés sobre el
catálogo actual; encoders, router y juez son controles. La reconstrucción de un
vector no mide inferencia musical. Hay 61 solicitudes temporizadas y cinco
adicionales para memoria. Las estadísticas distinguen primera ejecución y carga
secuencial/concurrente de una consulta repetida, sin medir transporte HTTP. Los
tiempos no incluyen trazado de memoria; esa fase
separada informa solo asignaciones Python. RSS/GPU y modelos/proveedor reales
permanecen desconocidos. Se conserva el grafo por solicitud hasta disponer de
evidencia para compartirlo sin compartir decisiones/cachés de forma indebida.
Ver `validation/block4/README.md` y su captura visual, rotulada como prueba local.
# Correcciones posteriores a la auditoría

El comportamiento y los informes actuales se documentan en
`../validation/pending_fixes/README.md`; las secciones de bloques previos describen
sus capturas históricas. El perfil actual añade RSS y contadores Torch de CUDA si
están disponibles, manteniendo separados controles y modelos reales.

`python -m evaluation.real_profile --output <ruta>` ejecuta el pipeline real con
muestreo de memoria; no descarga pesos por defecto. `--allow-model-downloads`
habilita esa opción explícitamente. Si falta el proveedor o hay errores de
inferencia, el informe no declara éxito real. `--qrels <ruta>` permite usar una
revisión humana vinculada al catálogo sin reemplazar el archivo original.

`python -m evaluation.annotations export --output <ruta-nueva>` prepara material
sin etiquetas. La importación usa `import --input <revisión> --output <ruta-nueva>`
y exige atribución humana, grados válidos y catálogo/consultas coincidentes.
No convierte desconocidos en ceros ni genera anotaciones automáticamente.

