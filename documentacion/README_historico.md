# SPELLFIND

## Módulos de medios y agentes — 20 de octubre de 2026

La búsqueda musical usa ahora `MusicModule` como frontera para planificar, buscar,
evaluar, refinar y consultar obras. `media_contracts.py` define capacidades,
condiciones, evidencias y veredictos comunes. La API musical anterior sigue
disponible y añade `media_type`, `common_plan`, `media_capabilities` y
`condition_verdicts`; `/media/capabilities` enumera únicamente música.
La coincidencia de audio continúa sin verificación acústica independiente.
El coordinador ya llama los puntos de extensión del módulo; la evaluación
individual completa y los catálogos de otros medios siguen pendientes.
Diseño y estado: [documentacion/agentes_multimedia/main.tex](documentacion/agentes_multimedia/main.tex).

## Continuación autónoma — 19 de octubre de 2026

Instrumentales con `instrumental:true` usan sólo el índice acústico: el mapa v2
mantiene posiciones independientes y sigue leyendo generaciones v1. La condición
instrumental queda declarada, sin atribuir una verificación del sonido.

Se añaden cuotas, mantenimiento con vista previa/cuarentena y copias verificadas
con restauración en una carpeta nueva. El entorno `.venv-ingestion` y sus locks
usan Transformers 4.57.6 compatible con Nomic; la inferencia real detectó el fallo
de la serie 5. Los pesos y código tienen revisiones fijadas.

La ingesta técnica real YouTube → FFmpeg → CLAP → catálogo pasó en un catálogo
aislado. También se verificaron Nomic y Whisper base en CPU; la calidad musical y
del reconocimiento de canciones requiere evaluación humana. Instalación y
mantenimiento: [INSTALL.md](INSTALL.md). Evidencia:
[validation/autonomous/README.md](validation/autonomous/README.md).

## Correcciones posteriores a la auditoría

Las restricciones explícitas ahora exigen correspondencia literal o equivalencias
revisadas y alcance completo de las negaciones. Las traducciones desconocidas
piden aclaración: esto pierde cobertura frente al modelo sin verificación, pero
evita promoverlas a evidencia. El juez exige JSON con citas, reintenta una vez y
se abstiene ante decisiones antiguas persistentes o citas no verificables.
El orquestador también impide seleccionar aprobaciones marcadas `legacy`.

La evaluación permite anotaciones humanas vinculadas al snapshot y dispone de
perfil real con RSS. Sin clave del proveedor ni anotaciones humanas no se declara
calidad musical ni rendimiento real. Evidencia y comandos:
[validation/pending_fixes/README.md](validation/pending_fixes/README.md).
Cambios, beneficios y costes:
[documentacion/correcciones_pendientes.tex](documentacion/correcciones_pendientes.tex).

## Bloque V: catálogo versionado e ingesta opcional

El catálogo se publica por generaciones validadas en `data/catalog/`, con puntero
e historial SQLite. Cada búsqueda fija una generación; nuevas consultas adoptan
las actualizaciones sin mezclar CSV, mapa e índices. `output/` se conserva como
fuente heredada. Una publicación fallida no reemplaza el catálogo utilizable.

La ingesta añade una cola persistente, progreso consultable, subtítulos oficiales
y fallback Whisper, embeddings Nomic/CLAP y reproducción ligada a la versión.
Está desactivada por defecto; la extensión opcional aún no tiene lock verificado.
Las pruebas locales usan adaptadores de control: no acreditan una descarga real,
calidad musical ni compatibilidad entre revisiones de los embeddings históricos.

Estado, beneficios y pendientes E17–E19: [documentacion/bloque5.tex](documentacion/bloque5.tex).
Comprobaciones reproducibles: [validation/block5/README.md](validation/block5/README.md).

Buscador musical híbrido de letras y audio. El servidor se ejecuta desde esta
carpeta con `python app.py`, en Windows x64/Python 3.12, después de instalar el lock de `requirements.txt` y configurar
`GROQ_API_KEY` en el entorno o `.env`.

## Contrato del plan

`search_contract.py` normaliza las cuatro claves `lyric_positive`,
`lyric_negative`, `clap_positive` y `clap_negative` a listas de textos.
Acepta cadenas individuales, vacíos/nulos y el formato lírico antiguo
`{text, weight}`; elimina espacios, vacíos y duplicados. Rechaza elementos
no textuales, campos ausentes y planes sin consultas positivas. El fallback
utiliza el mismo contrato.

## Validación, intención y exclusiones

Cada búsqueda tiene `request_id`, decisiones y exclusiones serializables.
El juez solo devuelve un veredicto; únicamente el grafo escribe exclusiones.
Las decisiones se vinculan a `plan_version`. Un rechazo histórico solo puede
volver al resultado seleccionado tras reevaluarse bajo un plan nuevo; no se
reintroduce como alternativa sin evaluar.

Estados: `approved` para aprobación solo lírica, `partially_validated` si la
letra se aprobó pero se pidió también audio, `retrieved` para recuperación solo
acústica sin validación independiente, `clarification_needed` para interpretación
incierta, y los estados anteriores de error, límite y ausencia de candidatos.
Cada resultado expone `validation.lyrics` y `validation.audio`; la interfaz los
muestra. Una canción instrumental no necesita letra ni una llamada al juez.
`judge_iterations` cuenta llamadas al juez y `candidate_iterations` selecciones.

El plan registra restricciones `user_explicit` con cita literal y negación
cuando corresponda, e inferencias `model_hypothesis`. Se conservan las primeras
y se pueden corregir las segundas. Activar/desactivar modalidades o reescribir
audio requiere evidencia textual; un fallo lírico no demuestra un fallo sonoro.
La cita se verifica, pero su equivalencia semántica con la traducción aún necesita
evaluación (E13). No se consideran garantizadas todas las restricciones posibles.

El fallback puede reutilizar un plan válido de la misma consulta o interpretar
frases acústicas de un vocabulario español/inglés acotado. Si no puede cubrir la
consulta o sus negaciones, devuelve `clarification_needed`, sin ejecutar búsquedas.
No copia el texto en ambas ramas con negativos vacíos ni promete traducción general.

## Pruebas

Desde la carpeta del proyecto:

```powershell
python -X utf8 -m unittest discover -s tests -v
```

Las pruebas usan LangGraph real (debe estar instalado). Reemplazan los modelos
y proveedores en los límites de importación y comprueban contratos, fallos,
exclusiones, solicitudes sucesivas y concurrentes, refinamiento por modalidad,
terminación, reutilización de rankings y recursos compartidos. No descargan
modelos ni certifican integración con las APIs reales.

## Flujo consolidado (puntos 5–8)

`workflow.py` es el único grafo soportado: planificar → buscar → evaluar →
refinar → buscar. La web y `python cli.py "consulta musical"` usan este motor.
`tarea_3.py` y `tarea_3_correcion.py` son entradas compatibles a la misma CLI;
los borradores originales están en `experiments/`, marcados como históricos.

Con `SPELLFIND_ENABLE_REFINEMENT=true`, después de un rechazo, `refiner.py` utiliza el feedback del juez y la consulta
original para proponer cambios de letras, audio o ambos. Valida las listas y protege las restricciones explícitas. El bloque II permite
corregir hipótesis y modalidades con evidencia, manteniendo intactas las ramas
no seleccionadas. El prompt preserva frases literales y evita
acumular positivos duplicados. La evaluación del juez sigue siendo solo lírica:
la reescritura acústica es una mejora de descriptores, no una verificación del
sonido de la canción. La respuesta lo declara como `validation_scope: lyrics`.
La existencia de refinamiento no demuestra todavía mayor relevancia musical;
hay comparación simulada pareada, pero faltan medidas reales y etiquetas humanas.

Los estados terminales incluyen recuperación acústica, validación parcial y
reformulación, además de aprobación lírica, límite, ausencia de candidatos y error.
Solo una aprobación explícita produce éxito; nunca se considera aprobado al
alcanzar tres intentos. No se ejecuta un refinamiento después del último intento.
Se devuelven `termination_reason`, `error_stage`, `initial_plan`, el plan final,
`refinement_log` y `workflow_trace`. La interfaz muestra una advertencia si no
se obtiene aprobación. Si falla la recuperación, no se muestra un ranking
antiguo como si correspondiera al nuevo plan.

Cada solicitud cachea los rankings por modalidad, consultas y top-K; cambiar
solo letras reutiliza audio y viceversa. No se repite la búsqueda para montar
la respuesta final. Las exclusiones se aplican después de recuperar, así que
no invalidan la caché. Nunca se comparten exclusiones entre solicitudes.

`resources.py` carga una sola vez índices, catálogo y mapa por proceso, con
inicialización protegida por bloqueo. Los workers conservan un único modelo
por modalidad y caches LRU de hasta 512 embeddings cada uno. Cada uso devuelve
una copia del embedding para evitar que la resta/normalización altere la caché.
Los workers ya no cargan copias del CSV. **Reiniciar el proceso después de
reconstruir el catálogo**: no hay actualización en caliente de estos recursos.

## Documentación y evaluación crítica

El documento principal está en [documentacion/main.tex](documentacion/main.tex)
e incluye el [registro de evolución](documentacion/evolucion.tex). El diagnóstico
de A/B/C se identifica como histórico. Cada cambio describe conducta anterior,
implementación, motivo, evidencia, contrapartidas y tareas pendientes.

El registro diferencia correcciones verificadas, reducción de trabajo por
conteo, capacidades nuevas y beneficios aún no medidos. Incluye 13 tareas extra
E01–E13, vinculadas a los antiguos puntos 9–14. El orden vigente de trabajo
es ahora P01–P20; las referencias anteriores quedan para trazabilidad.
Entre ellas: validación acústica e instrumentales, fallback fiel a restricciones,
recuperación parcial ante fallos, checkpoints/streaming perdidos respecto a C,
evaluación real del refinador e invalidación de caches.

Los cambios no son superiores en todas las dimensiones: la recuperación parcial
actual conserva disponibilidad con menor cobertura; refinar añade llamadas LLM;
el flujo consolidado no conserva los checkpoints de C. Las contrapartidas y
alternativas se registran en la documentación.

La suite actual aprobó 115 pruebas tanto en CP1252 como en UTF-8, incluyendo Flask y FAISS reales; no se midieron mejoras
de relevancia, latencia integral o coste con los modelos reales. Los perfiles se resolvieron en un entorno limpio y se verificaron imports reales;
la reinstalación del lock completada en un segundo entorno tiene evidencia separada en `validation/block1/`.

Cada bloque futuro debe actualizar el README y el registro de evolución con lo
implementado, sus pruebas y sus límites antes de declararse terminado. El formato
está en [documentacion/README.md](documentacion/README.md).

## Orden de trabajo pendiente

El [plan priorizado](documentacion/plan_mejoras.tex) unifica los pendientes en
cinco bloques de cuatro actividades. P01–P03 están implementadas y verificadas localmente; P04 tiene protocolo y
línea base simulada, con medidas musicales reales pendientes. P05–P09 y P11 están
implementados y verificados localmente. P10 tiene contrato/pruebas, con calibración
humana pendiente; P12 tiene comparación/política, con utilidad musical pendiente.
P13–P15 están implementados. P16 tiene instrumentación y perfil local con índices
reales/modelos de control; carga con modelos, RSS/GPU y optimización demostrada
siguen pendientes. P17–P20 siguen pendientes de implementación.

| Bloque | Actividades, en orden |
|---|---|
| I — Base verificable | P01 entorno limpio; P02 HTTP y arranque; P03 validación numérica; P04 línea base de evaluación. |
| II — Intención y estado | P05 propiedad del estado; P06 modalidades e instrumentales; P07 procedencia de restricciones; P08 fallback fiel. |
| III — Robustez y calidad | P09 recuperación parcial; P10 juez y evidencia; P11 presupuestos; P12 evaluación del refinamiento. |
| IV — Producto y rendimiento | P13 scores y pesos; P14 filtros; P15 evidencia en resultados; P16 optimización medida. |
| V — Catálogo e integración | P17 snapshots e invalidación; P18 decisión sobre checkpoints/streaming; P19 ingesta en segundo plano; P20 validación integrada. |

El siguiente bloque de implementación es **P17–P20**. P10/P12/P16 mantienen abiertas
sus verificaciones humanas y reales. P18 requiere
primero decidir si esas capacidades son necesarias; no se da por obligatoria
su implementación. Documentación y pruebas se actualizan en cada actividad.

El plan incluye dependencias, criterios de aceptación y una correspondencia
completa con 9–14 y E01–E16. E12 es seguimiento transversal, no otra tarea técnica.

## Bloque I: estado y evidencia

- **P01:** perfiles directos y locks para Windows x64/Python 3.12, entorno nuevo,
  `pip check`, imports reales y comprobación independiente de reinstalación.
  Instrucciones en [INSTALL.md](INSTALL.md); configuración en `.env.example`.
- **P02:** fábrica `create_app`, entrada `wsgi:app` para Waitress, debug desactivado
  por defecto, rutas estables y validación HTTP 400/413/500/503.
- **P03:** validación de vectores finitos, no degenerados y dimensiones correctas;
  índices unitarios y coherencia de los 258 IDs entre catálogo, mapa e índices.
- **P04:** 12 consultas fijas, 3 repeticiones, runners real/simulado, hashes y
  anotaciones humanas explícitas. Se completaron 36 corridas simuladas por modo,
  sin y con refinamiento, sin errores, duplicados ni reapariciones de excluidos.

La simulación no mide relevancia musical. Groq no tiene una clave configurada y
no se descargaron pesos: la evaluación real queda pendiente y sus métricas no se
inventan. Las pruebas importaron las bibliotecas reales sin cargar modelos.

Evidencia: [validation/block1/README.md](validation/block1/README.md).
Registro crítico: [documentacion/bloque1.tex](documentacion/bloque1.tex).
Protocolo: [evaluation/README.md](evaluation/README.md).

## Bloque II: evidencia y contrapartidas

P05–P08 están implementados con 72 pruebas aprobadas. El registro detallado está
en [documentacion/bloque2.tex](documentacion/bloque2.tex) y la evidencia en
[validation/block2/README.md](validation/block2/README.md). Las capturas del bloque
I permanecen intactas; las actuales usan `block2-pre-P09`.

En cada serie de 36 simulaciones hubo 12 aprobaciones líricas, 6 validaciones
parciales, 12 recuperaciones acústicas y 6 solicitudes de reformulación, sin errores.
Dos consultas del fixture antiguo, repetidas tres veces, carecen de evidencia de
sus negaciones y ahora se abstienen. Esto reduce cobertura automática; no demuestra
mayor relevancia. La amplitud del fallback y la fidelidad de las citas quedan
pendientes de evaluación, junto con la validación acústica independiente.

No se añadieron dependencias ni se llamaron modelos externos. Ese registro es
histórico: P09/P11 se implementaron en la continuación que se detalla abajo.

La revisión posterior confirmó tres defectos antes de la etapa III:
E14, reordenar consultas libera exclusiones y repite candidatos; E15, la ruta
de aclaración omite comprobar metadatos JSON finitos; E16, mensajes Unicode
interrumpen la inicialización con salida CP1252 (72 pruebas pasan con UTF-8;
sin esa opción hubo dos errores). Reproducción, evidencia y
criterios de corrección en [validation/revision_block2/README.md](validation/revision_block2/README.md).
Los defectos se corrigieron en la continuación: E14 se asignó al bloque II/P07,
E15 al bloque II/P05/P08 y E16 al bloque I/P01/P02. Se mantienen sus IDs e informes
históricos. Las regresiones actuales están en `tests/test_block3.py`.

## Bloque III: fallos, evidencia, presupuestos y refinamiento

- P09 conserva la rama disponible, registra fallos y pesos efectivos; continúa
  con el plan válido si falla el refinador y restaura plan/ranking juntos si falla
  toda la nueva recuperación. La degradación no aprueba sonido.
- P10 solicita decisiones JSON con citas verificables y abstención. Incluye hasta
  12000 caracteres de letra; para letras mayores selecciona inicio, centro y final,
  sin aprobar con cobertura incompleta. Un formato inválido permite un reintento;
  errores del proveedor no se reintentan por el juez. Los formatos antiguos quedan
  etiquetados `legacy`, sin evidencia verificada inventada.
- P11 separa hasta 20 visitas, 3 evaluaciones (hasta 6 intentos del modelo), 2
  refinamientos y una ampliación top-10 → top-20 por rama. Metadatos ausentes no
  consumen evaluaciones; el agotamiento se refiere a la ventana recuperada.
- P12 deja el refinamiento **desactivado por defecto**. La opción experimental
  `SPELLFIND_ENABLE_REFINEMENT=true` permite compararlo. Dos series de 36
  simulaciones tuvieron los mismos estados; el refinamiento añadió 18 invocaciones
  al refinador y 18 búsquedas líricas. No se midió relevancia musical ni precio.

Evidencia: [validation/block3/README.md](validation/block3/README.md).
Justificación y contrapartidas: [documentacion/bloque3.tex](documentacion/bloque3.tex).
La evaluación real quedó bloqueada por falta de `GROQ_API_KEY`; las etiquetas
humanas siguen pendientes. No se declara cerrada la calibración P10 ni la utilidad
real de P12. Tampoco se resuelve automáticamente la fidelidad semántica E13.

## Bloque IV: controles, resultados explicables y perfil local

- **P13:** puntuación por rangos ponderados de 0 a 1, sin porcentaje de acierto.
  `score_components` muestra posición, score del worker, peso y aporte de cada
  modalidad. La API v2 elimina `score_pct`; un score 1 no demuestra relevancia.
  La fusión puede cambiar el orden anterior y descarta magnitudes de los workers;
  su calidad musical requiere evaluación real.
- **P14:** filtros exactos por artista, género (`tag`) y años inclusivos,
  combinados con AND antes de la fusión y del juez. NFC, mayúsculas y espacios
  exteriores se normalizan; se conservan acentos y nombres completos. Con filtros
  se recupera todo el catálogo antes de seleccionar la ventana elegible. Sin
  coincidencias de metadatos no se llama a modelos. `/catalog/options` ofrece
  sugerencias sin letras. La estrategia completa debe revisarse al crecer el catálogo.
- **P15:** tarjetas con contribuciones, fragmentos, procedencia y alcance del
  juicio. Las citas del juez se comprueban en la letra; los demás extractos se
  marcan orientativos. Las alternativas y el audio conservan su falta de evaluación.
  La interfaz incorpora controles de pesos y filtros, y se verificó en navegador
  con un fixture visiblemente marcado como control.
- **P16:** tiempos de grafo/nodos/presentación y perfiles de cachés, codificación,
  búsqueda e interacción con bloqueos. El perfil local usa FAISS real, encoders
  de control y solicitudes secuenciales/concurrentes. Las mediciones reales de
  modelos, proveedor y memoria completa siguen pendientes. Se mantienen grafos
  por solicitud y cachés acotadas; no se atribuye una optimización no demostrada.

Opciones HTTP y CLI: [INSTALL.md](INSTALL.md). Evidencia y límites:
[validation/block4/README.md](validation/block4/README.md). Justificación técnica:
[documentacion/bloque4.tex](documentacion/bloque4.tex). No hay dependencias nuevas;
los informes históricos se conservan y las métricas musicales sin etiquetas
humanas siguen nulas.
