###############################################################################################################
# Tarea 3: Recomendación Musical con Feedback Iterativo (Solo Semántica)
###############################################################################################################

import operator
import json
import csv
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from semantic_search import _load_resources, _multi_query_search_raw, _reciprocal_rank_fusion

##################################################################################################
#
# CARGA GLOBAL DE METADATOS (Requerido para el Juez)
#
##################################################################################################

def _load_song_metadata(csv_path: str = "./output/datos_alineados.csv") -> Dict[int, Dict[str, Any]]:
    """Carga datos_alineados.csv en memoria para inyectar letras al Evaluador."""
    metadata: Dict[int, Dict[str, Any]] = {}
    path = Path(csv_path)
    if not path.exists():
        print(f"[METADATA] Advertencia: {csv_path} no encontrado.")
        return metadata
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get("song_id") or row.get("id") or ""
            try:
                sid = int(raw_id)
            except ValueError:
                continue
            metadata[sid] = dict(row)
    print(f"[METADATA] {len(metadata)} canciones cargadas en memoria global.")
    return metadata

# Cargar metadatos al iniciar el módulo
GLOBAL_METADATA = _load_song_metadata()


##################################################################################################
#
# ESTADO GLOBAL
#
##################################################################################################

class EstadoRecomendacionMusical(TypedDict):
    # ── Campos renombrados para coincidir con el patrón del notebook ──
    texto_fuente: str                                                    # antes: query
    esquema_json: Dict[str, Any]                                         # nuevo
    mensajes_investigador: Annotated[List[AnyMessage], operator.add]     # antes: messages
    contexto_investigado: str                                            # nuevo
    diapositivas_generadas: List[Dict[str, Any]]                         # antes: candidates
    indice_actual: int                                                   # nuevo
    estado_compilacion: str                                              # antes: estado_recomendacion

    # ── Canal aislado para el ciclo Ejecutor ↔ ToolNode ──
    # Sin reducer: se sobreescribe en cada ronda para que el LLM
    # no reutilice tool_calls de iteraciones anteriores.
    executor_messages: List[AnyMessage]

    excluded_ids: List[int]                          # IDs ya evaluados, para excluirlos en rondas siguientes
    semantic_feedback: Annotated[List[Dict[str, Any]], operator.add]
    sem_vector: Annotated[List[str], operator.add]
    sem_vector_neg: Annotated[List[str], operator.add]
    current_step: str
    veredicto: Annotated[List[Dict[str, Any]], operator.add]


##################################################################################################
#
# HERRAMIENTAS (Solo Búsqueda Semántica)
#
##################################################################################################

@tool
def busqueda_semantica(
    positive: List[str],
    negative: List[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Busca canciones por similaridad semántica de letras."""
    if not positive:
        return []

    resources = _load_resources()
    positive = [q.strip() for q in positive if len(q.strip()) > 5]
    negative = [n.strip() for n in negative if len(n.strip()) > 5]

    if not positive:
        return []

    print(f"  [TOOL busqueda_semantica] {len(positive)} positivas, {len(negative)} negativas, top_k={top_k}")

    all_raw = _multi_query_search_raw(
        lyric_queries=positive,
        lyric_negatives=negative,
        top_k=top_k * 3,
        resources=resources,
    )
    fused = _reciprocal_rank_fusion(all_raw)
    print(f"  [TOOL busqueda_semantica] → {len(fused)} candidatos tras RRF")

    result = [{"song_id": int(sid), "score": float(sc)} for sid, sc in fused[:top_k]]
    return result

herramientas = [busqueda_semantica]

# Wrapper sobre ToolNode para que opere sobre executor_messages en vez de messages.
# ToolNode por defecto lee el último AIMessage de state["messages"] y escribe
# el ToolMessage en state["messages"]. Aquí lo redirigimos al canal aislado.
_tool_node = ToolNode(tools=herramientas)

def nodo_ejecutor_herramientas(state: EstadoRecomendacionMusical) -> dict:
    """Ejecuta la tool_call usando executor_messages como canal aislado."""
    exec_msgs = state.get("executor_messages", [])
    # ToolNode necesita que el último AIMessage con tool_calls esté en "messages"
    # Le pasamos un estado temporal con sólo ese mensaje
    last_ai = exec_msgs[-1] if exec_msgs else None
    if last_ai is None or not (isinstance(last_ai, AIMessage) and getattr(last_ai, "tool_calls", None)):
        print("  [TOOL NODE] No hay tool_call pendiente.")
        return {}
    # Invocamos ToolNode con un estado mínimo donde messages = [last_ai]
    tool_result = _tool_node.invoke({"messages": [last_ai]})
    tool_msgs = tool_result.get("messages", [])
    # Añadimos el ToolMessage al canal aislado (sobreescritura completa)
    return {
        "executor_messages": exec_msgs + tool_msgs,
        "mensajes_investigador": tool_msgs,  # log general
    }


def _sanitize_messages_for_groq(messages: List[AnyMessage]) -> List[AnyMessage]:
    sanitized = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            if not content or content.strip() in ("", "[]", "null"):
                content = "[]"
            sanitized.append(ToolMessage(
                content=content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
            ))
        else:
            sanitized.append(msg)
    return sanitized


##################################################################################################
#
# PROMPTS DE SISTEMA
#
##################################################################################################

PROMPT_PLANIFICADOR = SystemMessage(content="""
You are the Search Planner of SPELLFIND.
Your job: convert a user query into semantic search vectors for lyric similarity search.

STEP 1 — CLASSIFY THE QUERY (do this mentally before writing anything):
Decide which type the query is:

  TYPE A — LITERAL: The query uses concrete, specific words that likely appear directly
    in song lyrics (e.g. "little brain", "leaving me behind", "heart is breaking").
    These are NOT metaphors — they are the actual subject matter of the song.

  TYPE B — ABSTRACT/EMOTIONAL: The query describes a feeling, situation, or theme
    without naming specific lyric-level words (e.g. "feeling lost after a breakup",
    "the sadness of moving to a new city").

STEP 2 — GENERATE VECTORS based on the type:

  If TYPE A (literal query):
    - Vector 1: Use the EXACT key phrases from the query in a short first-person sentence
      that a song lyric could actually say. 15-25 words. Stay close to the original wording.
      NEVER rephrase concrete images (e.g. do not map "little brain" to "cognitive decline").
    - Vector 2: Same imagery, slightly different sentence structure or emotional angle.
    - Vector 3 (optional): The emotional tone behind the literal image, still grounded.

  If TYPE B (abstract/emotional query):
    - Write 2-3 first-person narrative situations (25-40 words each) capturing the emotion
      and situation. Be specific about who is speaking, what happened, how they feel.

RULES FOR ALL VECTORS:
  - ALL vectors in English.
  - Before writing, clean the query: ignore typos, repeated words, filler sounds.
  - For TYPE A: never invent details not present in the query.
  - Negative vectors: only when the query explicitly excludes something.

Reply ONLY with a JSON:
{
  "query_type":      "A" | "B",
  "sem_vector":      ["vector 1", "vector 2"],
  "sem_vector_neg":  [],
  "current_step":    "planned"
}
""")

PROMPT_EJECUTOR = SystemMessage(content="""
You are the Plan Executor of SPELLFIND.

You have access to one tool:
- busqueda_semantica(positive, negative, top_k): searches songs by lyrics.

Mandatory execution order:
1. Call busqueda_semantica(positive=sem_vector, negative=sem_vector_neg).
2. Stop without any additional text.

Rules:
- Use exactly the vectors provided. Do not invent phrases.
- Emit tool_calls only.
""")

PROMPT_EVALUADOR = SystemMessage(content="""
You are the Judge of SPELLFIND.

You receive the original user query and the TOP-1 candidate song WITH ITS LYRICS.

EVALUATION RULES:

1. First, classify the query:
   - TYPE A (literal): the query contains specific concrete words/images that could
     appear literally in song lyrics (e.g. "little brain", "leaving me behind").
   - TYPE B (abstract): the query describes a theme or emotion without specific lyric words.

2. For TYPE A queries: the song passes ONLY if the lyrics contain the specific literal
   image or phrase (or a very close synonym). A song about "mental deterioration" does NOT
   pass a query about "little brain". Stay literal — match the exact imagery.

3. For TYPE B queries: evaluate whether the song's emotional narrative and theme match
   the described situation.

4. If "insuficiente", your feedback MUST:
   - State which specific words/images from the query are missing in the lyrics.
   - For TYPE A: explicitly forbid the refiner from abstracting (e.g. "do NOT use
     'cognitive decline' — the query says 'little brain', search for that literally").
   - For TYPE B: describe what emotional direction the refiner should take instead.

Reply ONLY with a JSON:
{
  "resultado": "suficiente" | "insuficiente",
  "justificacion": "Why the song passes or fails, referencing specific lyrics and query words.",
  "semantic_feedback": {
    "query_type": "A" | "B",
    "problema": "Which specific query elements are missing from the lyrics.",
    "sugerencia": "Precise instruction for the refiner, including what to avoid."
  }
}
""")

PROMPT_REFINADOR_SEMANTICO = SystemMessage(content="""
You are the Semantic Refiner of SPELLFIND.

You receive: the original query, the Judge's semantic_feedback (with query_type), and the
current vectors that failed.

STEP 1 — Read query_type from the feedback:

  If query_type = "A" (literal query):
    - The judge identified specific words/images missing from the lyrics.
    - Generate vectors that use those EXACT words or their closest synonyms.
    - Keep vectors SHORT (15-25 words). Do not over-narrate.
    - Do NOT abstract: if the query says "little brain", write "little brain", not
      "diminished cognitive function" or "mental deterioration".
    - Bad example: "I am struggling with cognitive decline and losing mental clarity"
    - Good example: "My little brain is leaving me behind and I am begging it to stay"

  If query_type = "B" (abstract/emotional query):
    - Generate 2-3 first-person narrative situations (25-40 words) that fix the
      emotional/thematic gap identified by the judge.
    - Be specific about situation, feeling, and what changed.

STEP 2 — Extend, do not replace the existing vectors (keep the ones that were close).

STEP 3 — Add negative vectors ONLY for concepts the judge explicitly said to exclude.

ALL vectors in English.

Reply ONLY with a JSON:
{
  "sem_vector":     ["vector 1", "vector 2"],
  "sem_vector_neg": [],
  "current_step":   "refined_semantic"
}""")

##################################################################################################
#
# INSTANCIAS LLM
#
##################################################################################################

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_ejecutor = llm.bind_tools(herramientas)

def _call_llm_json(system_prompt: SystemMessage, state_payload: dict) -> dict:
    user_msg = HumanMessage(content=json.dumps(state_payload, ensure_ascii=False))
    response = llm.invoke([system_prompt, user_msg])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


##################################################################################################
#
# NODOS OPERATIVOS
#
##################################################################################################

def nodo_planificador(state: EstadoRecomendacionMusical) -> dict:
    print("\n[PLANIFICADOR] Generando vectores de búsqueda semántica...")
    payload = {
        "original_query_verbatim": state["texto_fuente"],
    }
    resultado = _call_llm_json(PROMPT_PLANIFICADOR, payload)
    resultado["estado_compilacion"] = "buscando"
    resultado["executor_messages"] = []   # canal limpio para primera ronda
    resultado["excluded_ids"] = []        # sin exclusiones al inicio
    print(f"  query_type     : {resultado.get('query_type', '?')}")
    print(f"  sem_vector     : {resultado.get('sem_vector', [])}")
    print(f"  sem_vector_neg : {resultado.get('sem_vector_neg', [])}")
    return resultado


def nodo_ejecutor(state: EstadoRecomendacionMusical) -> dict:
    # executor_messages es el canal aislado de esta ronda (se sobreescribe, no acumula)
    exec_msgs = state.get("executor_messages", [])
    excluded_ids = state.get("excluded_ids", [])

    # ── Detectar si ya llegó el ToolMessage de esta ronda ──────────────────
    for msg in reversed(exec_msgs):
        if isinstance(msg, ToolMessage) and msg.name == "busqueda_semantica":
            print("[EJECUTOR] Resultados semánticos recibidos. Cerrando ronda de búsqueda.")
            try:
                raw = json.loads(msg.content) if msg.content else []
            except Exception:
                raw = []

            # Filtrar IDs ya evaluados en rondas anteriores
            candidates = []
            for item in raw:
                if isinstance(item, dict) and "song_id" in item:
                    sid = int(item["song_id"])
                    if sid not in excluded_ids:
                        candidates.append({"song_id": sid, "score": float(item.get("score", 0.0))})

            print(f"  [EJECUTOR] {len(candidates)} candidatos en Estado.")
            return {
                "diapositivas_generadas": candidates,
                "estado_compilacion": "finalizada",
                "current_step": "ejecutado",
            }

    # ── Primera llamada de la ronda: pedir tool_call al LLM ────────────────
    iteracion_msg = "primera llamada" if not exec_msgs else f"{len(exec_msgs)} mensajes en canal"
    print(f"[EJECUTOR] Invocando LLM con tools ({iteracion_msg})...")

    context = {
        "sem_vector":     state.get("sem_vector", []),
        "sem_vector_neg": state.get("sem_vector_neg", []),
    }
    user_msg = HumanMessage(content=json.dumps(context, ensure_ascii=False))

    safe_messages = _sanitize_messages_for_groq(exec_msgs)
    all_messages = [PROMPT_EJECUTOR] + safe_messages + [user_msg]

    response = llm_ejecutor.invoke(all_messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"  [EJECUTOR] tool_calls emitidos: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print("  [EJECUTOR] Sin tool_calls — forzando cierre.")

    # Acumular dentro del canal aislado de esta ronda
    return {
        "executor_messages": exec_msgs + [response],
        "mensajes_investigador": [response],  # log general (operator.add)
    }


def nodo_evaluador_recomendaciones(state: EstadoRecomendacionMusical) -> dict:
    n_iter = len(state.get("veredicto", [])) + 1
    candidates = state.get("diapositivas_generadas", [])
    top1 = candidates[0] if candidates else {}

    # INYECCIÓN DE METADATOS: El juez necesita leer la letra para evaluar
    top1_enrich = {}
    if top1 and top1.get("song_id") in GLOBAL_METADATA:
        meta = GLOBAL_METADATA[top1["song_id"]]
        top1_enrich = {
            "song_id": top1["song_id"],
            "title": meta.get("title", "?"),
            "artist": meta.get("artist", "?"),
            "lyrics": meta.get("lyrics", meta.get("lyric", "No lyrics available"))
        }

    print(f"\n[EVALUADOR] Iteración {n_iter} — evaluando TOP-1: {top1_enrich.get('title', 'Unknown')}")

    payload = {
        "query": state["texto_fuente"],
        "top1_candidate": top1_enrich,
        "iteracion": n_iter,
    }
    resultado = _call_llm_json(PROMPT_EVALUADOR, payload)

    print(f"  resultado              : {resultado.get('resultado')}")
    print(f"  justificacion          : {resultado.get('justificacion')}")

    # Agregar el top1 evaluado al pool de excluidos para que no vuelva a aparecer
    excluded_ids = list(state.get("excluded_ids", []))
    if top1.get("song_id") is not None:
        excluded_ids.append(top1["song_id"])
        print(f"  [EVALUADOR] song_id {top1['song_id']} agregado a excluded_ids ({len(excluded_ids)} total)")

    updates: dict = {
        "veredicto": [
            {
                "resultado": resultado.get("resultado"),
                "justificacion": resultado.get("justificacion"),
            }
        ],
        "current_step": "evaluado",
        "excluded_ids": excluded_ids,
    }
    if resultado.get("semantic_feedback"):
        updates["semantic_feedback"] = [resultado["semantic_feedback"]]

    return updates


def nodo_refinador_semantico(state: EstadoRecomendacionMusical) -> dict:
    print("\n[REFINADOR SEMÁNTICO] Ajustando vectores semánticos...")
    payload = {
        "original_query_verbatim": state["texto_fuente"],
        "judge_feedback": state.get("semantic_feedback", [])[-1:],
        "vectors_that_failed": state.get("sem_vector", []),
        "negative_vectors_current": state.get("sem_vector_neg", []),
    }
    resultado = _call_llm_json(PROMPT_REFINADOR_SEMANTICO, payload)
    resultado["estado_compilacion"] = "buscando"
    resultado["executor_messages"] = []   # resetear canal aislado para la nueva ronda
    print(f"  nuevos sem_vector    : {resultado.get('sem_vector', [])}")
    print(f"  nuevos sem_vector_neg: {resultado.get('sem_vector_neg', [])}")
    return resultado


##################################################################################################
#
# ROUTERS
#
##################################################################################################

MAX_ITERACIONES = 3

def router_juez(state: EstadoRecomendacionMusical) -> str:
    veredicto = state.get("veredicto", [])
    if len(veredicto) >= MAX_ITERACIONES:
        print(f"[ROUTER JUEZ] Límite de {MAX_ITERACIONES} iteraciones alcanzado → forzar fin")
        return "aceptado"

    ultimo = veredicto[-1] if veredicto else {}
    if ultimo.get("resultado", "insuficiente") == "suficiente":
        print("[ROUTER JUEZ] Veredicto suficiente → END")
        return "aceptado"

    print("[ROUTER JUEZ] Refinamiento requerido → Refinador Semántico")
    return "refina_semantico"


def router_ejecutor(state: EstadoRecomendacionMusical) -> str:
    if state.get("estado_compilacion") == "finalizada":
        return "juzgar"

    exec_msgs = state.get("executor_messages", [])
    if exec_msgs:
        last = exec_msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "usar_herramienta"

    return "juzgar"


##################################################################################################
#
# ENSAMBLADO DEL GRAFO
#
##################################################################################################

grafo_recomendacion = StateGraph(EstadoRecomendacionMusical)

grafo_recomendacion.add_node("Planificador",             nodo_planificador)
grafo_recomendacion.add_node("Ejecutor",                 nodo_ejecutor)
grafo_recomendacion.add_node("Ejecutor de Herramientas", nodo_ejecutor_herramientas)
grafo_recomendacion.add_node("Evaluador",                nodo_evaluador_recomendaciones)
grafo_recomendacion.add_node("Refinador Semántico",      nodo_refinador_semantico)

grafo_recomendacion.add_edge(START, "Planificador")
grafo_recomendacion.add_edge("Planificador", "Ejecutor")

grafo_recomendacion.add_conditional_edges(
    "Ejecutor",
    router_ejecutor,
    {
        "usar_herramienta": "Ejecutor de Herramientas",
        "juzgar":           "Evaluador",
    },
)

grafo_recomendacion.add_edge("Ejecutor de Herramientas", "Ejecutor")

grafo_recomendacion.add_conditional_edges(
    "Evaluador",
    router_juez,
    {
        "refina_semantico": "Refinador Semántico",
        "aceptado":         END,
    },
)

grafo_recomendacion.add_edge("Refinador Semántico", "Ejecutor")


##################################################################################################
#
# COMPILACIÓN Y VISUALIZACIÓN
#
##################################################################################################

# Compilar con checkpointer para persistencia y streaming
banco_memoria = MemorySaver()
app = grafo_recomendacion.compile(checkpointer=banco_memoria)


##################################################################################################
#
# HELPERS DE DISPLAY
#
##################################################################################################

def _display_top1(candidate: Dict[str, Any], metadata: Dict[int, Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("  TOP-1 RECOMENDACION")
    print("=" * 60)

    song_id = candidate.get("song_id")
    score = candidate.get("score", "?")

    print(f"  FAISS id : {song_id}")
    print(f"  Score    : {score:.4f}" if isinstance(score, float) else f"  Score    : {score}")

    if song_id is not None and song_id in metadata:
        meta = metadata[song_id]
        title  = meta.get("title")  or meta.get("song_name") or meta.get("name")  or "?"
        artist = meta.get("artist") or meta.get("artist_name") or meta.get("author") or "?"
        print(f"  Título   : {title}")
        print(f"  Artista  : {artist}")

        skip = {"title", "song_name", "name", "artist", "artist_name", "author", "lyrics", "lyric"}
        for k, v in meta.items():
            if k not in skip and v:
                print(f"  {k:<10}: {v}")
    else:
        print(f"  (song_id {song_id} no encontrado en datos_alineados.csv)")

    print("=" * 60)


##################################################################################################
#
# PUNTO DE ENTRADA
#
##################################################################################################

if __name__ == "__main__":
    print("=" * 60)
    print("  SPELLFIND — Recomendación Musical (Semantic Mode)")
    print("=" * 60)

    texto_plano = "song about a brain thats very little and is leaving me behind  little and is leaving me behind  little and is leaving me behind  little and is leaving me but i want it back please"

    input_inicial = {
        "texto_fuente": texto_plano,
        "esquema_json": {},
        "mensajes_investigador": [],
        "contexto_investigado": "",
        "diapositivas_generadas": [],
        "indice_actual": 0,
        "estado_compilacion": "iniciada",
        # ── canales internos ──
        "executor_messages": [],
        "excluded_ids": [],
        "semantic_feedback": [],
        "sem_vector": [],
        "sem_vector_neg": [],
        "current_step": "inicio",
        "veredicto": [],
    }

    print(f"\nQuery: {input_inicial['texto_fuente']}\n")

    # Identificador de esta sesión — cambiar para aislar corridas
    config = {"configurable": {"thread_id": "auditoria_SPELLFIND"}}

    try:
        print("🚀 Ejecutando con persistencia activa...")
        for update in app.stream(input_inicial, config=config, stream_mode="updates"):
            print(update)

    except Exception as e:
        print(f"\n💥 El grafo colapsó: {e}")
        print("🕵️  Recuperando estado del checkpointer...")

        estado_forense = app.get_state(config).values
        print("\n--- ¡Estado recuperado con éxito! ---")

        candidates = estado_forense.get("diapositivas_generadas", [])
        if candidates:
            _display_top1(candidates[0], GLOBAL_METADATA)
        else:
            print("[!] No se encontraron candidatos en el estado recuperado.")

        print(f"\nRondas de evaluación   : {len(estado_forense.get('veredicto', []))}")
        for v in estado_forense.get("veredicto", []):
            print(f"  [{v['resultado']}] {v['justificacion']}")

    else:
        # Flujo normal sin excepción: leer el estado final del checkpointer
        resultado = app.get_state(config).values
        candidates = resultado.get("diapositivas_generadas", [])

        if candidates:
            _display_top1(candidates[0], GLOBAL_METADATA)
        else:
            print("\n[!] No se encontraron candidatos.")

        print(f"\nCandidatos recuperados : {len(candidates)}")
        print(f"Rondas de evaluación   : {len(resultado.get('veredicto', []))}")
        for v in resultado.get("veredicto", []):
            print(f"  [{v['resultado']}] {v['justificacion']}")
    
    from IPython.display import Image, display
           
    # ── Visualización del grafo ──────────────────────────────────
    print("\n[GRAFO] Generando diagrama...")
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        output_path = "grafo_SPELLFIND.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"[GRAFO] Diagrama guardado en: {output_path}")
        # Abrir automáticamente con el visor de imágenes del sistema
        import subprocess, sys
        if sys.platform == "win32":
            subprocess.Popen(["start", output_path], shell=True)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_path])
        else:
            subprocess.Popen(["xdg-open", output_path])
    except Exception as e:
        # Fallback: imprimir el código Mermaid para pegar en mermaid.live
        print(f"[GRAFO] PNG falló ({e}), usando Mermaid text:")
        print(app.get_graph().draw_mermaid())