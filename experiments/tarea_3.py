###############################################################################################################
# Tarea 3: Recomendación Musical con Feedback Iterativo
###############################################################################################################

import operator
import json
from typing import TypedDict, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, START, END  
from langgraph.prebuilt import ToolNode, MessageNode, LLMNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage 
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool


from semantic_search import _load_resources, _multi_query_search_raw, _reciprocal_rank_fusion                                 # Importamos la herramienta de búsqueda semántica


# Esta version mantendra siempre dos vectores por dimensión. Uno para la parte positiva y otro para la parte negativa, y el nodo planificador decidirá cómo combinarlos (e.g., sumarlos, restarlos, o usar solo uno de ellos) según la query del usuario y el feedback del evaluador.

class EstadoRecomendacionMusical(TypedDict):
    query: str                                                               # Query original del usuario
    candidates: List[Dict[str, Any, str]]                                    # Lista de canciones candidatas ID, puntaje, letra.
    semantic_feedback: Annotated[List[Dict[str, Any]], operator.add]                   # Feedback específico sobre la dimensión semántica (e.g., relevancia de las letras)
    auditory_feedback: Annotated[List[Dict[str, Any]], operator.add]                    # Feedback específico sobre la dimensión auditiva (e.g., relevancia de las características sonoras)
    sem_vector: Annotated[List[str], operator.add]                         # Vector semántico acumulativo (si se decide usarlo)
    clap_vector: Annotated[List[str], operator.add]                        # Vector de características auditivas acumulativo (si se decide usarlo)
    current_step: str                                                        # Paso actual en la máquina de estados (e.g., "búsqueda", "evaluación", "feedback") 
    # Veredicto del juez . Si es insuficiente o necesita refinamiento.
    veredicto: Annotated[List[Dict[str, Any]], operator.add]                   # Veredicto del router juez basado en el feedback recibido (e.g., "refinar_semantico", "refinar_auditivo", "terminar")
    # Estado de la recomendacion musical (finalizada, ejecucion semantica, ejecucion auditiva, unificacion de resultados)
    estado_recomendacion: str
    
    
####################################################################################
#
# HERRAMIENTAS DE PROCESAMIENTO DE TEXTO Y BÚSQUEDA SEMÁNTICA
#
###############################################################################################


@tool
def busqueda_semantica(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    if not positive:
        return []

    resources = _load_resources()

    # Filtrar strings vacíos
    positive = [q.strip() for q in positive if len(q.strip()) > 5]
    negative = [n.strip() for n in negative if len(n.strip()) > 5]

    if not positive:
        return []

    print(f"[SEMANTIC] Buscando: {len(positive)} queries positivas, "
          f"{len(negative)} negativas, top_k={top_k}")

    all_raw = _multi_query_search_raw(
        lyric_queries=positive,
        lyric_negatives=negative,
        top_k=top_k * 3,   # margen amplio antes del RRF
        resources=resources,
    )

    fused = _reciprocal_rank_fusion(all_raw)
    print(f"[SEMANTIC] → {len(fused)} candidatos tras RRF interno")

    return fused[:top_k]


##################################################################################################
#
#  Nodos operatuvos de la máquina de estados (agentes especializados, evaluadores, herramientas)
#
##################################################################################################

# El Nodo Herramienta es preconstruido por LangGraph y se encarga de ejecutar la función de búsqueda semántica cuando el flujo lo requiere.
# El LLM se activa únicamente para procesar la query del usuario y generar el plan de
herramientas = [busqueda_semantica]
nodo_ejecutor_herramientas = ToolNode(herramientas=herramientas, 
                                      name="Ejecutor de Herramientas", 
                                      description="Ejecuta herramientas especializadas como la búsqueda semántica de letras para encontrar canciones relevantes basadas en la query del usuario.")

# Nodo que juzga las recomendaciones musicales
nodo_evaluador_recomendaciones = LLMNode(model="gpt-4", 
                                         name="Evaluador de Recomendaciones", 
                                         description="Evalúa la calidad de las recomendaciones musicales basadas en la query original, proporcionando una puntuación y justificación para cada canción recomendada.")

# EL sistema usa vectores representativos para hacer una busqueda vectorial de canciones similares a la query del usuario, y el evaluador juzga la relevancia de esas recomendaciones. 
# Si el feedback es negativo, el nodo planificador genera un nuevo plan para ajustar la búsqueda (e.g., cambiar términos de búsqueda, ajustar pesos, etc.) y vuelve a ejecutar la herramienta de búsqueda semántica con el nuevo plan.

# Nodo que planifica el primer paso: generar el plan de búsqueda basado en la query del usuario
# La busuqeda se basa en interpretar la query del usuario y generar un plan de busqueda (Dimension semantica, clap o ambos) que luego se ejecutará con la herramienta de búsqueda semántica.
# El planificador decide los vectores semanticos y/o clap a usar, y  el nodo ejecutor decide que herramienta usar para cada parte del plan (e.g., busqueda semantica para la parte de vectores, y una herramienta de filtrado local para la parte de clap)
# debe modificar los estados de vector semántico y/ o auditivo en el Estado global, que luego serán usados por el nodo ejecutor para realizar la búsqueda y por el nodo evaluador para juzgar las recomendaciones.
nodo_planificador = LLMNode(model="gpt-4",
                            name="Planificador de Búsqueda", 
                            description="Genera un plan de búsqueda detallado basado en la query del usuario, decidiendo qué dimensiones semánticas y auditivas explorar.")    

# Nodo ejecutor
nodo_ejecutor = LLMNode(model="gpt-4",
                        name="Ejecutor de Plan",
                        description="Ejecuta el plan generado por el planificador, actualizando el Estado global con los resultados de la búsqueda y preparando la información para el evaluador.")

# nodo refinador semantico
nodo_refinador_semantico = LLMNode(model="gpt-4",
                                  name="Refinador Semántico",
                                    description="Refina los vectores semánticos basados en el feedback del evaluador, ajustando la búsqueda para mejorar la relevancia de las recomendaciones.")

# nodo refinador auditivo
nodo_refinador_auditivo = LLMNode(model="gpt-4",
                                    name="Refinador Auditivo",
                                        description="Refina los vectores de características auditivas basados en el feedback del evaluador, ajustando la búsqueda para mejorar la relevancia de las recomendaciones.")


##################################################################################################
#
#  routers: Aristas condicionales, routers de flujo del grafo
#
##################################################################################################

# El router juez decide, basado en el feedback del evaluador, si el plan de búsqueda fue exitoso o si necesita ser ajustado. Si el feedback es positivo, el flujo termina y se devuelven las recomendaciones al usuario. 
# Si el feedback es negativo, el router dirige el flujo los nodos refinadores para ajustar los vectores semánticos y/o auditivos, y luego regresa al nodo planificador para generar un nuevo plan de búsqueda basado en el feedback recibido.
# La idea que fuera condicional hacia 2 nodos... refinar
def router_juez(state: EstadoRecomendacionMusical) -> str:
    """
    Decide el siguiente paso basado en el feedback del evaluador.
    Si el feedback es positivo, termina el flujo. Si es negativo, decide qué dimensión refinar (semántica, auditiva, o ambas).
    """
    veredicto = state.get("veredicto", [])
    if not veredicto:
        return "Planificador de Búsqueda"  # Paso inicial

    ultimo_veredicto = veredicto[-1]
    if ultimo_veredicto.get("resultado") == "suficiente":
        return "aceptado"  # Terminar el flujo

    # refinamiento segpun verredicto
    if ultimo_veredicto.get("recomienda_refinamiento") == "semantico":
        return "refina_semantico"
    elif ultimo_veredicto.get("recomienda_refinamiento") == "auditivo":
        return "refina_auditivo"
    else:
        # Si no se especifica, refinamos ambos
        return "refina_semantico"  # Podríamos decidir cuál refinar primero o hacer ambos en paralelo

def router_ejecutor(state: EstadoRecomendacionMusical) -> str:
    """
    Decide qué herramienta ejecutar basado en el plan generado por el planificador.
    Si el plan incluye una dimensión semántica, ejecuta la búsqueda semántica. Si incluye una dimensión auditiva, ejecuta el filtrado auditivo.
    Si ya se hicieron todas las busquedas definidas, decide llamar al unificador de resultados (recoomendacion musical segpun puntajes auditivos y semánticos) para preparar la información para el evaluador.
    """
    # Aquí podríamos analizar el plan generado por el planificador para decidir qué herramienta ejecutar
    # Por simplicidad, asumiremos que siempre ejecutamos la búsqueda semántica primero
    if state.get("estado_recomendacion") == "finalizada":
        return "juzgar"  # Si ya se ejecutaron todas las herramientas, vamos a juzgar las recomendaciones
    else:
        return "usar herramienta"  # Por defecto, ejecuta la búsqueda semántica


##################################################################################################
#
# Ensamblando el grafo 
#
##################################################################################################

grafo_recomendacion = StateGraph(
    name="Grafo de Recomendación Musical"
)

grafo_recomendacion.add_node("Planificador de Búsqueda", nodo_planificador)
grafo_recomendacion.add_node("Ejecutor", nodo_ejecutor)
grafo_recomendacion.add_node("Evaluador de Recomendaciones", nodo_evaluador_recomendaciones)
grafo_recomendacion.add_node("Refinador Semántico", nodo_refinador_semantico)
grafo_recomendacion.add_node("Refinador Auditivo", nodo_refinador_auditivo)
grafo_recomendacion.add_node("Ejecutor de Herramientas", nodo_ejecutor_herramientas)

grafo_recomendacion.add_edge(START, "Planificador de Búsqueda")
grafo_recomendacion.add_edge("Planificador de Búsqueda", "Ejecutor")

grafo_recomendacion.add_conditional_edges("Ejecutor", 
                                          router_ejecutor,
                                          
                                          {
                                              "usar herramienta": "Ejecutor de Herramientas",
                                              "juzgar": "Evaluador de Recomendaciones",
                                          }
                                          
                                          )

grafo_recomendacion.add_conditional_edges("Evaluador de Recomendaciones", 
                                          router_juez,
                                          {
                                             "refina_semantico": "Refinador Semántico",
                                             "refina_auditivo": "Refinador Auditivo",
                                             "aceptado": END    
                                          }
                                          
                                          
                                            )       



grafo_recomendacion.add_edge("Refinador Semántico", "Ejecutor")
grafo_recomendacion.add_edge("Refinador Auditivo", "Ejecutor")
grafo_recomendacion.add_edge("Ejecutor", "Ejecutor de Herramientas", router=router_ejecutor)  # Desde el nodo ejecutor, podemos decidir ejecutar herramientas específicas según el plan
grafo_recomendacion.add_edge("Ejecutor de Herramientas", "Ejecutor")  # Después de ejecutar herramientas, volvemos al evaluador para juzgar las nuevas recomendaciones



from IPython.display import Image, display
app = grafo_recomendacion.compile()
try:
    # Esto genera y muestra una imagen PNG real de tu arquitectura
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Test directo de tarea_3.py")
    print("=" * 60)

    # Aquí podríamos simular una query de usuario y ejecutar el grafo para ver cómo fluye la información a través de los nodos y routers.
    # Por ejemplo:
    estado_inicial = EstadoRecomendacionMusical(
        query="canciones melancólicas con guitarra acústica",
        candidates=[],
        semantic_feedback=[],
        auditory_feedback=[],
        sem_vector=[],
        clap_vector=[],
        current_step="inicio",
        veredicto=[],
        estado_recomendacion="iniciada"
    )

    resultado = grafo_recomendacion.run(estado_inicial)
    print(f"Resultado final: {resultado}")