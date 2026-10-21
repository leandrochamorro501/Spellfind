"""
router.py — SPELLFIND · Planificador (Manager)
=====================================================
Descarga el Router Prompt v4.1 desde el gist, construye la cadena
LangChain y expone get_router_chain() para ser importado por app.py.

Flujo:
    query (str)
        → ChatGroq (modelo configurado en GROQ_MODEL)
        → JsonOutputParser
        → dict con claves: clap_positive, clap_negative,
                           lyric_positive (list[str]),
                           lyric_negative

Regla de Oro: todos los campos del JSON se generan en inglés,
independientemente del idioma de la query original.
"""

from console import safe_print as print
import os
import json
import requests
from search_contract import normalize_plan, build_fallback_plan
from intent import prepare_plan
from config import PROMPT_PATH, BASE_DIR
from dotenv import load_dotenv  # Cargar variables de entorno desde .env
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(BASE_DIR / ".env")  # Carga GROQ_API_KEY y GROQ_MODEL del archivo .env

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — Plan A: API Key cargada desde archivo .env
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_URL = (
    "https://gist.githubusercontent.com/AlexandraJMV/" 
    "da3e4b9afaacd9a2a6808967c8ae4966/raw/" 
    "cd01fd2ef05e71699ebe40bc6219c489d16909c9"
)
PROMPT_CACHE_PATH = str(PROMPT_PATH)   # se guarda localmente tras la primera descarga

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ──────────────────────────────────────────────────────────────────────────────
# DESCARGA DEL PROMPT
# ──────────────────────────────────────────────────────────────────────────────

def _download_prompt() -> str:
    """
    Descarga el prompt desde el gist y lo cachea en disco.
    Si ya existe el archivo local, lo usa directamente (evita
    una petición de red en cada arranque del servidor).
    """
    if os.path.exists(PROMPT_CACHE_PATH):
        print(f"[ROUTER] Usando prompt cacheado en {PROMPT_CACHE_PATH}")
        with open(PROMPT_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[ROUTER] Descargando prompt desde gist...")
    try:
        response = requests.get(PROMPT_URL, timeout=15)
        response.raise_for_status()
        prompt_text = response.text

        # Guardar en disco para próximos arranques
        with open(PROMPT_CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        print(f"[ROUTER] Prompt descargado ({len(prompt_text)} chars). "
              f"Primeros 200: {prompt_text[:200]}...")
        return prompt_text

    except requests.RequestException as e:
        raise RuntimeError(
            f"No se pudo descargar el prompt desde el gist: {e}\n"
            f"Asegúrate de tener conexión a internet o coloca api_prompt.txt "
            f"manualmente en el directorio raíz del proyecto."
        )


# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK — plan degradado si el LLM falla
# ──────────────────────────────────────────────────────────────────────────────

def _build_fallback_plan(query: str) -> dict:
    """
    Si el LLM o el parser fallan, devuelve un plan mínimo usando
    la query cruda. Permite que app.py continúe con la búsqueda.
    """
    return build_fallback_plan(query)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA CADENA LANGCHAIN
# ──────────────────────────────────────────────────────────────────────────────

# Módulo-level cache: la cadena se construye una sola vez
_chain_cache = None

def get_router_chain():
    """
    Construye (o devuelve del caché) la cadena LangChain:
        prompt_template | ChatGroq | JsonOutputParser

    La cadena se invoca con:
        chain.invoke({"query": "..."})
    y devuelve un dict con las 4 claves del Router Prompt v4.1.
    """
    global _chain_cache
    if _chain_cache is not None:
        return _chain_cache

    # 1. Cargar el system prompt
    system_prompt = _download_prompt()
    contract_note = """\nCURRENT CONTRACT (overrides earlier always-two-modalities examples):
Return exactly these four required keys, each a list of strings:
clap_positive, clap_negative, lyric_positive, lyric_negative.
Do not rename clap_positive to audio_positive or omit empty lists.
Leave lyric lists empty for sound-only/instrumental requests. Leave audio lists
empty for lyrics-only requests. Return the four query lists plus constraints:
[{"modality":"lyrics" or "audio", "polarity":"positive" or "negative",
"text":"exact English phrase present in the corresponding query list",
"origin":"user_explicit" or "model_hypothesis", "source_quote":"verbatim user quote"}].
Every explicit restriction needs its complete verbatim quote, including negation.
Use minimal, literal descriptors for explicit constraints, not expanded scenes.
The application accepts only exact phrase matches and reviewed phrase translations
as explicit evidence. Unknown translations or ambiguous negation require
planning_status="needs_clarification", never a fabricated verified translation.
When one negated quote contains several concepts, preserve every concept with a
separate constraint referencing that same complete quote.
Interpretations inferred without direct evidence are model_hypothesis.
Do not invent restrictions or label assumptions as explicit user instructions.
Keep precise imagery for literal lyric requests. Never claim audio validation.
"""
    system_prompt += contract_note.replace("{", "{{").replace("}", "}}")

    # 2. Verificar API key desde .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY no encontrada en .env. "
            f"Verifica que el archivo {PROMPT_CACHE_PATH} tenga la clave configurada."
        )

    # 3. Prompt template
    #    El system prompt del gist ya contiene las instrucciones completas.
    #    Solo inyectamos la query del usuario como human message.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}"),
    ])

    # 4. Modelo
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=0,        # determinista: el router no debe ser creativo
        max_tokens=1024,      # incluye procedencia de restricciones
    )

    # 5. Parser con manejo de errores
    parser = JsonOutputParser()

    # 6. Cadena con fallback ante JSON malformado
    base_chain = prompt_template | llm | parser

    # Wrapper que captura fallos del parser y degrada graciosamente
    class RobustChain:
        def __init__(self, chain):
            self._chain = chain

        def invoke(self, inputs: dict) -> dict:
            query = inputs.get("query", "")
            try:
                result = self._chain.invoke(inputs)

                # Validar que el resultado tiene las claves mínimas esperadas
                return prepare_plan(result, query)

            except Exception as e:
                print(f"[ROUTER] ⚠ Parser/LLM falló: {e}. Usando fallback.")
                return _build_fallback_plan(query)

    _chain_cache = RobustChain(base_chain)
    print(f"[ROUTER] ✅ Cadena lista — modelo: {GROQ_MODEL}")
    return _chain_cache


# ──────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DEL PLAN
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"clap_positive", "clap_negative", "lyric_positive", "lyric_negative"}

def _validate_plan(plan: dict):
    """Valida y devuelve el contrato normalizado de consultas."""
    return normalize_plan(plan)


# ──────────────────────────────────────────────────────────────────────────────
# TEST RÁPIDO (ejecutar directamente: python router.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Test directo de router.py")
    print("=" * 60)

    chain = get_router_chain()

    test_queries = [
        "cancion que habla literal de un pequeño cerebro que se me va. se va mi cerebro",
    ]

    for q in test_queries:
        print(f"\n── Query: {q!r}")
        plan = chain.invoke({"query": q})
        print(json.dumps(plan, indent=2, ensure_ascii=False))
