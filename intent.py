"""Quoted constraint provenance and a deliberately bounded offline fallback."""
import hashlib
import re
import unicodedata
from copy import deepcopy

KEYS = ("lyric_positive", "lyric_negative", "clap_positive", "clap_negative")
PREFIX = {"lyrics": "lyric", "audio": "clap"}
ALIASES = {
    "piano": "piano", "guitarra acustica": "acoustic guitar", "guitarra": "guitar",
    "acoustic guitar": "acoustic guitar", "guitar": "guitar", "bateria": "drums", "drums": "drums",
    "sintetizadores": "synthesizers", "sintetizador": "synthesizers", "synthesizers": "synthesizers",
    "voces": "vocals", "voz": "vocals", "canto": "vocals", "vocals": "vocals", "singing": "vocals",
    "instrumental": "instrumental", "lento": "slow", "lenta": "slow", "slow": "slow",
    "suave": "soft", "soft": "soft", "tranquila": "quiet", "tranquilo": "quiet", "quiet": "quiet",
    "relajante": "relaxing", "relaxing": "relaxing", "rapida": "fast", "rapido": "fast", "fast": "fast",
    "fuerte": "loud", "loud": "loud", "energetica": "energetic", "energetico": "energetic",
    "energetic": "energetic", "rock": "rock", "jazz": "jazz", "electronica": "electronic",
}
STOP = {"musica", "music", "una", "un", "a", "the", "con", "with", "y", "and", "de", "en"}
NEGATION = re.compile(r"\b(sin|without|no|not|avoid|evita|excluir)\b", re.I)

# Reviewed phrase equivalences, not model-supplied claims of a translation.
# Unknown translations fail closed instead of being promoted to user evidence.
LYRIC_ALIASES = {
    "romance": "romance", "amor romantico": "romance", "amor romántico": "romance", "romantic love": "romance",
    "violencia": "violence", "violence": "violence",
    "guerra": "war", "war": "war", "tristeza": "sadness", "sadness": "sadness",
    "soledad": "loneliness", "loneliness": "loneliness",
    "volver a casa": "returning home", "returning home": "returning home",
}
AUDIO_GROUNDING_ALIASES = {**ALIASES,
    "guitarra acústica": "acoustic guitar", "batería": "drums", "rápida": "fast",
    "rápido": "fast", "enérgica": "energetic", "enérgico": "energetic", "electrónica": "electronic"}


def phrase(text):
    # Keep accents: removing them can change meaning (sí/si, él/el).
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def canonical_concept(text, modality):
    value = phrase(text)
    aliases = LYRIC_ALIASES if modality == "lyrics" else AUDIO_GROUNDING_ALIASES
    # Unknown phrases retain case as well as accents (US/us, Polish/polish).
    return aliases.get(value.casefold(), value)


def quoted_concepts(item):
    quote = phrase(item["source_quote"])
    if item["polarity"] == "negative":
        match = re.fullmatch(r"(?:sin|without|no|not|avoid|evita|excluir)\s+(.+)", quote, re.I)
        if not match:
            raise ValueError("Negative quote requires an unambiguous leading negation")
        if re.match(r"(?:solo|solamente|only|just)\b", match[1], re.I):
            raise ValueError("Not-only constructions have ambiguous negative scope")
        parts = re.split(r"\s+(?:y|and|ni)\s+", match[1], flags=re.I)
    else:
        parts = [quote]
    if any(NEGATION.search(part) or not part for part in parts):
        raise ValueError("Ambiguous or nested negation in explicit constraint")
    return {canonical_concept(part, item["modality"]) for part in parts}


def folded(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(c) != "Mn")


def constraint(modality, polarity, text, origin="model_hypothesis", source_quote=""):
    identity = "|".join((modality, polarity, text, origin, source_quote))
    return {"id": hashlib.sha256(identity.encode()).hexdigest()[:16], "modality": modality,
            "polarity": polarity, "text": text, "origin": origin, "source_quote": source_quote}


def quoted_constraint(item, query):
    if not isinstance(item, dict) or item.get("modality") not in PREFIX or item.get("polarity") not in {"positive", "negative"}:
        raise ValueError("Invalid constraint schema")
    text, quote = item.get("text"), item.get("source_quote", "")
    if not isinstance(text, str) or not text.strip() or not isinstance(quote, str):
        raise ValueError("Invalid constraint text or quote")
    origin = item.get("origin", "model_hypothesis")
    if origin not in {"user_explicit", "model_hypothesis"}:
        raise ValueError("Invalid constraint origin")
    if origin == "user_explicit":
        query = unicodedata.normalize("NFC", query)
        quote = unicodedata.normalize("NFC", quote)
        if not quote or quote not in query:
            raise ValueError("Explicit constraints require a verbatim user quote")
        left = r"(?<!\w)" if quote[0].isalnum() or quote[0] == "_" else ""
        right = r"(?!\w)" if quote[-1].isalnum() or quote[-1] == "_" else ""
        if not re.search(left + re.escape(quote) + right, query):
            raise ValueError("Explicit quotes must cover complete words")
        if item["polarity"] == "negative" and not NEGATION.search(folded(quote)):
            raise ValueError("Negative constraint quote must include its negation")
        if NEGATION.search(text):
            raise ValueError("Constraint descriptors must not reverse polarity")
        if canonical_concept(text, item["modality"]) not in quoted_concepts(dict(item, source_quote=quote)):
            raise ValueError("Explicit constraint translation is not verified; clarification required")
        if item["polarity"] == "negative":
            for occurrence in re.finditer(re.escape(quote), query):
                tail = query[occurrence.end():]
                if re.match(r"\s*(?:,|\b(?:y|and|ni|or|o)\b)", tail, re.I):
                    raise ValueError("Negative quote omits a continued list; include its full scope")
    return constraint(item["modality"], item["polarity"], text.strip(), origin, quote)


def audio_fallback(query):
    """Recognizes only fully covered audio phrases, never arbitrary lyric intent."""
    original = unicodedata.normalize("NFC", query)
    text = folded(original)
    # Offsets used for source quotes require one folded character per original character.
    if len(text) != len(original):
        return None
    aliases = sorted(ALIASES, key=len, reverse=True)
    term = "(?:" + "|".join(re.escape(t) for t in aliases) + ")"
    negative = re.compile(r"\b(?:sin|without|no|not|avoid|evita)\s+(" + term +
                          r"(?:\s+(?:y|and|ni|or|o)\s+" + term + r")*)\b")
    constraints, negatives = [], []
    masked = list(text)
    for match in negative.finditer(text):
        if text[match.end():].lstrip().startswith(","):
            return None  # Do not guess whether a following list remains negated.
        tail = text[match.end():].strip()
        if re.search(r"\w", tail) and not re.match(r"^(con|with|sin|without|no|not|avoid|evita)\b", tail):
            return None  # A bare following term has uncertain negative scope.
        quote = original[match.start():match.end()]
        for found in re.finditer(r"\b" + term + r"\b", match.group(1)):
            value = ALIASES[found.group()]
            negatives.append(value)
            constraints.append(constraint("audio", "negative", value, "user_explicit", quote))
        masked[match.start():match.end()] = " " * (match.end() - match.start())
    rest = "".join(masked)
    positives = []
    for match in re.finditer(r"\b" + term + r"\b", rest):
        value = ALIASES[match.group()]
        positives.append(value)
        constraints.append(constraint("audio", "positive", value, "user_explicit",
                                      original[match.start():match.end()]))
        masked[match.start():match.end()] = " " * (match.end() - match.start())
    unresolved = [word for word in re.findall(r"\w+", "".join(masked)) if word not in STOP]
    # A mood alone does not establish whether the user means lyrics or sound.
    anchors = {"piano", "guitar", "acoustic guitar", "drums", "synthesizers", "vocals",
               "instrumental", "rock", "jazz", "electronic"}
    if unresolved or set(positives) & set(negatives) or not (set(positives) & anchors):
        return None
    return {"lyric_positive": [], "lyric_negative": [],
            "clap_positive": list(dict.fromkeys(positives)),
            "clap_negative": list(dict.fromkeys(negatives)), "constraints": constraints,
            "planning_status": "ready", "interpretation": "bounded_audio_grammar"}


def prepare_plan(raw, query, previous=None):
    from search_contract import normalize_plan
    plan = normalize_plan(raw)
    if plan.get("planning_status") == "needs_clarification":
        result = {**plan, **{key: [] for key in KEYS}, "constraints": [], "intent_query": query}
        import json
        json.dumps(result, allow_nan=False)
        return result
    supplied = plan.get("constraints", [])
    if not isinstance(supplied, list):
        raise ValueError("Constraints must be a list")
    explicit = [quoted_constraint(item, query) for item in supplied]
    # Only a completely recognized grammar can override a gratuitous lyric branch.
    recognized = audio_fallback(query)
    if recognized:
        plan.update({key: recognized[key] for key in KEYS})
        explicit = recognized["constraints"]
        plan["interpretation"] = recognized["interpretation"]
    if previous:
        for item in previous.get("constraints", []):
            if item["origin"] == "user_explicit" and item not in explicit:
                explicit.append(quoted_constraint(deepcopy(item), query))
    for item in explicit:
        if item["origin"] == "user_explicit" and item["polarity"] == "negative":
            represented = {canonical_concept(other["text"], other["modality"]) for other in explicit
                if other["origin"] == "user_explicit" and other["polarity"] == "negative"
                and other["modality"] == item["modality"] and other["source_quote"] == item["source_quote"]}
            if quoted_concepts(item) != represented:
                raise ValueError("Explicit negative quote contains an unrepresented restriction")
    positives = {(item["modality"], canonical_concept(item["text"], item["modality"]))
                 for item in explicit if item["origin"] == "user_explicit" and item["polarity"] == "positive"}
    negatives = {(item["modality"], canonical_concept(item["text"], item["modality"]))
                 for item in explicit if item["origin"] == "user_explicit" and item["polarity"] == "negative"}
    if positives & negatives:
        raise ValueError("Contradictory explicit constraints require clarification")
    # A legacy/malformed router must not silently discard a visible user negation.
    # Translation and conjunction coverage were checked above before marker coverage.
    original = unicodedata.normalize("NFC", query)
    for marker in NEGATION.finditer(folded(original)):
        covered = False
        for item in explicit:
            if item["origin"] != "user_explicit" or item["polarity"] != "negative":
                continue
            if any(match.start() <= marker.start() and marker.end() <= match.end()
                   for match in re.finditer(re.escape(item["source_quote"]), original)):
                covered = True
                break
        if not covered:
            raise ValueError("A user negation has no explicit quoted constraint")
    records = []
    for item in explicit:
        key = PREFIX[item["modality"]] + "_" + item["polarity"]
        if item["origin"] == "user_explicit":
            if item["text"] not in plan[key]:
                raise ValueError("Plan removed an explicit user constraint")
        if item["text"] in plan[key] and item not in records:
            records.append(item)
    for key in KEYS:
        modality = "lyrics" if key.startswith("lyric") else "audio"
        polarity = key.split("_")[1]
        for text in plan[key]:
            if not any(c["modality"] == modality and c["polarity"] == polarity and c["text"] == text for c in records):
                records.append(constraint(modality, polarity, text))
    plan.update(constraints=records, intent_query=query, planning_status="ready")
    import json
    json.dumps(plan, allow_nan=False)  # reject opaque or non-finite plan metadata
    return plan


def plan_fingerprint(plan):
    import json
    # Lists are sets of descriptors, not a sequence of retrieval operations.
    # Keep provenance in the fingerprint, but ignore its presentation order.
    records = sorted({json.dumps(c, sort_keys=True, ensure_ascii=False)
                      for c in plan.get("constraints", [])})
    return hashlib.sha256(json.dumps({**{k: sorted(set(plan[k])) for k in KEYS},
        "constraints": records}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fallback(query, last_valid_plan=None):
    if isinstance(last_valid_plan, dict) and last_valid_plan.get("intent_query") == query:
        try:
            plan = prepare_plan(deepcopy(last_valid_plan), query)
            if plan["planning_status"] == "ready":
                return {**plan, "_fallback": True, "degradation": "reused_valid_plan"}
        except ValueError:
            pass
    plan = audio_fallback(query)
    if plan:
        return {**prepare_plan(plan, query), "_fallback": True, "degradation": "bounded_audio_grammar"}
    return {**{key: [] for key in KEYS}, "planning_status": "needs_clarification",
            "intent_query": query, "constraints": [], "_fallback": True,
            "degradation": "interpretation_unavailable",
            "planning_warning": "No se pudo interpretar la consulta con seguridad. Especifica sonido, letra y restricciones por separado o vuelve a intentarlo con el router disponible."}
