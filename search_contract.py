"""Dependency-free normalization shared by the router and web application."""

PLAN_KEYS = ("lyric_positive", "lyric_negative", "clap_positive", "clap_negative")


def normalize_plan(plan: dict) -> dict:
    if not isinstance(plan, dict) or any(key not in plan for key in PLAN_KEYS):
        raise ValueError("El plan debe incluir las cuatro listas de consulta")
    result = dict(plan)
    for key in PLAN_KEYS:
        value = plan[key]
        if key == "lyric_positive" and isinstance(value, dict):
            weight = value.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                raise ValueError("Peso lírico antiguo inválido")
            value = value.get("text")
            if not isinstance(value, str):
                raise ValueError("Texto lírico antiguo inválido")
        if value is None:
            value = []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{key} debe ser una lista de textos")
        result[key] = list(dict.fromkeys(item.strip() for item in value if item.strip()))
    if result.get("planning_status", "ready") not in {"ready", "needs_clarification"}:
        raise ValueError("Estado de planificación inválido")
    if not result["lyric_positive"] and not result["clap_positive"] and result.get("planning_status") != "needs_clarification":
        raise ValueError("El plan no contiene consultas positivas")
    return result


def build_fallback_plan(query: str, last_valid_plan=None) -> dict:
    from intent import fallback
    return fallback(query, last_valid_plan)
