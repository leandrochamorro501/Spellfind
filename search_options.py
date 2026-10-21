"""Validated request controls; metadata filters are distinct from semantic hints."""
import math
import unicodedata


def exact_text(value):
    return unicodedata.normalize("NFC", value).strip().casefold()


def validate_weights(weights):
    if weights is None:
        return None
    if not isinstance(weights, dict) or set(weights) != {"lyrics", "audio"}:
        raise ValueError("weights debe incluir únicamente lyrics y audio")
    for value in weights.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Los pesos deben ser números finitos entre 0 y 1")
    if not math.isclose(sum(weights.values()), 1., rel_tol=0., abs_tol=1e-9):
        raise ValueError("Los pesos de letra y audio deben sumar 1")
    return {name: float(weights[name]) for name in ("lyrics", "audio")}


def validate_filters(filters):
    if filters is None:
        return {}
    allowed = {"artist", "genre", "year_min", "year_max"}
    if not isinstance(filters, dict) or set(filters) - allowed:
        raise ValueError("filters admite artist, genre, year_min y year_max")
    result = dict(filters)
    for key in ("artist", "genre"):
        if key in result:
            value = result[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 300:
                raise ValueError(f"{key} debe ser un texto no vacío de hasta 300 caracteres")
            result[key] = unicodedata.normalize("NFC", value).strip()
    for key in ("year_min", "year_max"):
        if key in result and (isinstance(result[key], bool) or not isinstance(result[key], int)
                              or not 1 <= result[key] <= 9999):
            raise ValueError(f"{key} debe ser un año entero entre 1 y 9999")
    if result.get("year_min", 1) > result.get("year_max", 9999):
        raise ValueError("El año inicial no puede ser posterior al final")
    return result


def catalog_year(value):
    """Do not truncate fractional years or interpret missing values as a date."""
    try:
        if isinstance(value, bool) or str(value).strip() == "":
            return None
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() and 1 <= number <= 9999 else None
    except (ValueError, TypeError):
        return None


def matches_metadata(row, filters):
    for field, column in (("artist", "artist"), ("genre", "tag")):
        if field in filters and exact_text(str(row.get(column, ""))) != exact_text(filters[field]):
            return False
    if "year_min" in filters or "year_max" in filters:
        year = catalog_year(row.get("year"))
        if year is None or not filters.get("year_min", 1) <= year <= filters.get("year_max", 9999):
            return False
    return True


def catalog_options(rows):
    years = sorted({year for row in rows if (year := catalog_year(row.get("year"))) is not None})
    return dict(artists=sorted({str(row.get("artist", "")).strip() for row in rows if str(row.get("artist", "")).strip()}, key=exact_text),
                genres=sorted({str(row.get("tag", "")).strip() for row in rows if str(row.get("tag", "")).strip()}, key=exact_text),
                year_min=min(years) if years else None, year_max=max(years) if years else None,
                songs=len(rows), semantics="exact_nfc_casefold_trim_and_inclusive_years")
