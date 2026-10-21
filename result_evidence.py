"""Short lyric excerpts with honest origin labels, not fabricated validation."""
import re
from bisect import bisect_left, bisect_right


def lyric_excerpt(lyrics, validation, positive_queries, requested, limit=320):
    if not requested or not isinstance(lyrics, str) or not lyrics.strip():
        return None
    decision = validation.get("lyrics", {})
    quotes = decision.get("evidence", []) if decision.get("format") == "structured" else []
    for quote in quotes:
        if isinstance(quote, str) and quote.strip() and quote in lyrics:
            start = lyrics.index(quote)
            end = min(start + len(quote), start + limit)
            return dict(text=lyrics[start:end], start=start, end=end, source="judge_quote",
                        truncated=end < start + len(quote), label="Cita verificada del texto por el juez")
    # Lexical matching is only an excerpt selection heuristic, never a verdict.
    stop = {"the", "and", "with", "about", "song", "lyrics", "una", "con", "sobre", "letra", "cancion"}
    terms = {word.casefold() for text in positive_queries for word in re.findall(r"\w+", text)
             if len(word) > 2 and word.casefold() not in stop}
    matches = [match for match in re.finditer(r"\w+", lyrics) if match.group().casefold() in terms]
    positions = [match.start() for match in matches]
    anchor = max(positions, key=lambda pos: bisect_right(positions, pos + limit // 2)
                 - bisect_left(positions, pos - limit // 2)) if positions else 0
    start = max(0, anchor - limit // 3)
    if start and not lyrics[start-1].isspace():
        boundary = re.search(r"\s", lyrics[start:min(start+30, anchor)])
        if boundary:
            start += boundary.end()
    end = min(len(lyrics), start + limit)
    if end < len(lyrics) and not lyrics[end-1].isspace():
        boundaries = list(re.finditer(r"\s", lyrics[max(start, end-30):end]))
        if boundaries:
            end = max(start, end-30) + boundaries[-1].start()
    return dict(text=lyrics[start:end], start=start, end=end, source="lexical_excerpt" if matches else "opening_excerpt",
                truncated=start > 0 or end < len(lyrics), label="Fragmento orientativo; no es evidencia del juez")
