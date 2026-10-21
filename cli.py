"""Console interface to the same hybrid workflow served by Flask."""
import argparse
import json
from console import safe_print as print


def main():
    from app import engine, get_top_k_results
    parser = argparse.ArgumentParser(description="SPELLFIND hybrid music search")
    parser.add_argument("query", help="Consulta musical en lenguaje natural")
    parser.add_argument("--lyrics-weight", type=float, help="Peso de letras entre 0 y 1; audio usa el complemento")
    parser.add_argument("--artist", help="Artista exacto del catálogo")
    parser.add_argument("--genre", help="Género exacto del campo tag")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    args = parser.parse_args()
    from search_options import validate_weights, validate_filters
    try:
        weights = validate_weights({"lyrics": args.lyrics_weight, "audio": 1-args.lyrics_weight}) if args.lyrics_weight is not None else None
        filters = validate_filters({name: value for name, value in dict(
            artist=args.artist, genre=args.genre, year_min=args.year_min, year_max=args.year_max).items() if value is not None})
    except ValueError as exc:
        parser.error(str(exc))
    engine.load_indices()
    result = get_top_k_results(args.query, weights=weights, filters=filters)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("error") or result.get("judge_status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
