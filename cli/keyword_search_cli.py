import argparse

from core.utils import load_json_file, remove_punctuation, search_movies, tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    query_text = args.query

    match args.command:
        case "search":
            print(f"Searching for: {query_text}")
            movies = load_json_file("data/movies.json")["movies"]
            results = search_movies(query_text, movies)

            for i, result in enumerate(results[:5], start=1):
                print(f"{i}. {result['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
