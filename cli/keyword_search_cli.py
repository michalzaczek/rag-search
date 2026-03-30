import argparse

from core.utils import load_json_file, search_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            """
            Searching for: QUERY
            1. Movie Title 1
            2. Movie Title 2
            3. Movie Title 3
            ...
            """
            print(f"Searching for: {args.query}")
            movies = load_json_file("data/movies.json")["movies"]
            query = args.query.strip().lower()
            results = search_movies(query, movies)

            for i, result in enumerate(results[:5], start=1):
                print(f"{i}. {result['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
