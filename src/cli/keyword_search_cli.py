import argparse

from src.core.utils import load_json_file, search_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            movies = load_json_file("data/movies.json")["movies"]
            query = args.query.strip().lower()
            results = search_movies(query, movies)
            print(f"Found {len(results)} results")
            print("First 5 results:")
            for result in results[:5]:
                print(f"- {result['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
