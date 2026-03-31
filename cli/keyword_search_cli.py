import argparse

from core.index import InvertedIndex
from core.utils import load_json_file, remove_punctuation, search_movies, tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()

    movies_data = load_json_file("data/movies.json")["movies"]

    match args.command:
        case "build":
            print("Building inverted index...")

            index = InvertedIndex()
            index.build(movies_data)
            index.save()

        case "search":
            query_text = args.query
            print(f"Searching for: {query_text}")
            results = search_movies(query_text, movies_data)

            for i, result in enumerate(results[:5], start=1):
                print(f"{i}. {result['title']}")

            print(f"\nFound {len(results)} results")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
