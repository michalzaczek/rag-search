import argparse

from core.index import InvertedIndex
from core.utils import load_json_file, search_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build inverted index")
    subparsers.add_parser("load", help="Load inverted index")

    args = parser.parse_args()

    index = InvertedIndex()

    match args.command:

        case "load":
            # load the index from disk. If it doesn't exist, just print an error message and exit.
            print("Loading inverted index...")
            try:

                index.load()
                print("Inverted index loaded successfully.")
                movies_data = index.docmap
                print(f"Loaded {len(movies_data)} movies.")

            except FileNotFoundError:
                print("Inverted index not found. Please build it first.")
                exit(1)

        case "build":
            print("Building inverted index...")

            index = InvertedIndex()
            movies_data = load_json_file("data/movies.json")["movies"]
            index.build(movies_data)
            index.save()

        case "search":
            index.load()
            query_text = args.query
            print(f"Searching for: {query_text}")
            results = search_movies(query_text, index.index, index.docmap)

            for i, result in enumerate(results[:5], start=1):
                print(f"{i}. {result['title']}")

            print(f"\nFound {len(results)} results")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
