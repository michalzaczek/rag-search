import argparse

from core.index import InvertedIndex
from core.utils import load_json_file, search_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # search command
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # build command
    subparsers.add_parser("build", help="Build inverted index")
    # load command
    subparsers.add_parser("load", help="Load inverted index")

    # tf command
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to check")

    # idf command
    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to check")

    # tfidf command
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check")

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

            movies_data = load_json_file("data/movies.json")["movies"]
            index.build(movies_data)
            index.save()

        case "tf":
            index.load()

            if not index.docmap:
                print("Error: Index not built. Run 'build' command first.")
                exit(1)

            try:
                tf_score = index.get_tf(args.doc_id, args.term)
                print(tf_score)

            except ValueError as e:
                print(f"Error: {e}")
                exit(1)

        case "idf":
            index.load()
            try:
                idf_score = index.get_idf(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf_score:.2f}")
            except ValueError as e:
                print(f"Error: {e}")
                exit(1)

        case "tfidf":
            index.load()
            try:
                tf_idf = index.get_tfidf(args.doc_id, args.term)
                print(
                    f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
                )
            except ValueError as e:
                print(f"Error: {e}")
                exit(1)

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
