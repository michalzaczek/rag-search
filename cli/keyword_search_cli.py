from core.constans import BM25_B, BM25_K1
from core.index import InvertedIndex
from core.utils import load_json_file, search_movies
from cli.parsers.keyword_search_parsers import args, parser


def bm25_idf_command(term: str) -> None:
    index = InvertedIndex()
    index.load()

    try:
        bm25idf = index.get_bm25_idf(term)
        print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> None:
    index = InvertedIndex()
    index.load()

    try:
        bm25tf = index.get_bm25_tf(doc_id, term, k1, b)
        print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


def main() -> None:

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

        case "bm25idf":
            bm25_idf_command(args.term)

        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term)

        case "bm25search":
            index.load()
            if not index.docmap:
                print("Error: Index not built. Run 'build' command first.")
                exit(1)

            query_text = args.query
            limit_val = args.limit

            results = index.bm25_search(query_text, limit_val)

            if not results:
                print("No results found.")
            else:
                for i, result in enumerate(results, start=1):
                    doc = result["document"]
                    score = result["score"]
                    print(f"{i}. ({doc['id']}) {doc['title']} - Score: {score:.2f}")

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
