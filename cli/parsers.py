import argparse
from core.constans import BM25_K1


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

# bm25idf command
bm25_idf_parser = subparsers.add_parser(
    "bm25idf", help="Get BM25 IDF score for a given term"
)
bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

# bm25tf command
bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
)
bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
bm25_tf_parser.add_argument(
    "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
)

args = parser.parse_args()
