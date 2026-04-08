import argparse


parser = argparse.ArgumentParser(description="Semantic Search CLI")
subparsers = parser.add_subparsers(
    dest="command", required=True, help="Available commands"
)

# verify command
subparsers.add_parser("verify", help="Verify by printing the model information.")

# embed_text command
embed_parser = subparsers.add_parser("embed_text", help="Embed typed text.")
embed_parser.add_argument("text", type=str, help="Text to embed")

# verify_embeddings command
subparsers.add_parser(
    "verify_embeddings",
    help="Verify embeddings by printing num and dimensions of vectors.",
)

# embedquery command
embedquery_parser = subparsers.add_parser("embedquery", help="Embed typed query.")
embedquery_parser.add_argument("query", type=str, help="Query text to embed")

# search command
search_parser = subparsers.add_parser(
    "search", help="Search movies using semantic search scoring"
)
search_parser.add_argument("query", type=str, help="Search query")
search_parser.add_argument(
    "--limit", type=int, default=5, help="Number of results to return"
)

args = parser.parse_args()
