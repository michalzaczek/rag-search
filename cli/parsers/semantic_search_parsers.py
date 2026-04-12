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

# chunk command
chunk_parser = subparsers.add_parser(
    "chunk",
    help='Chunk text by grouping n words together into a single string, where n is the "chunk size" parameter.',
)
chunk_parser.add_argument("text", type=str, help="Text to chunk")
chunk_parser.add_argument(
    "--chunk-size", type=int, default=200, help="Number of words per chunk"
)
chunk_parser.add_argument(
    "--overlap",
    type=int,
    default=0,
    help="Number of words to overlap between consecutive chunks",
)

# semantic_chunk command
semantic_chunk_parser = subparsers.add_parser(
    "semantic_chunk",
    help="Semantic chunk text by splitting input into individual sentences.",
)
semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
semantic_chunk_parser.add_argument(
    "--max-chunk-size", type=int, default=4, help="Max number of sentences per chunk"
)
semantic_chunk_parser.add_argument(
    "--overlap",
    type=int,
    default=0,
    help="Number of sentences to overlap between consecutive chunks",
)

args = parser.parse_args()
