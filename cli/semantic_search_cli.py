#!/usr/bin/env python3

from cli.lib import semantic_search
from cli.parsers.semantic_search_parsers import args, parser
from cli.lib.semantic_search import SemanticSearch, verify_model
from core.utils import load_json_file


def main():

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            semantic_search.embed_text(args.text)

        case "verify_embeddings":
            semantic_search.verify_embeddings()

        case "embedquery":
            semantic_search.embed_query_text(args.query)

        case "search":
            ss = SemanticSearch()
            documents = load_json_file("data/movies.json")["movies"]
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for idx, movie in enumerate(results, start=1):
                print(
                    f'{idx}. {movie["title"]} (score: {movie["score"]:.4f})\n{movie["description"]}\n'
                )

        case "chunk":
            words = args.text.split()
            chunk_size = args.chunk_size

            chunks = []

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i : chunk_size + i]
                chunk_text = " ".join(chunk_words)
                chunks.append(chunk_text)

            print(f"Chunking {len(args.text)} characters")

            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
