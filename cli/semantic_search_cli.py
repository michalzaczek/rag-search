#!/usr/bin/env python3

import argparse
from cli.lib import semantic_search
from cli.parsers.semantic_search_parsers import args, parser
from cli.lib.semantic_search import verify_model


def main():

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            semantic_search.embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
