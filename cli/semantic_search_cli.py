#!/usr/bin/env python3

import argparse
from cli.parsers.semantic_search import args, parser
from cli.lib.semantic_search import verify_model


def main():

    match args.command:
        case "verify":

            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
