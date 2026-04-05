import argparse


parser = argparse.ArgumentParser(description="Semantic Search CLI")
subparsers = parser.add_subparsers(
    dest="command", required=True, help="Available commands"
)

# verify command
subparsers.add_parser("verify", help="Verify by printing the model information.")

args = parser.parse_args()
