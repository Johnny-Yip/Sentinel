"""Command-line interface for the Sentinel repository miner."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sentinel.miner import (
    RepositoryMiningError,
    export_to_csv,
    mine_repository,
    validate_github_url,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine Java source-file changes from a public GitHub repository."
    )
    parser.add_argument(
        "repository_url",
        help="Public GitHub HTTPS URL, for example https://github.com/apache/commons-lang.git",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="CSV destination (default: <repository>_java_history.csv)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        repository = validate_github_url(args.repository_url)
        output_path = args.output or Path(f"{repository.name}_java_history.csv")
        data = mine_repository(args.repository_url)
        destination = export_to_csv(data, output_path)
    except RepositoryMiningError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Error: could not write CSV: {exc}", file=sys.stderr)
        return 1

    print(f"Mined {len(data)} Java file changes from {repository.slug}.")
    print(f"CSV written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

