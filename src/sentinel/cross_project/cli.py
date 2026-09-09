"""Command-line entry point for Sentinel V4 cross-project evaluation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from sentinel.cross_project.data import load_multi_repository_datasets
from sentinel.cross_project.experiment import run_cross_project_evaluation
from sentinel.cross_project.reporting import (
    print_cross_project_report,
    print_repository_summaries,
    save_cross_project_artifacts,
)
from sentinel.ml.data import MLError
from sentinel.ml.models import DEFAULT_RANDOM_STATE


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Sentinel baselines with leave-one-project-out validation."
        )
    )
    parser.add_argument(
        "feature_csvs",
        nargs="+",
        type=Path,
        help="V2 feature CSVs containing at least two distinct repositories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cross-project"),
        help="V4 artifact directory (default: artifacts/cross-project)",
    )
    parser.add_argument(
        "--random-state",
        type=_non_negative_integer,
        default=DEFAULT_RANDOM_STATE,
        help=f"Deterministic model seed (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--same-project-metadata",
        type=Path,
        default=Path("artifacts/commons-lang/metadata.json"),
        help=(
            "V3 commons-lang metadata for same-vs-cross comparison "
            "(default: artifacts/commons-lang/metadata.json)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print("Loading and validating cross-project feature datasets...")
        dataset = load_multi_repository_datasets(args.feature_csvs)
        print(
            f"Validated {len(dataset.data):,} snapshots from "
            f"{len(dataset.summaries)} repositories with "
            f"{len(dataset.feature_columns)} historical predictors."
        )
        result = run_cross_project_evaluation(
            dataset,
            random_state=args.random_state,
            same_project_metadata=args.same_project_metadata,
            progress=print,
            retain_models=False,
        )
        print_repository_summaries(result)
        print_cross_project_report(result)
        paths = save_cross_project_artifacts(result, args.output_dir)
    except (
        MLError,
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nV4 artifacts")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
