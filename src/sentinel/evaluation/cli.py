"""Command-line interface for unified Sentinel evaluation reports."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from sentinel.cross_project.data import load_multi_repository_datasets
from sentinel.evaluation.experiment import (
    evaluate_cross_project,
    evaluate_within_project,
)
from sentinel.evaluation.reporting import save_evaluation_report
from sentinel.evaluation.risk import DEFAULT_RISK_THRESHOLD, DEFAULT_TOP_RISK
from sentinel.ml.data import MLError, load_dataset
from sentinel.ml.models import DEFAULT_RANDOM_STATE


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for this experiment's unified report artifacts",
    )
    parser.add_argument(
        "--random-state",
        type=_non_negative_integer,
        default=DEFAULT_RANDOM_STATE,
        help=f"Deterministic model seed (default: {DEFAULT_RANDOM_STATE})",
    )
    parser.add_argument(
        "--top-risk",
        type=_positive_integer,
        default=DEFAULT_TOP_RISK,
        help=(
            "Maximum high-risk rows in risk_summary.json "
            f"(default: {DEFAULT_TOP_RISK})"
        ),
    )
    parser.add_argument(
        "--risk-threshold",
        type=_unit_interval,
        default=DEFAULT_RISK_THRESHOLD,
        help=(
            "Score cutoff for the high-risk summary count "
            f"(default: {DEFAULT_RISK_THRESHOLD})"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Sentinel models and generate a unified V6 report."
    )
    commands = parser.add_subparsers(dest="experiment_type", required=True)

    within = commands.add_parser(
        "within-project", help="Run chronological evaluation on one feature CSV"
    )
    within.add_argument("feature_csv", type=Path)
    _add_common_options(within)

    cross = commands.add_parser(
        "cross-project", help="Run leave-one-project-out evaluation"
    )
    cross.add_argument("feature_csvs", nargs="+", type=Path)
    cross.add_argument(
        "--same-project-metadata",
        type=Path,
        help="Optional V3 metadata.json for same-vs-cross comparison",
    )
    _add_common_options(cross)
    return parser


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.experiment_type == "within-project":
        stem = args.feature_csv.stem.removesuffix("_features")
        return Path("reports") / f"{stem}-within-project"
    return Path("reports") / "cross-project"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or _default_output_dir(args)
    try:
        if args.experiment_type == "within-project":
            print(f"Loading and validating {args.feature_csv}...")
            data = load_dataset(args.feature_csv)
            report = evaluate_within_project(
                data,
                random_state=args.random_state,
                top_risk=args.top_risk,
                risk_threshold=args.risk_threshold,
                progress=print,
            )
        else:
            print("Loading and validating cross-project datasets...")
            dataset = load_multi_repository_datasets(args.feature_csvs)
            report = evaluate_cross_project(
                dataset,
                random_state=args.random_state,
                same_project_metadata=args.same_project_metadata,
                top_risk=args.top_risk,
                risk_threshold=args.risk_threshold,
                progress=print,
            )
        paths = save_evaluation_report(report, output_dir)
    except (
        MLError,
        OSError,
        UnicodeError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nV6 report directory: {Path(output_dir).expanduser().resolve()}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
