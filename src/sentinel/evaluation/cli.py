"""Command-line interface for unified Sentinel evaluation reports."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from sentinel.cross_project.data import load_multi_repository_datasets
from sentinel.evaluation.calibration import DEFAULT_CALIBRATION_BINS, build_risk_calibration
from sentinel.evaluation.experiment import (
    evaluate_cross_project,
    evaluate_within_project,
)
from sentinel.evaluation.reporting import save_evaluation_report, save_risk_calibration
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
    _add_calibration_options(parser)
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
    parser.add_argument(
        "--explain-top-k",
        type=_positive_integer,
        help=(
            "Maximum feature-contribution rows per evaluated sample "
            "(default: explain all features)"
        ),
    )


def _add_calibration_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration-bins", type=_positive_integer, default=DEFAULT_CALIBRATION_BINS,
        help="Number of equal-width probability bins (default: 10)",
    )
    parser.add_argument(
        "--analysis-thresholds", type=_unit_interval, nargs="+",
        help="Retrospective threshold candidates (default: 0 to 1 in steps of 0.05)",
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
    calibration = commands.add_parser(
        "calibrate", help="Analyze an existing risk_ranking.csv without fitting models"
    )
    calibration.add_argument("risk_ranking_csv", type=Path)
    calibration.add_argument(
        "--evaluation-mode", choices=("within_project", "cross_project"), required=True,
        help="Evaluation scope of the source artifact",
    )
    calibration.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: source CSV directory); updates report.md",
    )
    _add_calibration_options(calibration)
    return parser


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.experiment_type == "calibrate":
        return args.risk_ranking_csv.parent
    if args.experiment_type == "within-project":
        stem = args.feature_csv.stem.removesuffix("_features")
        return Path("reports") / f"{stem}-within-project"
    return Path("reports") / "cross-project"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or _default_output_dir(args)
    try:
        if args.experiment_type == "calibrate":
            ranking = pd.read_csv(args.risk_ranking_csv, float_precision="round_trip")
            result = build_risk_calibration(
                ranking, experiment_type=args.evaluation_mode,
                calibration_bins=args.calibration_bins,
                analysis_thresholds=args.analysis_thresholds,
            )
            paths = save_risk_calibration(result, output_dir)
        elif args.experiment_type == "within-project":
            print(f"Loading and validating {args.feature_csv}...")
            data = load_dataset(args.feature_csv)
            report = evaluate_within_project(
                data,
                random_state=args.random_state,
                top_risk=args.top_risk,
                risk_threshold=args.risk_threshold,
                explain_top_k=args.explain_top_k,
                calibration_bins=args.calibration_bins,
                analysis_thresholds=args.analysis_thresholds,
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
                explain_top_k=args.explain_top_k,
                calibration_bins=args.calibration_bins,
                analysis_thresholds=args.analysis_thresholds,
                progress=print,
            )
        if args.experiment_type != "calibrate":
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
