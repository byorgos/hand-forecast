from __future__ import annotations

import argparse
from pathlib import Path

from hand_surgery_forecast import (
    FUTURE_YEARS,
    build_forecast_paths,
    build_model_metrics,
    build_prediction_intervals,
    build_structural_shift,
    evaluate_all_models,
    load_aggregate_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the hand-surgery publication-trend forecast outputs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to observed_global_counts.csv (columns: year, hand_surgery_publications).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the four output CSVs into.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=400,
        help="Residual bootstrap iterations per model (default 400, matches manuscript).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap seed. Each model uses seed + its position in the model order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed = load_aggregate_csv(args.input)
    x = observed["year"].to_numpy(dtype=float)
    y = observed["hand_surgery_publications"].to_numpy(dtype=float)

    evaluations = evaluate_all_models(
        x,
        y,
        FUTURE_YEARS,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_model_metrics(observed, evaluations).to_csv(
        args.output_dir / "model_metrics.csv", index=False
    )
    build_forecast_paths(evaluations).to_csv(
        args.output_dir / "forecast_paths.csv", index=False
    )
    build_prediction_intervals(evaluations).to_csv(
        args.output_dir / "prediction_intervals.csv", index=False
    )
    build_structural_shift(x, y).to_csv(
        args.output_dir / "structural_shift.csv", index=False
    )


if __name__ == "__main__":
    main()
