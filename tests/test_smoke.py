from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "data" / "observed_global_counts.csv"
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_forecast.py"

EXPECTED_MODELS = [
    "Linear",
    "Exponential",
    "Holt",
    "Piecewise",
    "Logistic",
    "New Baseline",
    "InterventionLevelShift",
]

REFERENCE_2030 = {
    "Linear": 1342.1691176471,
    "Exponential": 1439.0904402643,
    "Holt": 1637.8733284738,
    "Piecewise": 1822.8260869565,
    "Logistic": 1399.6525858552,
    "New Baseline": 1571.1504638471,
    "InterventionLevelShift": 1485.6964285714,
}


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("forecast_out")
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--input",
            str(INPUT_CSV),
            "--output-dir",
            str(out),
            "--bootstrap-iterations",
            "400",
            "--seed",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return out


def test_all_four_csvs_written(output_dir: Path) -> None:
    expected = {
        "model_metrics.csv",
        "forecast_paths.csv",
        "prediction_intervals.csv",
        "structural_shift.csv",
    }
    assert {p.name for p in output_dir.glob("*.csv")} >= expected


def test_model_metrics_shape(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "model_metrics.csv")
    assert list(df["model"]) == EXPECTED_MODELS
    for column in [
        "R",
        "R2",
        "AIC",
        "BIC",
        "MAE",
        "RMSE",
        "MAPE",
        "observed_2025",
        "projection_2030",
    ]:
        assert column in df.columns


def test_forecast_paths_shape(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "forecast_paths.csv")
    assert set(df.columns) == {"model", "year", "point_forecast"}
    for model in EXPECTED_MODELS:
        sub = df[df["model"] == model].sort_values("year")
        assert len(sub) == 5
        assert list(sub["year"]) == [2026, 2027, 2028, 2029, 2030]


def test_prediction_intervals_shape(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "prediction_intervals.csv")
    assert set(df.columns) == {"model", "year", "lower_95", "upper_95"}
    assert (df["lower_95"] <= df["upper_95"]).all()
    for model in EXPECTED_MODELS:
        sub = df[df["model"] == model].sort_values("year")
        assert len(sub) == 5
        assert list(sub["year"]) == [2026, 2027, 2028, 2029, 2030]


def test_structural_shift_shape(output_dir: Path) -> None:
    df = pd.read_csv(output_dir / "structural_shift.csv")
    assert len(df) == 1
    assert set(df.columns) == {
        "observed_2025",
        "expected_2025",
        "expected_lower_95",
        "expected_upper_95",
        "in_range",
        "intervention_coefficient",
        "p_value",
        "delta_aic",
    }
    row = df.iloc[0]
    assert 0.0 <= float(row["p_value"]) <= 1.0
    assert float(row["observed_2025"]) == 1333.0
    assert float(row["expected_lower_95"]) < float(row["expected_upper_95"])


@pytest.mark.parametrize("model,expected", list(REFERENCE_2030.items()))
def test_2030_projection_matches_reference(output_dir: Path, model: str, expected: float) -> None:
    df = pd.read_csv(output_dir / "model_metrics.csv")
    actual = float(df.loc[df["model"] == model, "projection_2030"].iloc[0])
    assert actual == pytest.approx(expected, rel=1e-6), (
        f"{model} 2030 projection {actual:.6f} drifted from reference {expected:.6f}"
    )


def test_run_is_deterministic(tmp_path: Path) -> None:
    runs = []
    for label in ("first", "second"):
        out = tmp_path / label
        subprocess.run(
            [
                sys.executable,
                str(RUN_SCRIPT),
                "--input",
                str(INPUT_CSV),
                "--output-dir",
                str(out),
                "--bootstrap-iterations",
                "400",
                "--seed",
                "42",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        runs.append(out)

    for csv_name in (
        "model_metrics.csv",
        "forecast_paths.csv",
        "prediction_intervals.csv",
        "structural_shift.csv",
    ):
        a = (runs[0] / csv_name).read_bytes()
        b = (runs[1] / csv_name).read_bytes()
        assert a == b, f"{csv_name} not byte-equal across runs"
