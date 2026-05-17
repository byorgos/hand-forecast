from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


FULL_MODEL_ORDER = (
    "Linear",
    "Exponential",
    "Holt",
    "Piecewise",
    "Logistic",
    "New Baseline",
    "InterventionLevelShift",
)
FUTURE_YEARS = np.arange(2026, 2031, dtype=float)


@dataclass
class FitResult:
    name: str
    fitted: np.ndarray
    params: object
    r2: float
    r: float
    aic: float
    bic: float
    sse: float
    note: str = ""


@dataclass
class ModelEvaluation:
    name: str
    fit: FitResult
    future_pred: np.ndarray
    pi_low: np.ndarray
    pi_high: np.ndarray
    rolling_n: int
    mae: float
    rmse: float
    mape: float
    bootstrap_successes: int
    bootstrap_iterations: int


def load_aggregate_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"year", "hand_surgery_publications"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing required columns: {sorted(missing)}")
    df = df[["year", "hand_surgery_publications"]].copy()
    df["year"] = df["year"].astype(int)
    df["hand_surgery_publications"] = df["hand_surgery_publications"].astype(float)
    return df.sort_values("year").reset_index(drop=True)


def info_criteria(sse: float, n_obs: int, n_params: int) -> tuple[float, float]:
    if n_obs <= 0 or sse <= 0:
        return float("nan"), float("nan")
    aic = n_obs * math.log(sse / n_obs) + 2 * n_params
    bic = n_obs * math.log(sse / n_obs) + n_params * math.log(n_obs)
    return aic, bic


def corr_value(actual: np.ndarray, fitted: np.ndarray) -> float:
    if len(actual) < 2:
        return float("nan")
    return float(np.corrcoef(actual, fitted)[0, 1])


def metrics(actual: list[float], predicted: list[float]) -> tuple[float, float, float]:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    errors = actual_arr - pred_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / actual_arr)) * 100)
    return mae, rmse, mape


def fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    slope, intercept, _, _, _ = stats.linregress(x, y)
    fitted = intercept + slope * x
    sse = float(np.sum((y - fitted) ** 2))
    r2 = float(1 - sse / np.sum((y - np.mean(y)) ** 2))
    aic, bic = info_criteria(sse, len(y), 2)
    return FitResult("Linear", fitted, (intercept, slope), r2, corr_value(y, fitted), aic, bic, sse)


def predict_linear(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    intercept, slope = fit.params
    return intercept + slope * x_new


def fit_exponential(x: np.ndarray, y: np.ndarray) -> FitResult:
    if np.any(y <= 0):
        raise ValueError("Exponential fit requires positive values.")
    log_y = np.log(y)
    slope, intercept, _, _, _ = stats.linregress(x, log_y)
    fitted = np.exp(intercept + slope * x)
    sse = float(np.sum((y - fitted) ** 2))
    r2 = float(1 - sse / np.sum((y - np.mean(y)) ** 2))
    aic, bic = info_criteria(sse, len(y), 2)
    return FitResult("Exponential", fitted, (intercept, slope), r2, corr_value(y, fitted), aic, bic, sse)


def predict_exponential(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    intercept, slope = fit.params
    return np.exp(intercept + slope * x_new)


def fit_holt(x: np.ndarray, y: np.ndarray) -> FitResult:
    best = None
    grid = np.linspace(0.1, 0.9, 9)
    for alpha in grid:
        for beta in grid:
            level = float(y[0])
            trend = float(y[1] - y[0])
            fitted = [float(y[0])]
            for value in y[1:]:
                fitted.append(level + trend)
                previous_level = level
                level = alpha * float(value) + (1 - alpha) * (level + trend)
                trend = beta * (level - previous_level) + (1 - beta) * trend
            fitted_array = np.asarray(fitted, dtype=float)
            sse_one_step = float(np.sum((y[1:] - fitted_array[1:]) ** 2))
            if best is None or sse_one_step < best[0]:
                best = (sse_one_step, alpha, beta, fitted_array, level, trend)

    assert best is not None
    sse_one_step, alpha, beta, fitted_array, level, trend = best
    sse_full = float(np.sum((y - fitted_array) ** 2))
    r2 = float(1 - sse_full / np.sum((y - np.mean(y)) ** 2))
    aic, bic = info_criteria(sse_one_step, len(y) - 1, 2)
    note = f"alpha={alpha:.1f}, beta={beta:.1f}"
    return FitResult(
        "Holt",
        fitted_array,
        (alpha, beta, level, trend, int(x[-1])),
        r2,
        corr_value(y, fitted_array),
        aic,
        bic,
        sse_full,
        note,
    )


def predict_holt(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    _, _, level, trend, anchor_year = fit.params
    return np.asarray([level + (int(year) - anchor_year) * trend for year in x_new], dtype=float)


def fit_piecewise(x: np.ndarray, y: np.ndarray) -> FitResult:
    best = None
    total_ss = np.sum((y - np.mean(y)) ** 2)
    for breakpoint_idx in range(5, len(x) - 2):
        knot = x[breakpoint_idx]
        hinge = np.maximum(0.0, x - knot)
        design = np.column_stack([np.ones_like(x), x, hinge])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        sse = float(np.sum((y - fitted) ** 2))
        if best is None or sse < best[0]:
            best = (sse, knot, beta, fitted)

    if best is None:
        raise ValueError("Piecewise fit requires at least 8 observations.")

    sse, knot, beta, fitted = best
    r2 = float(1 - sse / total_ss)
    aic, bic = info_criteria(sse, len(y), 3)
    note = f"knot={int(knot)}"
    return FitResult("Piecewise", fitted, (knot, beta), r2, corr_value(y, fitted), aic, bic, sse, note)


def predict_piecewise(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    knot, beta = fit.params
    hinge = np.maximum(0.0, x_new - knot)
    design = np.column_stack([np.ones_like(x_new), x_new, hinge])
    return design @ beta


def logistic_curve(x: np.ndarray, capacity: float, growth: float, midpoint: float) -> np.ndarray:
    return capacity / (1 + np.exp(-growth * (x - midpoint)))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> FitResult:
    lower_bounds = [max(y), 0.0001, x.min() - 10]
    upper_bounds = [max(y) * 20, 2.0, x.max() + 10]
    initial_guess = [max(y) * 1.5, 0.1, float(np.median(x))]
    params, _ = curve_fit(
        logistic_curve,
        x,
        y,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )
    fitted = logistic_curve(x, *params)
    sse = float(np.sum((y - fitted) ** 2))
    r2 = float(1 - sse / np.sum((y - np.mean(y)) ** 2))
    aic, bic = info_criteria(sse, len(y), 3)
    return FitResult(
        "Logistic",
        fitted,
        tuple(float(value) for value in params),
        r2,
        corr_value(y, fitted),
        aic,
        bic,
        sse,
    )


def predict_logistic(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    return logistic_curve(x_new, *fit.params)


def fit_intervention_level_shift(x: np.ndarray, y: np.ndarray, knot: int = 2025) -> FitResult:
    centered = x - x.min()
    post = (x >= knot).astype(float)
    design = np.column_stack([np.ones(len(x)), centered, post])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    sse = float(np.sum((y - fitted) ** 2))
    r2 = float(1 - sse / np.sum((y - np.mean(y)) ** 2))
    aic, bic = info_criteria(sse, len(y), design.shape[1])
    note = f"fixed level shift at {knot}"
    return FitResult(
        "InterventionLevelShift",
        fitted,
        (beta, int(x.min()), knot),
        r2,
        corr_value(y, fitted),
        aic,
        bic,
        sse,
        note,
    )


def predict_intervention_level_shift(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    beta, x_min, knot = fit.params
    centered = x_new - x_min
    post = (x_new >= knot).astype(float)
    design = np.column_stack([np.ones(len(x_new)), centered, post])
    return design @ beta


FITTERS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], FitResult], Callable[[FitResult, np.ndarray], np.ndarray]]] = {
    "Linear": (fit_linear, predict_linear),
    "Exponential": (fit_exponential, predict_exponential),
    "Holt": (fit_holt, predict_holt),
    "Piecewise": (fit_piecewise, predict_piecewise),
    "Logistic": (fit_logistic, predict_logistic),
    "InterventionLevelShift": (fit_intervention_level_shift, predict_intervention_level_shift),
}


def fit_new_baseline(x: np.ndarray, y: np.ndarray) -> FitResult:
    if len(y) < 3:
        raise ValueError("New Baseline requires at least 3 observations.")
    growth_start_index = 0
    growth_end_index = len(y) - 2
    baseline_index = len(y) - 1
    growth_start_value = float(y[growth_start_index])
    growth_end_value = float(y[growth_end_index])
    baseline_value = float(y[baseline_index])
    years_between = int(x[growth_end_index] - x[growth_start_index])
    if growth_start_value <= 0 or growth_end_value <= 0 or years_between <= 0:
        raise ValueError("New Baseline requires positive pre-baseline growth values.")
    hist_cagr = (growth_end_value / growth_start_value) ** (1 / years_between) - 1
    note = f"CAGR {int(x[growth_start_index])}-{int(x[growth_end_index])}={hist_cagr * 100:.2f}%"
    fitted = np.full_like(y, np.nan, dtype=float)
    return FitResult(
        "New Baseline",
        fitted,
        (hist_cagr, baseline_value, int(x[growth_start_index]), int(x[growth_end_index]), int(x[baseline_index])),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        note,
    )


def predict_new_baseline(fit: FitResult, x_new: np.ndarray) -> np.ndarray:
    hist_cagr, baseline_value, _, _, baseline_year = fit.params
    horizons = x_new - baseline_year
    return baseline_value * (1 + hist_cagr) ** horizons


def bootstrap_prediction_interval(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    fit: FitResult,
    future_years: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []

    if model_name == "New Baseline":
        hist_cagr, baseline_value, _, growth_end_year, baseline_year = fit.params
        growth_end_index = int(growth_end_year - x[0])
        if growth_end_index < 1:
            fallback = predict_new_baseline(fit, future_years)
            return fallback.copy(), fallback.copy(), 0

        hist_log_growth = np.log(y[1 : growth_end_index + 1] / y[:growth_end_index])
        log_growth_center = np.log1p(hist_cagr)
        growth_residuals = hist_log_growth - log_growth_center

        for _ in range(iterations):
            value = baseline_value
            boot_path = np.zeros(len(future_years), dtype=float)
            for horizon_index in range(len(future_years)):
                shock = float(rng.choice(growth_residuals))
                value = value * np.exp(log_growth_center + shock)
                boot_path[horizon_index] = value
            draws.append(boot_path)

        stacked = np.vstack(draws)
        lower = np.percentile(stacked, 2.5, axis=0)
        upper = np.percentile(stacked, 97.5, axis=0)
        return lower, upper, len(draws)

    fit_function, predict_function = FITTERS[model_name]
    residuals = np.asarray(y - fit.fitted, dtype=float)
    residuals = residuals - np.mean(residuals)

    for _ in range(iterations):
        sampled = rng.choice(residuals, size=len(residuals), replace=True)
        y_boot = fit.fitted + sampled
        y_boot = np.clip(y_boot, 1.0, None)
        try:
            boot_fit = fit_function(x, y_boot)
            boot_pred = np.asarray(predict_function(boot_fit, future_years), dtype=float)
            if np.all(np.isfinite(boot_pred)):
                draws.append(boot_pred)
        except Exception:
            continue

    if not draws:
        fallback = np.asarray(predict_function(fit, future_years), dtype=float)
        return fallback.copy(), fallback.copy(), 0

    stacked = np.vstack(draws)
    lower = np.percentile(stacked, 2.5, axis=0)
    upper = np.percentile(stacked, 97.5, axis=0)
    return lower, upper, len(draws)


def evaluate_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    future_years: np.ndarray,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> ModelEvaluation:
    if model_name == "New Baseline":
        fit = fit_new_baseline(x, y)
        future_pred = np.asarray(predict_new_baseline(fit, future_years), dtype=float)
        predict_function = predict_new_baseline
        fit_function = fit_new_baseline
    else:
        fit_function, predict_function = FITTERS[model_name]
        fit = fit_function(x, y)
        future_pred = np.asarray(predict_function(fit, future_years), dtype=float)

    rolling_predictions: list[float] = []
    rolling_actuals: list[float] = []
    for index in range(7, len(y)):
        x_train = x[:index]
        y_train = y[:index]
        try:
            rolling_fit = fit_function(x_train, y_train)
            predicted = float(
                np.asarray(predict_function(rolling_fit, np.asarray([x[index]], dtype=float)))[0]
            )
        except Exception:
            continue
        if math.isfinite(predicted):
            rolling_predictions.append(predicted)
            rolling_actuals.append(float(y[index]))

    mae, rmse, mape = metrics(rolling_actuals, rolling_predictions)
    pi_low, pi_high, successes = bootstrap_prediction_interval(
        model_name,
        x,
        y,
        fit,
        future_years,
        bootstrap_iterations,
        bootstrap_seed,
    )
    return ModelEvaluation(
        name=model_name,
        fit=fit,
        future_pred=future_pred,
        pi_low=pi_low,
        pi_high=pi_high,
        rolling_n=len(rolling_actuals),
        mae=mae,
        rmse=rmse,
        mape=mape,
        bootstrap_successes=successes,
        bootstrap_iterations=bootstrap_iterations,
    )


def linear_prediction_interval_2025(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_train = x[:-1]
    y_train = y[:-1]
    slope, intercept, _, _, _ = stats.linregress(x_train, y_train)
    n_obs = len(x_train)
    mean_x = float(np.mean(x_train))
    ss_x = float(np.sum((x_train - mean_x) ** 2))
    residuals = y_train - (slope * x_train + intercept)
    mse = float(np.sum(residuals**2) / (n_obs - 2))
    t_value = float(stats.t.ppf(0.975, n_obs - 2))
    x_target = float(x[-1])
    prediction = intercept + slope * x_target
    half_width = t_value * math.sqrt(mse * (1 + 1 / n_obs + ((x_target - mean_x) ** 2) / ss_x))
    return prediction, prediction - half_width, prediction + half_width


def fit_intervention_evidence(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    centered = x - x.min()
    indicator_2025 = (x == 2025).astype(float)
    base_design = np.column_stack([np.ones(len(x)), centered])
    intervention_design = np.column_stack([np.ones(len(x)), centered, indicator_2025])

    beta_base, *_ = np.linalg.lstsq(base_design, y, rcond=None)
    beta_int, *_ = np.linalg.lstsq(intervention_design, y, rcond=None)
    residual_base = y - base_design @ beta_base
    residual_int = y - intervention_design @ beta_int

    sse_base = float(np.sum(residual_base**2))
    sse_int = float(np.sum(residual_int**2))
    aic_base, _ = info_criteria(sse_base, len(y), base_design.shape[1])
    aic_int, _ = info_criteria(sse_int, len(y), intervention_design.shape[1])

    sigma2 = sse_int / (len(y) - intervention_design.shape[1])
    cov_beta = sigma2 * np.linalg.inv(intervention_design.T @ intervention_design)
    se_dummy = math.sqrt(float(cov_beta[2, 2]))
    t_dummy = float(beta_int[2] / se_dummy)
    p_value = float(2 * (1 - stats.t.cdf(abs(t_dummy), df=len(y) - intervention_design.shape[1])))
    delta_aic = float(aic_base - aic_int)
    return float(beta_int[2]), p_value, delta_aic


def evaluate_all_models(
    x: np.ndarray,
    y: np.ndarray,
    future_years: np.ndarray,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, ModelEvaluation]:
    return {
        model_name: evaluate_model(
            model_name,
            x,
            y,
            future_years,
            bootstrap_iterations,
            bootstrap_seed + index,
        )
        for index, model_name in enumerate(FULL_MODEL_ORDER)
    }


def build_model_metrics(
    observed: pd.DataFrame,
    evaluations: dict[str, ModelEvaluation],
) -> pd.DataFrame:
    observed_2025 = float(
        observed.loc[observed["year"] == 2025, "hand_surgery_publications"].iloc[0]
    )
    rows = []
    for model_name in FULL_MODEL_ORDER:
        evaluation = evaluations[model_name]
        rows.append(
            {
                "model": model_name,
                "R": evaluation.fit.r,
                "R2": evaluation.fit.r2,
                "AIC": evaluation.fit.aic,
                "BIC": evaluation.fit.bic,
                "rolling_origin_n": evaluation.rolling_n,
                "MAE": evaluation.mae,
                "RMSE": evaluation.rmse,
                "MAPE": evaluation.mape,
                "observed_2025": int(observed_2025),
                "projection_2030": float(evaluation.future_pred[-1]),
                "bootstrap_successes": evaluation.bootstrap_successes,
                "bootstrap_iterations": evaluation.bootstrap_iterations,
                "fit_note": evaluation.fit.note,
            }
        )
    return pd.DataFrame(rows)


def build_forecast_paths(evaluations: dict[str, ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for model_name in FULL_MODEL_ORDER:
        evaluation = evaluations[model_name]
        for year, value in zip(FUTURE_YEARS.astype(int), evaluation.future_pred):
            rows.append({"model": model_name, "year": int(year), "point_forecast": float(value)})
    return pd.DataFrame(rows)


def build_prediction_intervals(evaluations: dict[str, ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for model_name in FULL_MODEL_ORDER:
        evaluation = evaluations[model_name]
        for year, lower, upper in zip(
            FUTURE_YEARS.astype(int), evaluation.pi_low, evaluation.pi_high
        ):
            rows.append(
                {
                    "model": model_name,
                    "year": int(year),
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                }
            )
    return pd.DataFrame(rows)


def build_structural_shift(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    prediction, lower, upper = linear_prediction_interval_2025(x, y)
    intervention_coef, p_value, delta_aic = fit_intervention_evidence(x, y)
    observed_2025 = float(y[-1])
    in_range = bool(lower <= observed_2025 <= upper)
    return pd.DataFrame(
        [
            {
                "observed_2025": observed_2025,
                "expected_2025": prediction,
                "expected_lower_95": lower,
                "expected_upper_95": upper,
                "in_range": in_range,
                "intervention_coefficient": intervention_coef,
                "p_value": p_value,
                "delta_aic": delta_aic,
            }
        ]
    )
