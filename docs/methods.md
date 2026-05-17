# Methods

This repository reproduces the predictive forecasting algorithm used in the manuscript. It does not argue for any particular model choice; the comparison metrics are emitted to `model_metrics.csv` and a reader can apply their own selection criterion.

## Input series

Annual count of PubMed records classified in-scope for hand surgery and assigned a non-empty country attribution. Years 2010–2025 (16 observations).

## Candidate models

Seven time-series models are fit on the historical years and used to project 2026–2030:

| Model | Description |
| --- | --- |
| Linear | OLS regression on year. |
| Exponential | OLS on `log(y)` against year. |
| Holt | Double exponential smoothing with grid-searched `α`, `β` ∈ {0.1, …, 0.9}; minimizes one-step-ahead SSE. |
| Piecewise | Linear with a single knot, optimal knot selected by SSE over candidate positions. |
| Logistic | Three-parameter logistic curve fit via non-linear least squares (`scipy.optimize.curve_fit`). |
| New Baseline | Compound annual growth rate (CAGR) from the first to second-to-last year, anchored at the latest observation. |
| InterventionLevelShift | Linear trend plus an indicator for years ≥ 2025 (fixed knot). |

## Fit statistics

For each model, the algorithm reports Pearson `R`, `R²`, `AIC`, and `BIC` computed from in-sample residuals. The information criteria use the SSE form:

```
AIC = n · log(SSE / n) + 2k
BIC = n · log(SSE / n) + k · log(n)
```

## Rolling-origin validation

For each year `t` in 2017–2025, the model is refit on years `2010..t-1` and used to predict year `t`. The resulting one-step-ahead errors are summarized as `MAE`, `RMSE`, and `MAPE`.

## Bootstrap prediction intervals

For all models except `New Baseline`, 95% intervals are constructed by sampling fitted residuals with replacement, refitting the model on each bootstrap series, projecting forward, and taking the 2.5%/97.5% percentiles across `--bootstrap-iterations` draws. `New Baseline` resamples log-growth residuals from the pre-baseline window.

Bootstraps are deterministic given the `--seed` argument. Each model uses `seed + model_index` (its position in `FULL_MODEL_ORDER`).

## Structural-shift summary

A linear model is fit on 2010–2024 and used to construct a 95% prediction interval for 2025. The observed 2025 count is compared to that interval. As supporting evidence, a linear model with a 2025 indicator dummy is fit on the full series, and the dummy's coefficient, p-value, and ΔAIC versus the no-dummy model are reported.
