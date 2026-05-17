# Hand Surgery Publication Forecast

Reproducibility package for the manuscript's forecasting analysis of global hand-surgery publication output. Given annual publication counts, this repository reproduces:

- Model-comparison metrics (`R`, `R²`, `AIC`, `BIC`, rolling-origin `MAE`/`RMSE`/`MAPE`) for eight candidate forecast models.
- Year-by-year point forecasts for 2026–2030.
- 95% residual-bootstrap prediction intervals.
- A structural-shift summary comparing the observed 2025 count to the pre-2025 linear trend.

## Install

Requires Python ≥ 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python scripts/run_forecast.py \
    --input data/observed_global_counts.csv \
    --output-dir out/ \
    --bootstrap-iterations 400 \
    --seed 42
```

Typical runtime is under 30 seconds on a current laptop. The bootstrap is deterministic for a given seed and pinned dependency versions.

## Outputs

Written to `--output-dir`:

| File | Shape | Contents |
| --- | --- | --- |
| `model_metrics.csv` | one row per model | `model`, fit statistics (`R`, `R2`, `AIC`, `BIC`), rolling-origin metrics (`MAE`, `RMSE`, `MAPE`), observed 2025 count, `projection_2030`, bootstrap counts, fit notes |
| `forecast_paths.csv` | long (model × year) | `model`, `year`, `point_forecast` for years 2026–2030 |
| `prediction_intervals.csv` | long (model × year) | `model`, `year`, `lower_95`, `upper_95` |
| `structural_shift.csv` | single row | observed 2025 vs. pre-2025 linear prediction interval, intervention-dummy coefficient, p-value, ΔAIC |

## Data

`data/observed_global_counts.csv` contains aggregate yearly counts of hand-surgery PubMed records included after rules-based classification and country attribution. Raw article-level data and reviewer-classified workbooks are not shared (see `docs/data_statement.md`). The `pubmed_retrieved_date` column records the snapshot date.

## Citation

Citation metadata is provided in `CITATION.cff`. The Zenodo archive of this repository is available at [doi:10.5281/zenodo.20257890](https://doi.org/10.5281/zenodo.20257890).

## License

MIT (see `LICENSE`).
