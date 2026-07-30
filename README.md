# Visit Forecast (Darts, multi-model) + Streamlit App

This repo provides:
- A reusable Python package (`visit_forecast`) for forecasting monthly outpatient visits using **Darts** models (Prophet, Holt-Winters ETS, Theta, SARIMA, seasonal naive baseline, or automatic model selection)
- A local **Streamlit** web app UI
- A notebook workspace for exploration

## Forecast models
The app supports five models plus an automatic mode:
- **Auto (compare models)**: backtests every eligible model with the same rolling protocol and selects the best balanced MAPE and RMSE. Diagnostics show all candidates, including a seasonal naive baseline so you can confirm the winner actually beats a naive repeat of last year.
- **Prophet**: the original model. The only model with Prophet-native intervals, CV, and component plots.
- **Holt-Winters (ETS)** and **Theta**: fast classical models that are strong performers on monthly seasonal data.
- **SARIMA**: seasonal ARIMA (1,1,1)x(1,1,1,12) when at least two full seasons of history exist.
- **Seasonal naive**: repeats the last 12 months; useful as an honesty check.

Non-Prophet models always use conformal calibrated intervals (Prophet-native intervals do not exist for them). Models needing two seasonal cycles automatically fall back to non-seasonal variants on short histories.

CLI: `--model auto|prophet|ets|theta|arima|seasonal_naive`
Python: `forecast_visits(..., model_choice="auto")`

## Calendar covariates
Enable **Calendar covariates (business days per month)** in the sidebar (or `use_calendar_covariates=True` in Python) to add business days per month as a regressor. Months with more clinic days generate more visits, so this can explain month-to-month variation that pure seasonality misses. Prophet and SARIMA use the covariate in fitting, backtesting, and auto model selection; ETS, Theta, and seasonal naive ignore it (a note in the run explains this). When Prophet uses the covariate, its effect appears as a component plot.

## Prophet component plots
The Prophet plots section now renders trend, yearly seasonality (average effect by month), and the calendar covariate effect directly from Prophet's native output. No fitted model object needs to be cached.

## Persistent run history
Run history is session-only by default so one hosted user cannot see another
user's forecast metadata. The localhost quick launcher opts into persistence and
saves the last 20 runs to `~/.visit_forecast/run_history.json`.

To opt in explicitly, set `VISIT_FORECAST_PERSIST_HISTORY=1`. Override the
location with `VISIT_FORECAST_HISTORY_DIR`. Only enable disk persistence when
the app is single-user or each user has isolated storage.

## Department forecasts (page 3)
The Department Forecasts page fits each department independently. With Auto
selected, every department gets its own model comparison, since departments
often have different trend and seasonality profiles. An "Include Prophet in
auto comparison" toggle (off by default) keeps multi-department runs fast; the
classical models are usually competitive on monthly data. Results include a
per-department model and accuracy summary, a combined chart, and Excel/CSV
export. The total is labeled `All (sum of departments)` only when every
requested department succeeds. Otherwise it is labeled
`Partial total (x/y departments)` and exports include completeness metadata.
Programmatic access: `forecast_departments(...)` returns per-department results,
a summary table, and the combined frame.

## Scenario comparison (page 4)
The Scenario Comparison page fits the forecast once, then applies two capacity phase configurations (Scenario A and Scenario B) to the same fitted baseline, so differences between scenarios are purely the phase assumptions rather than refit noise. It shows a three-line chart (baseline, A, B), a monthly detail table with A minus Baseline, B minus Baseline, and A minus B deltas, fiscal year totals, and Excel export with Monthly_Comparison, FY_Summary, and Scenario_Definitions sheets. Programmatic access: `apply_capacity_phases_to_future_df(future_df, phases)` recomputes adjustments from `yhat_original`, so it never compounds on prior adjustments.

## Dependencies
Supported dependency ranges are constrained in `pyproject.toml`. Darts uses the
current `darts` package name rather than the deprecated `u8darts` compatibility
package. `uv.lock` pins the complete dependency graph used by CI and release
verification.

## Repo layout
- `src/visit_forecast/` core library (import from notebooks + Streamlit)
- `apps/streamlit/` Streamlit UI
- `data/` place your Excel/CSV in `data/raw/`

## Data requirements
Input must contain columns:
- `Date`
- `Visits`
- `Department`

Use `.xlsx` or `.csv` files. Dates are normalized to month start, duplicate
rows for the same department/month are summed, Visits must be finite and
non-negative, and each forecasted series must contain every month between its
first and last observation.

## Install (editable)
From repo root:
```bash
pip install -e .
```

For a reproducible development or deployment environment:

```bash
uv sync --locked --extra dev
```

## Run the Streamlit app
```bash
streamlit run apps/streamlit/Home.py
```

## Quick launch shortcut (start + open browser)
```bash
python scripts/launch_streamlit_app.py
```

This helper starts Streamlit and automatically opens `http://127.0.0.1:8501` in your default web browser.
Because it binds to localhost, it also enables cross-session run-history
persistence. Direct or hosted Streamlit launches remain session-only by default.

If you see `No module named streamlit`, use the same Python environment where this project is installed:
```bash
python -m pip install -e .
python scripts/launch_streamlit_app.py
```

## Use from Python / notebooks
```python
from visit_forecast import forecast_visits

result = forecast_visits(
    filename="data/raw/intake.xlsx",
    use_synthetic=False,
    timeframe_start="2020-04-01",
    timeframe_end="2025-03-01",
    forecast_periods=12,
    department="All",
    adjustment_percent=10,
)
print(result.fiscal_summary)
```

## Notes
- Dates are normalized to month-start.
- Fiscal year is Apr → Mar (Apr–Dec -> FY+1, Jan–Mar -> FY same year).


## Streamlit data source options
The app supports:

- In synthetic mode, department options default to Cardiology / Oncology / Neurology.
- Upload `.xlsx` or `.csv` (legacy `.xls` is not supported)
- Paste CSV values directly in the sidebar (headers required: `Date, Visits, Department`)
- Use a built-in synthetic sample (no file needed)

## Sample data
Sample CSVs are included:
- `data/sample/synthetic_visits.csv`
- `data/sample/synthetic_visits_small.csv`

Regenerate them anytime:
```bash
python scripts/build_sample_data.py
```

## Notebook quickstart
Open:
- `notebooks/00_Quickstart.ipynb`


### Pandas note
If you're on very new pandas versions, month-start normalization uses `to_timestamp(how='start')` for compatibility.

- Backtesting automatically adapts when you choose long forecast horizons.

## Capacity adjustment
Capacity adjustments support both the original single-phase inputs and new multi-phase configuration.

### Single-phase (backward compatible)
You can still provide:
- `adjustment_mode` (`loss` or `add`)
- `adjustment_percent`
- `adjustment_start_month` and `adjustment_end_month`

If `capacity_phases` is not provided, these legacy inputs are converted into one phase internally.

### Multi-phase configuration
Provide `capacity_phases` as a list of up to four phase objects:
- `enabled` (bool)
- `mode` (`loss` or `add`)
- `percent` (non-negative)
- loss percentages must be between 0 and 100; additions may exceed 100
- `start_month` and `end_month` (1-indexed and within the forecast horizon)

Example configuration:
- Months 1–3 add 10%
- Months 5–7 loss 15%

```python
capacity_phases = [
    {"enabled": True, "mode": "add", "percent": 10, "start_month": 1, "end_month": 3},
    {"enabled": True, "mode": "loss", "percent": 15, "start_month": 5, "end_month": 7},
]
```

### Toggle and overlap behavior
- Disabled phases (`enabled=False`) are ignored, are not validated, and do not affect output.
- Phase windows extending past the forecast horizon are not an error; the phase applies to whatever part of the window falls inside the horizon. A phase entirely beyond the horizon is a no-op.
- Zero-percent phases are no-ops and do not mark months as adjusted.
- In the Streamlit app, if you shorten the forecast horizon after configuring phases, phase month ranges are clamped to the new horizon and a sidebar warning tells you which enabled phases were affected.
- Disjoint phases apply only to their own month ranges.
- Overlapping enabled phases are applied sequentially in the list order (multiplicative factors).
- The output includes `applied_phase_ids` so you can see exactly which phase(s) affected each forecast month.

### CLI examples
JSON payload:
```bash
python scripts/run_forecast_cli.py \
  --use_synthetic \
  --capacity_phases_json '[{"enabled": true, "mode": "add", "percent": 10, "start_month": 1, "end_month": 3}, {"enabled": true, "mode": "loss", "percent": 15, "start_month": 5, "end_month": 7}]'
```

Repeated `--phase` arguments:
```bash
python scripts/run_forecast_cli.py \
  --use_synthetic \
  --phase add:10:1:3 \
  --phase loss:15:5:7
```

The future forecast table includes:
- `forecast_month` (1-indexed)
- `adj_applied` (True/False)
- `applied_phase_ids` (comma-separated phase order)

## Forecast validation (backtesting)
Backtesting scores every month of each rolling forecast fold (not just the final month), so MAPE, MAE, and RMSE reflect the average error per forecasted month as described below.

The app reports:
- **MAPE** (percent error; easiest to compare across departments)
- **MAE** (average absolute error, in visit units)
- **RMSE** (like MAE but penalizes large misses more)

The UI includes a simple heuristic status indicator (Optimal/Good/Fair/Poor).
For MAE/RMSE, the status is based on error relative to the mean actual volume.

## Confidence & error bands
The forecast chart shows a confidence band. Prophet-native intervals support
calendar covariates. Conformal intervals are calibrated on older rolling
forecast cutoffs and evaluated on held-out newer cutoffs; if there is too
little history for held-out evaluation, coverage is reported as unavailable
rather than resubstitution coverage. If the selected method cannot produce
explicit bounds, the app falls back to an RMSE approximation and records the
source used. It also overlays MAE and RMSE error bands around the adjusted
forecast.

## Run history
The Streamlit app keeps a log of the last 20 runs in the current browser
session, including inputs and validation metrics. Local disk persistence is
opt-in as described above. You can clear history from the sidebar.

## Deployment checklist

Before a production deployment:

1. Install uv 0.9.6 and run `uv sync --locked --extra dev`.
2. Run `uv run python -m pytest -q`.
3. Run `uv run python -m compileall -q src apps scripts`.
4. Keep `VISIT_FORECAST_PERSIST_HISTORY` unset for multi-user hosting.
5. Store secrets in the hosting platform, never in `.streamlit/secrets.toml`.
6. Confirm the deployment starts `streamlit run apps/streamlit/Home.py`.

GitHub Actions repeats installation, tests, compilation, and wheel construction
for pushes to `main` and pull requests.


## Metrics & interpretation

### Backtesting metrics (shown at top)
- **MAPE**: average absolute % error. Lower is better. Heuristic: <10% excellent; 10–20% good; 20–50% fair; >50% poor.
- **MAE**: average absolute error in visit units. Interpret relative to typical monthly volume.
- **RMSE**: like MAE but penalizes large misses more; RMSE >> MAE indicates occasional big errors.

### Prophet cross-validation metrics (CV tab)
- **mae/rmse/mse**: error magnitude (visit units)
- **mape/mdape/smape**: percent-based error (lower better)
- **coverage**: fraction of actuals inside the interval; should be close to your interval width (e.g., ~0.90 for 90% intervals)

### Chart guide
- **Forecast chart**: Original vs Adjusted. Shaded region shows months where capacity adjustment applies.
- **Confidence interval**: Prophet bounds when available (responds to interval_width); otherwise RMSE-based approximation.
- **Error bands**: ±MAE and ±RMSE around adjusted forecast to show typical error scale.
- **CV plots**: error vs horizon; expect error to rise with horizon.
