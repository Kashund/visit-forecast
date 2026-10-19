# Visit Forecast (Darts, multi-model) + Streamlit App

This repo provides:
- A reusable Python package (`visit_forecast`) for forecasting monthly outpatient visits using **Darts** models (Prophet, Holt-Winters ETS, Theta, SARIMA, seasonal naive baseline, or automatic model selection)
- A local **Streamlit** web app UI
- A notebook workspace for exploration

## Forecast models
The app supports five models plus an automatic mode:
- **Auto (compare models)**: backtests every eligible model with the same rolling protocol and selects the best balanced MAPE and RMSE (MAE and RMSE when MAPE is unavailable, such as all-zero demand). Eligible candidates are scored on an explicitly shared validation start and stride; ineligible candidates retain their failure reasons. Diagnostics show all candidates, including a seasonal naive baseline so you can confirm the winner actually beats a naive repeat of last year.
- **Prophet**: the original model. The only model with Prophet-native intervals, CV, and component plots.
- **Holt-Winters (ETS)** and **Theta**: fast classical models that are strong performers on monthly seasonal data.
- **SARIMA**: seasonal ARIMA (1,1,1)x(1,1,1,12) when at least 30 monthly observations exist (the Darts ARIMA wrapper requires 30 observations).
- **Seasonal naive**: repeats the last 12 months; useful as an honesty check.

Non-Prophet models select conformal calibrated intervals; if explicit bounds cannot be generated, a stored and labeled normal RMSE approximation is used when validation errors are available. They never select Prophet uncertainty. Prophet-native intervals do not exist for these models. ETS and Theta fall back to non-seasonal variants on short histories. The installed Darts ARIMA wrapper requires 30 observations even for non-seasonal ARIMA; shorter histories receive an actionable eligibility error or are excluded from Auto comparison.

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
export. All department forecasts must share a forecast origin and future date grid. A department whose history ends early is reported as a failure, not silently treated as zero. The total is labeled `All (sum of departments)` only when every
requested department succeeds and contributes to every forecast date. Otherwise it is labeled
`Partial total (x/y departments)` and exports include per-date completeness metadata, failure reasons, candidate diagnostics, and run manifests. Expand a department to inspect its forecast, fiscal summary, evaluation, and model comparison.
Programmatic access: `forecast_departments(...)` returns per-department results,
a summary table, and the combined frame.

## Scenario comparison (page 4)
The Scenario Comparison page fits the forecast once, then applies two capacity phase configurations (Scenario A and Scenario B) to the same fitted baseline, so differences between scenarios are purely the phase assumptions rather than refit noise. Phase-only reruns reuse the fitted baseline; changing data or baseline settings invalidates that reuse. Both scenarios support four phases, 100% loss, and additions above 100%. It shows a three-line chart (baseline, A, B), a monthly detail table with A minus Baseline, B minus Baseline, and A minus B deltas, fiscal year totals, and Excel export with Monthly_Comparison, FY_Summary, and Scenario_Definitions sheets. Programmatic access: `apply_capacity_phases_to_future_df(future_df, phases)` recomputes adjustments from `yhat_original`, so it never compounds on prior adjustments.

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
non-negative. Invalid dates, visits, and blank departments produce explicit validation errors; they are never silently dropped. Each forecasted series must contain every month between its
first and last observation. Aggregating All also requires complete department/month contributions across the selected observed window, so another department cannot hide a missing submission. Select a common complete window or explicitly exclude inactive departments; missing observations are never automatically zero-filled.

## Install
From repo root:
```bash
pip install .
```

For a reproducible development or deployment environment:

```bash
uv sync --locked --extra dev --no-editable
```

The non-editable install is intentional for Python 3.13 compatibility and
matches CI. It verifies the built package instead of relying on an
editable-install `.pth` file. After changing package source, refresh the installed copy with `uv sync --locked --extra dev --no-editable --reinstall-package visit-forecast`; a plain sync may reuse the previous non-editable build.

## Run the Streamlit app
```bash
uv run --no-editable streamlit run apps/streamlit/Home.py
```

## Quick launch shortcut (start + open browser)
```bash
uv run --no-editable python scripts/launch_streamlit_app.py
```

This helper starts Streamlit and automatically opens `http://127.0.0.1:8501` in your default web browser.
Because it binds to localhost, it also enables cross-session run-history
persistence. Direct or hosted Streamlit launches remain session-only by default.

If you see `No module named streamlit`, use the same Python environment where this project is installed:
```bash
uv sync --locked --no-editable
uv run --no-editable python scripts/launch_streamlit_app.py
```

## Run from the CLI

```bash
uv run --no-editable python scripts/run_forecast_cli.py \
  --use_synthetic \
  --model seasonal_naive \
  --months 3
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

## Fiscal Summary views

The Fiscal Summary section has a single **Fiscal Month / Fiscal Quarter / Fiscal Year** selector that updates both its table and chart. Every view keeps **Historical Visits**, **Original Forecast**, and **Adjusted Forecast** separate, including periods containing a mixture of actuals and forecasts. Changing views reuses the existing run.

The calendar is Apr–Mar: April is fiscal month 1 and March month 12. For example, April 2024 is `FY2025 M01 (Apr 2024)`, and January 2025 is `FY2025 M10 (Jan 2025)`. Quarter/year values sum the same monthly values at full precision. Quarter subtotal rows are excluded from chart totals.

`Historical_Months`, `Forecast_Months`, `Covered_Months`, `Period_Months`, and `Coverage_Status` identify partial periods. A missing value is unavailable, not a genuine zero. The CSV download matches the selected view. Validation error metrics remain month-based regardless of fiscal view.

Python: `aggregate_summary_by_period(history, future, period="month" | "quarter" | "year")`, imported from `visit_forecast`.

## Public APIs and workflow exports

```python
from visit_forecast import (
    forecast_visits, forecast_departments,
    apply_capacity_phases_to_future_df, inspect_data,
)
```

All forecast pages accept CSV/XLSX, pasted CSV, or synthetic input. Department and Scenario pages can also reuse a dataset from another run or Data Check in the same browser session. A warning identifies controls changed since the displayed result. Results are updated only when the corresponding Run button is pressed.

CLI workflows keep the original forecast command and add these options:

```bash
uv run --no-editable python scripts/run_forecast_cli.py \
  --filename data/sample/synthetic_visits.csv --model seasonal_naive \
  --months 3 --uncertainty conformal --fiscal-view month --output-dir /tmp/visit-output

uv run --no-editable python scripts/run_forecast_cli.py \
  --use_synthetic --workflow departments --model auto \
  --candidates ets theta seasonal_naive --exclude Neurology --output-dir /tmp/departments

uv run --no-editable python scripts/run_forecast_cli.py \
  --use_synthetic --workflow scenarios --model seasonal_naive --months 3 \
  --scenario-a-json '[{"mode":"add","percent":125,"start_month":1,"end_month":3}]' \
  --scenario-b-json '[{"mode":"loss","percent":100,"start_month":2,"end_month":3}]' \
  --output-dir /tmp/scenarios
```

Use `--calendar-covariates`, repeated `--exclude`, `--tuning manual|auto`, and `--uncertainty prophet|conformal|auto` as appropriate. Calendar covariates count Monday–Friday, without regional holidays or clinic closures. Empty or unknown Auto candidate lists are rejected, and a failed comparison never reintroduces an excluded model.

Output directories contain full-precision CSV tables, `forecast.xlsx`, and `run_manifest.json`. Manifests identify the input fingerprint, code/dependency versions, requested/effective settings, evaluation period, warnings, and run ID. They do not contain the raw input dataset. Run Forecast also offers a manifest download, and department/scenario workbooks include a manifest sheet. Repeated identical main forecasts can reuse up to three results within the current session, keyed by data, settings, and engine source. A new run ID still records each deliberate run. Progress identifies validation, selection, fitting, and evaluation stages. Session history records all three workflows; opt-in local persistence uses atomic writes and reports save failures. On shared machines, use separate accounts/history directories; localhost alone does not isolate users.

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

Notebook tooling is optional: `uv run --no-editable --with jupyterlab --with ipykernel jupyter lab notebooks`. Select the project Python environment as the kernel. Dependencies for the application remain pinned in `uv.lock`; optional notebook tooling is not part of that lock.


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

When history contains at least `max(48, 2 * forecast_months + 26)` observations, the final forecast horizon is reserved before model/parameter selection. Reported point metrics then score that final block; final live forecasts are fitted on all history. Reserved-block interval coverage is reported separately from development interval diagnostics, using calibration outcomes preceding that block. One reserved origin is limited evidence, not a coverage guarantee.

Shorter histories retain development backtests, explicitly labeled as lacking independent final evaluation. The UI reports unavailable MAPE rather than assigning a quality rating when all actuals are zero. Selection scores and final evaluation can differ. Negative model point predictions are clipped at zero visits; manifests disclose this policy.
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
forecast cutoffs and evaluated on newer cutoffs; calibration outcomes unavailable at the first evaluation origin are purged, including overlapping target dates; if there is too
little history for held-out evaluation, coverage is reported as unavailable
rather than resubstitution coverage. If the selected method cannot produce
explicit bounds, the app stores a normal RMSE approximation at the requested nominal level and records `rmse_approx` as the source. These approximate bounds are exported as well as plotted. Empirical quantile intervals assume stable errors and do not guarantee nominal coverage for dependent monthly data. It also overlays MAE and RMSE error bands around the adjusted
forecast.

## Run history
The Streamlit app keeps a log of the last 20 forecast, department, and scenario runs in the current browser
session, including settings and validation metadata. Local disk persistence is
opt-in as described above. You can clear history from the sidebar.

## Deployment checklist

Before a production deployment:

1. Install uv 0.9.6 and run
   `uv sync --locked --extra dev --no-editable`.
2. Run
   `uv run --no-editable python -c "import visit_forecast"`.
3. Run `uv run --no-editable python -m pytest -q`.
4. Run
   `uv run --no-editable python -m compileall -q src apps scripts`.
5. Run
   `uv run --no-editable python scripts/run_forecast_cli.py --help`.
6. Confirm the deployment starts
   `uv run --no-editable streamlit run apps/streamlit/Home.py`.
7. Keep `VISIT_FORECAST_PERSIST_HISTORY` unset for multi-user hosting.
8. Store secrets in the hosting platform, never in
   `.streamlit/secrets.toml`.

GitHub Actions repeats the non-editable install, package import, CLI smoke,
tests, compilation, and wheel construction for pushes to `main` and pull
requests.


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
