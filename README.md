# Visit Forecast (Darts Prophet) + Streamlit App

This repo provides:
- A reusable Python package (`visit_forecast`) for forecasting monthly outpatient visits using **Darts Prophet**
- A local **Streamlit** web app UI
- A notebook workspace for exploration

## Repo layout
- `src/visit_forecast/` core library (import from notebooks + Streamlit)
- `apps/streamlit/` Streamlit UI
- `data/` place your Excel/CSV in `data/raw/`

## Data requirements
Input must contain columns:
- `Date`
- `Visits`
- `Department`

## Install (editable)
From repo root:
```bash
pip install -e .
```

## Run the Streamlit app
```bash
streamlit run apps/streamlit/Home.py
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
- Upload Excel/CSV
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
- Disabled phases (`enabled=False`) are ignored and do not affect output.
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
The app reports:
- **MAPE** (percent error; easiest to compare across departments)
- **MAE** (average absolute error, in visit units)
- **RMSE** (like MAE but penalizes large misses more)

The UI includes a simple heuristic status indicator (Optimal/Good/Fair/Poor).
For MAE/RMSE, the status is based on error relative to the mean actual volume.

## Confidence & error bands
The forecast chart shows a confidence band. If the model output does not include explicit lower/upper bounds, the app approximates a confidence interval using RMSE (±1.96·RMSE for ~95%). It also overlays MAE and RMSE error bands around the adjusted forecast.

## Run history
The Streamlit app keeps a log of the last 20 runs (per browser session) including inputs and validation metrics. You can clear this history from the sidebar.


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
