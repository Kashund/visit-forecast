"""Run with Python -I outside the checkout after installing the built wheel."""

from pathlib import Path
import tempfile
import pandas as pd
import visit_forecast
from visit_forecast import (
    forecast_visits,
    forecast_departments,
    apply_capacity_phases_to_future_df,
)
from visit_forecast.exports import forecast_tables, write_workflow_export

assert "site-packages" in str(Path(visit_forecast.__file__).resolve()), (
    visit_forecast.__file__
)
result = forecast_visits(
    use_synthetic=True,
    timeframe_start="2020-01-01",
    timeframe_end="2022-12-01",
    forecast_periods=3,
    model_choice="seasonal_naive",
    uncertainty_method="conformal",
)
assert len(result.future_forecast_df) == 3
assert result.future_forecast_df[["yhat_lower", "yhat_upper"]].notna().all().all()
with tempfile.TemporaryDirectory() as directory:
    write_workflow_export(
        directory, forecast_tables(result, "month"), result.run_manifest
    )
    frame = pd.read_csv(Path(directory) / "future_forecast.csv")
    assert len(frame) == 3
    assert (Path(directory) / "run_manifest.json").is_file()
print(
    "Installed wheel: imports, synthetic forecast, fiscal monthly export, and manifest passed."
)
