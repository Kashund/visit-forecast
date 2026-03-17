import importlib.util
from pathlib import Path

import pandas as pd


charts_module_path = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "streamlit"
    / "components"
    / "charts.py"
)
charts_module_spec = importlib.util.spec_from_file_location(
    "streamlit_charts", charts_module_path
)
charts_module = importlib.util.module_from_spec(charts_module_spec)
assert charts_module_spec is not None and charts_module_spec.loader is not None
charts_module_spec.loader.exec_module(charts_module)
forecast_line_chart = charts_module.forecast_line_chart


def test_forecast_line_chart_rescales_prophet_interval_by_selected_confidence():
    future_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "yhat_original": [100.0, 110.0],
            "yhat_adjusted": [100.0, 110.0],
            "yhat_lower": [90.0, 98.0],
            "yhat_upper": [110.0, 122.0],
        }
    )

    fig_95 = forecast_line_chart(
        future_df,
        ci_mode="rmse_95",
        source_interval_width=0.90,
    )
    fig_68 = forecast_line_chart(
        future_df,
        ci_mode="rmse_68",
        source_interval_width=0.90,
    )

    ci_upper_95 = list(fig_95.data[0].y)
    ci_upper_68 = list(fig_68.data[0].y)

    assert fig_95.data[0].name == "CI upper (95%)"
    assert fig_68.data[0].name == "CI upper (68%)"
    assert ci_upper_95 != ci_upper_68
    assert ci_upper_95[0] > ci_upper_68[0]
