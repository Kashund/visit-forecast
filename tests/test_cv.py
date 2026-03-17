import pandas as pd

from visit_forecast.cv import add_prophet_cv_indicators


def test_add_prophet_cv_indicators_adds_interpretation_columns():
    cv_metrics = pd.DataFrame(
        {
            "horizon": ["30 days", "60 days", "90 days"],
            "coverage": [0.91, 0.82, 0.70],
            "mae": [10.0, 10.0, 10.0],
            "rmse": [10.5, 12.0, 16.0],
            "mape": [12.0, 18.0, 30.0],
            "mdape": [11.0, 12.0, 15.0],
            "smape": [13.0, 14.0, 17.0],
        }
    )

    result = add_prophet_cv_indicators(cv_metrics, interval_width=0.90)

    assert result["Horizon_Days"].tolist() == [30, 60, 90]
    assert result["Coverage_Check"].tolist() == ["On target", "Watch", "Off target"]
    assert result["Error_Shape"].tolist() == ["Uniform", "Some spikes", "Big misses"]
    assert result["Outlier_Check"].tolist() == [
        "Aligned",
        "Some outliers",
        "MDAPE more reliable",
    ]
    assert result["Horizon_Trend"].tolist() == [
        "Baseline",
        "Higher error",
        "Sharp jump",
    ]
