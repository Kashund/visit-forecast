from __future__ import annotations

import numpy as np
import pandas as pd


def add_prophet_cv_indicators(
    cv_metrics_df: pd.DataFrame,
    interval_width: float | None = None,
) -> pd.DataFrame:
    if cv_metrics_df is None or cv_metrics_df.empty:
        return cv_metrics_df.copy()

    display_df = cv_metrics_df.copy()

    horizon_days = None
    if "horizon" in display_df.columns:
        try:
            horizon_days = pd.to_timedelta(display_df["horizon"]).dt.days
            display_df["Horizon_Days"] = horizon_days
        except Exception:
            horizon_days = None

    if "coverage" in display_df.columns and interval_width is not None:
        coverage_gap = (pd.to_numeric(display_df["coverage"], errors="coerce") - interval_width).abs()
        coverage_gap = coverage_gap.round(6)
        display_df["Coverage_Check"] = np.select(
            [coverage_gap <= 0.03, coverage_gap <= 0.08],
            ["On target", "Watch"],
            default="Off target",
        )

    if {"mae", "rmse"}.issubset(display_df.columns):
        mae_series = pd.to_numeric(display_df["mae"], errors="coerce").abs()
        rmse_series = pd.to_numeric(display_df["rmse"], errors="coerce").abs()
        ratio = rmse_series / mae_series.replace(0, np.nan)
        zero_error_mask = mae_series.eq(0) & rmse_series.eq(0)
        display_df["Error_Shape"] = np.select(
            [zero_error_mask | ratio.le(1.15), ratio.le(1.40)],
            ["Uniform", "Some spikes"],
            default="Big misses",
        )

    if {"mape", "mdape"}.issubset(display_df.columns):
        mape_series = pd.to_numeric(display_df["mape"], errors="coerce").abs()
        mdape_series = pd.to_numeric(display_df["mdape"], errors="coerce").abs()
        ratio = mape_series / mdape_series.replace(0, np.nan)
        zero_percent_error_mask = mape_series.eq(0) & mdape_series.eq(0)
        display_df["Outlier_Check"] = np.select(
            [zero_percent_error_mask | ratio.le(1.15), ratio.le(1.50)],
            ["Aligned", "Some outliers"],
            default="MDAPE more reliable",
        )

    if horizon_days is not None and "rmse" in display_df.columns:
        horizon_sort_df = pd.DataFrame(
            {
                "index": display_df.index,
                "horizon_days": horizon_days,
                "rmse": pd.to_numeric(display_df["rmse"], errors="coerce"),
            }
        ).sort_values("horizon_days")
        rmse_change = horizon_sort_df["rmse"].pct_change()
        horizon_sort_df["Horizon_Trend"] = np.select(
            [
                rmse_change.isna(),
                rmse_change.le(-0.05),
                rmse_change.lt(0.05),
                rmse_change.lt(0.20),
            ],
            ["Baseline", "Improving", "Stable", "Higher error"],
            default="Sharp jump",
        )
        display_df["Horizon_Trend"] = (
            horizon_sort_df.set_index("index")["Horizon_Trend"].reindex(display_df.index)
        )

    preferred_columns = [
        "horizon",
        "Horizon_Days",
        "Horizon_Trend",
        "coverage",
        "Coverage_Check",
        "mae",
        "rmse",
        "Error_Shape",
        "mape",
        "mdape",
        "Outlier_Check",
        "mse",
        "smape",
    ]
    ordered_columns = [
        column_name for column_name in preferred_columns if column_name in display_df.columns
    ]
    ordered_columns.extend(
        [
            column_name
            for column_name in display_df.columns
            if column_name not in ordered_columns
        ]
    )
    return display_df[ordered_columns]
