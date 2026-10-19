from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def add_fiscal_year(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["Fiscal_Year"] = np.where(d.dt.month >= 4, d.dt.year + 1, d.dt.year)
    return out


def add_fiscal_quarter(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    out = add_fiscal_year(df, date_col=date_col)
    d = pd.to_datetime(out[date_col])
    out["Fiscal_Quarter"] = ((d.dt.month - 4) % 12) // 3 + 1
    out["Fiscal_Period_Label"] = (
        "FY"
        + out["Fiscal_Year"].astype(int).astype(str)
        + " Q"
        + out["Fiscal_Quarter"].astype(int).astype(str)
    )
    return out


def aggregate_summary_by_period(
    hist_df: pd.DataFrame,
    future_df: pd.DataFrame,
    period: Literal["month", "quarter", "year"] = "year",
) -> pd.DataFrame:
    """Roll up the same monthly values and disclose partial fiscal coverage.

    A missing observation remains unavailable. Zero visits are real observations.
    Historical and forecast alternatives are always kept separate.
    """
    if period not in {"month", "quarter", "year"}:
        raise ValueError("period must be either 'month', 'quarter', or 'year'")
    history = hist_df[["ds", "y"]].copy()
    forecast = future_df[["ds", "yhat_original", "yhat_adjusted"]].copy()
    for frame in (history, forecast):
        frame["ds"] = pd.to_datetime(frame["ds"]).dt.to_period("M").dt.to_timestamp()
    history = history.groupby("ds")["y"].sum(min_count=1).rename("Historical_Visits")
    forecast = (
        forecast.groupby("ds")[["yhat_original", "yhat_adjusted"]]
        .sum(min_count=1)
        .rename(
            columns={
                "yhat_original": "Forecast_Original",
                "yhat_adjusted": "Forecast_Adjusted",
            }
        )
    )
    monthly = pd.concat([history, forecast], axis=1).sort_index()
    if monthly.empty:
        return pd.DataFrame(
            columns=[
                "Fiscal_Year",
                "Historical_Visits",
                "Forecast_Original",
                "Forecast_Adjusted",
            ]
        )
    monthly = monthly.reindex(
        pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    )
    monthly.index.name = "ds"
    monthly = add_fiscal_quarter(monthly.reset_index())
    monthly["Fiscal_Month"] = ((monthly.ds.dt.month - 4) % 12) + 1
    monthly["Historical_Months"] = monthly.Historical_Visits.notna().astype(int)
    monthly["Forecast_Months"] = monthly.Forecast_Original.notna().astype(int)
    monthly["Covered_Months"] = (
        monthly.Historical_Visits.notna() | monthly.Forecast_Original.notna()
    ).astype(int)
    if period == "month":
        monthly["Fiscal_Period_Label"] = (
            "FY"
            + monthly.Fiscal_Year.astype(str)
            + " M"
            + monthly.Fiscal_Month.astype(str).str.zfill(2)
            + " ("
            + monthly.ds.dt.strftime("%b %Y")
            + ")"
        )
        keys = ["Fiscal_Year", "Fiscal_Month", "Fiscal_Period_Label"]
    elif period == "quarter":
        keys = ["Fiscal_Year", "Fiscal_Quarter", "Fiscal_Period_Label"]
    else:
        keys = ["Fiscal_Year"]
    values = ["Historical_Visits", "Forecast_Original", "Forecast_Adjusted"]
    counts = ["Historical_Months", "Forecast_Months", "Covered_Months"]
    grouped = monthly.groupby(keys, sort=True)
    out = grouped[values].sum(min_count=1).join(grouped[counts].sum()).reset_index()
    out["Period_Months"] = {"month": 1, "quarter": 3, "year": 12}[period]
    out["Coverage_Status"] = np.where(
        out.Covered_Months.eq(out.Period_Months), "Complete", "Partial"
    )
    return out


def append_fiscal_year_totals(quarter_summary_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "Fiscal_Year",
        "Fiscal_Quarter",
        "Fiscal_Period_Label",
        "Historical_Visits",
        "Forecast_Original",
        "Forecast_Adjusted",
    }
    missing_columns = required_columns.difference(quarter_summary_df.columns)
    if missing_columns:
        missing_columns_list = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"quarter_summary_df is missing required columns: {missing_columns_list}"
        )

    if quarter_summary_df.empty:
        return quarter_summary_df.copy()

    value_columns = [
        name
        for name in [
            "Historical_Visits",
            "Forecast_Original",
            "Forecast_Adjusted",
            "Historical_Months",
            "Forecast_Months",
            "Covered_Months",
        ]
        if name in quarter_summary_df
    ]

    detail_rows = quarter_summary_df.copy()
    detail_rows["Fiscal_Quarter"] = detail_rows["Fiscal_Quarter"].astype("Int64")
    detail_rows["_sort_order"] = detail_rows["Fiscal_Quarter"]

    total_rows = detail_rows.groupby("Fiscal_Year", as_index=False)[value_columns].sum(
        min_count=1
    )
    total_rows["Fiscal_Quarter"] = pd.Series([pd.NA] * len(total_rows), dtype="Int64")
    total_rows["Fiscal_Period_Label"] = (
        "FY" + total_rows["Fiscal_Year"].astype(int).astype(str) + " Total"
    )
    total_rows["_sort_order"] = 5
    if "Period_Months" in detail_rows:
        total_rows["Period_Months"] = 12
        total_rows["Coverage_Status"] = np.where(
            total_rows.Covered_Months.eq(12), "Complete", "Partial"
        )

    combined_summary = pd.concat(
        [detail_rows, total_rows[detail_rows.columns]], ignore_index=True
    )
    combined_summary = combined_summary.sort_values(
        ["Fiscal_Year", "_sort_order"]
    ).reset_index(drop=True)
    return combined_summary.drop(columns="_sort_order")


def fiscal_summary(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_summary_by_period(hist_df, future_df, period="year")


def fiscal_quarter_summary(
    hist_df: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    return aggregate_summary_by_period(hist_df, future_df, period="quarter")
