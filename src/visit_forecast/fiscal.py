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
    period: Literal["year", "quarter"] = "year",
) -> pd.DataFrame:
    if period == "year":
        history_with_period = add_fiscal_year(hist_df, "ds")
        forecast_with_period = add_fiscal_year(future_df, "ds")
        grouping_columns = ["Fiscal_Year"]
        sorting_columns = ["Fiscal_Year"]
    elif period == "quarter":
        history_with_period = add_fiscal_quarter(hist_df, "ds")
        forecast_with_period = add_fiscal_quarter(future_df, "ds")
        grouping_columns = ["Fiscal_Year", "Fiscal_Quarter", "Fiscal_Period_Label"]
        sorting_columns = ["Fiscal_Year", "Fiscal_Quarter"]
    else:
        raise ValueError("period must be either 'year' or 'quarter'")

    historical_aggregated = (
        history_with_period.groupby(grouping_columns)["y"]
        .sum()
        .reset_index()
        .rename(columns={"y": "Historical_Visits"})
    )
    forecast_aggregated = (
        forecast_with_period.groupby(grouping_columns)[
            ["yhat_original", "yhat_adjusted"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "yhat_original": "Forecast_Original",
                "yhat_adjusted": "Forecast_Adjusted",
            }
        )
    )

    merged_summary = pd.merge(
        historical_aggregated, forecast_aggregated, on=grouping_columns, how="outer"
    ).fillna(0)
    merged_summary = merged_summary.sort_values(sorting_columns).reset_index(drop=True)
    return merged_summary


def fiscal_summary(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_summary_by_period(hist_df, future_df, period="year")


def fiscal_quarter_summary(
    hist_df: pd.DataFrame, future_df: pd.DataFrame
) -> pd.DataFrame:
    return aggregate_summary_by_period(hist_df, future_df, period="quarter")
