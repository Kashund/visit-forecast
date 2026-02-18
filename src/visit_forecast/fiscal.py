from __future__ import annotations

import numpy as np
import pandas as pd


def add_fiscal_year(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["Fiscal_Year"] = np.where(d.dt.month >= 4, d.dt.year + 1, d.dt.year)
    return out


def fiscal_summary(hist_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    hist = (
        hist_df.groupby("Fiscal_Year")["y"]
        .sum()
        .reset_index()
        .rename(columns={"y": "Historical_Visits"})
    )
    fc = (
        future_df.groupby("Fiscal_Year")[["yhat_original", "yhat_adjusted"]]
        .sum()
        .reset_index()
        .rename(columns={"yhat_original": "Forecast_Original", "yhat_adjusted": "Forecast_Adjusted"})
    )
    out = pd.merge(hist, fc, on="Fiscal_Year", how="outer").fillna(0)
    out = out.sort_values("Fiscal_Year").reset_index(drop=True)
    return out
