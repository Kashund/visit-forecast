from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def to_month_start(dt: pd.Series) -> pd.Series:
    """Normalize datetimes to *month start* timestamps.

    We convert to monthly Periods and then back to timestamps at the start of the period.
    Note: pandas does not accept 'MS' in PeriodArray.to_timestamp(), so we use how='start'.
    """
    return pd.to_datetime(dt).dt.to_period("M").dt.to_timestamp(how="start")


def generate_synthetic_data(
    timeframe_start: str,
    timeframe_end: str,
    departments: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    departments = departments or ["Cardiology", "Oncology", "Neurology"]
    dates = pd.date_range(start=timeframe_start, end=timeframe_end, freq="MS")

    rng = np.random.default_rng(seed)
    df_list = []

    dept_offsets = {"Cardiology": 5, "Oncology": 0, "Neurology": -5}

    for dept in departments:
        noise = rng.normal(0, 5, len(dates))
        offset = dept_offsets.get(dept, 0)
        visits = (
            100
            + np.arange(len(dates)) * 2
            + 10 * np.sin(np.linspace(0, 3 * np.pi, len(dates)))
            + noise
            + offset
        )
        df_list.append(pd.DataFrame({"Date": dates, "Visits": visits, "Department": dept}))

    return pd.concat(df_list, ignore_index=True)


def load_data(use_synthetic: bool, filename: Optional[str], start: str, end: str) -> pd.DataFrame:
    if use_synthetic:
        return generate_synthetic_data(start, end)
    if not filename:
        raise ValueError("filename must be provided if use_synthetic=False")
    return pd.read_excel(filename)


def prepare_dataframe(
    df_raw: pd.DataFrame,
    timeframe_start: str,
    timeframe_end: str,
    department: str,
    exclude_departments: Optional[List[str]],
) -> Tuple[pd.DataFrame, str]:
    validate_columns(df_raw, ["Date", "Visits", "Department"])

    df = df_raw.copy()
    df["Date"] = to_month_start(df["Date"])
    df["Visits"] = pd.to_numeric(df["Visits"], errors="coerce")
    df["Department"] = df["Department"].astype("string").str.strip()
    df = df.dropna(subset=["Date", "Visits", "Department"])
    df = df[df["Department"].ne("")]
    df = df[np.isfinite(df["Visits"])]
    negative_visit_count = int(df["Visits"].lt(0).sum())
    if negative_visit_count:
        raise ValueError(
            "Visits must be non-negative; "
            f"found {negative_visit_count} negative value(s)."
        )

    start_dt = pd.to_datetime(timeframe_start).to_period("M").to_timestamp()
    end_dt = pd.to_datetime(timeframe_end).to_period("M").to_timestamp()
    if start_dt > end_dt:
        raise ValueError("timeframe_start must be on or before timeframe_end.")
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]

    if exclude_departments:
        normalized_exclusions = {
            str(department_name).strip()
            for department_name in exclude_departments
        }
        df = df[~df["Department"].isin(normalized_exclusions)]

    if df.empty:
        raise ValueError("No data remains after timeframe/exclusion filtering.")

    normalized_department = str(department).strip()
    if normalized_department != "All":
        df_f = df[df["Department"] == normalized_department].copy()
        if df_f.empty:
            raise ValueError(
                f"No data found for department='{normalized_department}' after filtering."
            )
        # Multiple source rows can normalize to the same month. Aggregate them
        # before constructing a Darts series so each month has one observation.
        df_f = df_f.groupby("Date", as_index=False)["Visits"].sum()
        df_f["Department"] = normalized_department
        dept_info = normalized_department
    else:
        included = sorted(df["Department"].astype(str).unique())
        dept_info = ", ".join(included)
        df_f = df.groupby("Date", as_index=False)["Visits"].sum()
        df_f["Department"] = "All"

    df_f = df_f.rename(columns={"Date": "ds", "Visits": "y"})
    df_f = df_f.sort_values("ds").reset_index(drop=True)

    expected_months = pd.date_range(
        df_f["ds"].min(), df_f["ds"].max(), freq="MS"
    )
    missing_months = expected_months.difference(pd.DatetimeIndex(df_f["ds"]))
    if not missing_months.empty:
        missing_preview = ", ".join(
            timestamp.strftime("%Y-%m")
            for timestamp in missing_months[:6]
        )
        suffix = "…" if len(missing_months) > 6 else ""
        raise ValueError(
            "Monthly history contains missing observations: "
            f"{missing_preview}{suffix}. Fill or remove the gap before forecasting."
        )

    return df_f, dept_info
