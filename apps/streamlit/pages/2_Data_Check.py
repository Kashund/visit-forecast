from __future__ import annotations

from bootstrap import ensure_project_paths
import numpy as np
import streamlit as st
import pandas as pd

ensure_project_paths()

from visit_forecast.io import to_month_start

st.set_page_config(page_title="Data Check", page_icon="✅", layout="wide")
st.title("✅ Data Check")

uploaded = st.file_uploader(
    "Upload Excel (.xlsx) or CSV",
    type=["xlsx", "csv"],
)


def read_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


if uploaded is None:
    st.info("Upload a dataset to validate required columns and detect missing months.")
    st.stop()

try:
    df = read_file(uploaded)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {type(exc).__name__}: {exc}")
    st.stop()

st.dataframe(df.head(30), width="stretch")

required = {"Date", "Visits", "Department"}
missing = required - set(df.columns)

if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

st.success("Required columns are present.")

df2 = df.copy()
df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
df2["Visits"] = pd.to_numeric(df2["Visits"], errors="coerce")
df2["Department"] = df2["Department"].astype("string").str.strip()

st.write("Null counts")
st.dataframe(df2[["Date", "Visits", "Department"]].isna().sum().to_frame("nulls"), width="stretch")

invalid_date_count = int(df2["Date"].isna().sum())
invalid_visit_count = int(
    df2["Visits"].isna().sum()
    + (
        df2["Visits"].notna()
        & ~np.isfinite(df2["Visits"].fillna(0))
    ).sum()
)
negative_visit_count = int(df2["Visits"].lt(0).sum())
blank_department_count = int(
    (df2["Department"].isna() | df2["Department"].eq("")).sum()
)

metric_columns = st.columns(4)
metric_columns[0].metric("Invalid dates", invalid_date_count)
metric_columns[1].metric("Invalid visits", invalid_visit_count)
metric_columns[2].metric("Negative visits", negative_visit_count)
metric_columns[3].metric("Blank departments", blank_department_count)

quality_error_counts = [
    invalid_date_count,
    invalid_visit_count,
    negative_visit_count,
    blank_department_count,
]
if any(quality_error_counts):
    st.error(
        "This dataset is not forecast-ready. Correct the invalid values above "
        "before running a forecast."
    )

df2 = df2.dropna(subset=["Date"])
if df2.empty:
    st.error("No valid dates remain after parsing the Date column.")
    st.stop()

df2["Date"] = to_month_start(df2["Date"])
duplicate_count = int(
    df2.dropna(subset=["Department"]).duplicated(
        subset=["Department", "Date"],
        keep=False,
    ).sum()
)
if duplicate_count:
    st.warning(
        f"{duplicate_count} rows share a department/month. "
        "Forecasting will sum those rows into one monthly observation."
    )

min_d, max_d = df2["Date"].min(), df2["Date"].max()
all_months = pd.date_range(min_d, max_d, freq="MS")

st.subheader("Missing months by department")
out = []
valid_department_rows = df2[
    df2["Department"].notna() & df2["Department"].ne("")
]
for dept, g in valid_department_rows.groupby("Department"):
    have = set(g["Date"].unique())
    miss = [d for d in all_months if d not in have]
    out.append({"Department": dept, "Missing_Months_Count": len(miss)})

if not out:
    st.error("No valid departments remain after parsing the Department column.")
    st.stop()

missing_months_df = pd.DataFrame(out).sort_values(
    "Missing_Months_Count",
    ascending=False,
)
st.dataframe(missing_months_df, width="stretch")
if missing_months_df["Missing_Months_Count"].gt(0).any():
    st.error(
        "One or more departments have missing monthly observations. "
        "Fill or remove gaps before forecasting those departments."
    )
elif not any(quality_error_counts):
    st.success("Dataset is forecast-ready.")
