from __future__ import annotations

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Check", page_icon="✅", layout="wide")
st.title("✅ Data Check")

uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])


def read_file(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


if uploaded is None:
    st.info("Upload a dataset to validate required columns and detect missing months.")
    st.stop()

df = read_file(uploaded)
st.dataframe(df.head(30), use_container_width=True)

required = {"Date", "Visits", "Department"}
missing = required - set(df.columns)

if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

st.success("Required columns are present.")

df2 = df.copy()
df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
df2["Visits"] = pd.to_numeric(df2["Visits"], errors="coerce")

st.write("Null counts")
st.dataframe(df2[["Date", "Visits", "Department"]].isna().sum().to_frame("nulls"), use_container_width=True)

df2 = df2.dropna(subset=["Date"])
df2["Date"] = df2["Date"].dt.to_period("M").dt.to_timestamp("MS")

min_d, max_d = df2["Date"].min(), df2["Date"].max()
all_months = pd.date_range(min_d, max_d, freq="MS")

st.subheader("Missing months by department")
out = []
for dept, g in df2.groupby("Department"):
    have = set(g["Date"].unique())
    miss = [d for d in all_months if d not in have]
    out.append({"Department": dept, "Missing_Months_Count": len(miss)})

st.dataframe(pd.DataFrame(out).sort_values("Missing_Months_Count", ascending=False), use_container_width=True)
