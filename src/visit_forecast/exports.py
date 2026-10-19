"""Shared, full-precision workflow exports."""

from __future__ import annotations
import io
import json
from pathlib import Path
import pandas as pd
from .fiscal import aggregate_summary_by_period
from .service import apply_capacity_phases_to_future_df
from .provenance import manifest_json, json_safe


def scenario_tables(result, phases_a, phases_b):
    baseline = result.future_forecast_df
    a = apply_capacity_phases_to_future_df(baseline, phases_a)
    b = apply_capacity_phases_to_future_df(baseline, phases_b)
    monthly = baseline[["ds", "Fiscal_Year", "forecast_month"]].copy()
    for column, series in [
        ("Baseline", baseline.yhat_original),
        ("Scenario_A", a.yhat_adjusted),
        ("Scenario_B", b.yhat_adjusted),
    ]:
        monthly[column] = series
    monthly["A_minus_Baseline"] = monthly.Scenario_A - monthly.Baseline
    monthly["B_minus_Baseline"] = monthly.Scenario_B - monthly.Baseline
    monthly["A_minus_B"] = monthly.Scenario_A - monthly.Scenario_B
    fiscal = monthly.groupby("Fiscal_Year", as_index=False)[
        [
            "Baseline",
            "Scenario_A",
            "Scenario_B",
            "A_minus_Baseline",
            "B_minus_Baseline",
            "A_minus_B",
        ]
    ].sum()
    counts = monthly.groupby("Fiscal_Year").ds.nunique()
    fiscal["Forecast_Months"] = fiscal.Fiscal_Year.map(counts)
    fiscal["Coverage_Status"] = fiscal.Forecast_Months.map(
        lambda n: "Complete" if n == 12 else "Partial"
    )
    definitions = pd.DataFrame(
        [
            {"Scenario": name, "Phase": i, **phase}
            for name, phases in [("A", phases_a), ("B", phases_b)]
            for i, phase in enumerate(phases, 1)
        ]
    )
    return monthly, fiscal, definitions


def excel_bytes(sheets, manifest=None):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        if manifest:
            pd.DataFrame(
                [
                    {
                        "field": key,
                        "value": json.dumps(
                            json_safe(value), default=str, allow_nan=False
                        ),
                    }
                    for key, value in manifest.items()
                ]
            ).to_excel(writer, sheet_name="Run_Manifest", index=False)
    return buffer.getvalue()


def forecast_tables(result, period="year"):
    history = result.series.to_dataframe().reset_index()
    history = history.rename(
        columns={history.columns[0]: "ds", history.columns[1]: "y"}
    )[["ds", "y"]]
    return {
        "Future_Forecast": result.future_forecast_df,
        "Fiscal_Summary": aggregate_summary_by_period(
            history, result.future_forecast_df, period
        ),
    }


def write_workflow_export(directory, tables, manifest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name.lower()}.csv", index=False)
    (directory / "forecast.xlsx").write_bytes(excel_bytes(tables, manifest))
    (directory / "run_manifest.json").write_text(
        manifest_json(manifest), encoding="utf-8"
    )
