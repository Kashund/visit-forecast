from __future__ import annotations

from statistics import NormalDist

import pandas as pd
import plotly.graph_objects as go


def forecast_line_chart(
    future_df: pd.DataFrame,
    mae: float | None = None,
    rmse: float | None = None,
    ci_mode: str = "rmse_95",
    source_interval_width: float | None = None,
    phase_metadata: list[dict] | None = None,
):
    """Forecast chart with optional confidence interval + error bands.

    Hover behavior:
    - Shows all plotted traces (original/adjusted + any band boundaries).
    - Uses unified x hover so you can read all values for a date.
    """
    if future_df is None or future_df.empty:
        return None

    df = future_df.copy().sort_values("ds")
    x = df["ds"]

    y_orig = df["yhat_original"]
    y_adj = df["yhat_adjusted"]

    fig = go.Figure()

    # Confidence interval (prefer explicit columns if present)
    lower_col = None
    upper_col = None
    for lc, uc in [("yhat_lower_95", "yhat_upper_95"), ("yhat_lower", "yhat_upper")]:
        if lc in df.columns and uc in df.columns:
            lower_col, upper_col = lc, uc
            break

    if lower_col and upper_col:
        center = (df[upper_col] + df[lower_col]) / 2.0
        half_width = (df[upper_col] - df[lower_col]).abs() / 2.0

        target_label = "95%"
        scaled_upper = df[upper_col]
        scaled_lower = df[lower_col]

        if source_interval_width is not None and 0 < source_interval_width < 1:
            source_z = NormalDist().inv_cdf((1.0 + float(source_interval_width)) / 2.0)
            target_z = 1.96 if ci_mode == "rmse_95" else 1.0
            if source_z > 0:
                width_scale = target_z / source_z
                scaled_upper = center + half_width * width_scale
                scaled_lower = center - half_width * width_scale
                target_label = "95%" if ci_mode == "rmse_95" else "68%"

        # Plot as two boundary lines, plus a fill trace so hover shows actual numbers.
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_upper,
                mode="lines",
                name=f"CI upper ({target_label})",
                line=dict(dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=scaled_lower,
                mode="lines",
                name=f"CI lower ({target_label})",
                line=dict(dash="dot"),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.10)",
            )
        )
    else:
        # Approximate CI using RMSE if available
        if rmse is not None and pd.notna(rmse):
            mult = 1.96 if ci_mode == "rmse_95" else 1.0
            upper = y_adj + mult * rmse
            lower = y_adj - mult * rmse

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper,
                    mode="lines",
                    name=f"Approx CI upper (+{mult}·RMSE)",
                    line=dict(dash="dot"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower,
                    mode="lines",
                    name=f"Approx CI lower (-{mult}·RMSE)",
                    line=dict(dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(0,0,0,0.10)",
                )
            )

    # MAE band boundaries (around adjusted)
    if mae is not None and pd.notna(mae) and mae > 0:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_adj + mae,
                mode="lines",
                name="MAE upper (+MAE)",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_adj - mae,
                mode="lines",
                name="MAE lower (-MAE)",
                line=dict(dash="dash"),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.06)",
            )
        )

    # RMSE band boundaries (around adjusted)
    if rmse is not None and pd.notna(rmse) and rmse > 0:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_adj + rmse,
                mode="lines",
                name="RMSE upper (+RMSE)",
                line=dict(dash="longdash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_adj - rmse,
                mode="lines",
                name="RMSE lower (-RMSE)",
                line=dict(dash="longdash"),
                fill="tonexty",
                fillcolor="rgba(0,0,0,0.04)",
            )
        )

    # Main lines
    fig.add_trace(go.Scatter(x=x, y=y_orig, mode="lines+markers", name="Original"))
    fig.add_trace(go.Scatter(x=x, y=y_adj, mode="lines+markers", name="Adjusted"))

    # Highlight adjusted windows
    adjustment_flag_column = None
    if "adj_applied_any" in df.columns:
        adjustment_flag_column = "adj_applied_any"
    elif "adj_applied" in df.columns:
        adjustment_flag_column = "adj_applied"

    if (
        adjustment_flag_column is not None
        and df[adjustment_flag_column].fillna(False).any()
    ):
        if phase_metadata and "forecast_month" in df.columns:
            for phase_number, phase in enumerate(phase_metadata, start=1):
                if not phase.get("enabled", False):
                    continue

                month_start = int(phase.get("start_month", 1))
                month_end = int(phase.get("end_month", month_start))
                phase_rows = df[
                    df["forecast_month"].between(
                        month_start, month_end, inclusive="both"
                    )
                ]
                if phase_rows.empty:
                    continue

                phase_mode = str(phase.get("mode", "")).capitalize() or "Adjusted"
                phase_percent = float(phase.get("percent", 0.0))
                annotation_text = (
                    f"Phase {phase_number}: {phase_mode} {phase_percent:.1f}%"
                )
                fig.add_vrect(
                    x0=phase_rows["ds"].min(),
                    x1=phase_rows["ds"].max(),
                    opacity=0.12,
                    line_width=0,
                    annotation_text=annotation_text,
                    annotation_position="top left",
                )
        else:
            adjusted_rows = df[df[adjustment_flag_column] == True].copy()
            adjusted_rows["segment_id"] = (
                adjusted_rows.index.to_series().diff().fillna(1).ne(1).cumsum()
            )

            for segment_number, segment_rows in enumerate(
                adjusted_rows.groupby("segment_id", sort=True),
                start=1,
            ):
                _, rows = segment_rows
                fig.add_vrect(
                    x0=rows["ds"].min(),
                    x1=rows["ds"].max(),
                    opacity=0.12,
                    line_width=0,
                    annotation_text=f"Adjusted segment {segment_number}",
                    annotation_position="top left",
                )

    fig.update_layout(
        title="Future Forecast (Original vs Adjusted) with Confidence & Error Bands",
        legend_title_text="Series",
        xaxis_title="Date",
        yaxis_title="Visits",
        hovermode="x unified",
    )
    return fig


def fiscal_bar_chart(
    fiscal_df: pd.DataFrame,
    period_column: str = "Fiscal_Year",
    title_text: str | None = None,
):
    if fiscal_df is None or fiscal_df.empty:
        return None

    import plotly.express as px

    chart_data = fiscal_df.copy()
    category_orders = None

    if period_column == "Fiscal_Period_Label" and {
        "Fiscal_Year",
        "Fiscal_Quarter",
        "Fiscal_Period_Label",
    }.issubset(chart_data.columns):
        chart_data = chart_data.sort_values(
            ["Fiscal_Year", "Fiscal_Quarter"]
        ).reset_index(drop=True)
        category_orders = {
            "Fiscal_Period_Label": chart_data["Fiscal_Period_Label"].tolist()
        }

    if title_text is None:
        title_text = (
            "Fiscal Quarter Aggregates"
            if period_column == "Fiscal_Period_Label"
            else "Fiscal Year Aggregates"
        )

    fig = px.bar(
        chart_data,
        x=period_column,
        y=["Historical_Visits", "Forecast_Original", "Forecast_Adjusted"],
        barmode="group",
        title=title_text,
        category_orders=category_orders,
    )
    fig.update_layout(legend_title_text="Series")
    return fig
