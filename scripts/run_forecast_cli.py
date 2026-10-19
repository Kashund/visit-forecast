from __future__ import annotations

import argparse
import json
import pandas as pd
from typing import Any

from visit_forecast import forecast_visits, forecast_departments
from visit_forecast.exports import (
    forecast_tables,
    scenario_tables,
    write_workflow_export,
)
from visit_forecast.provenance import build_run_manifest


def _parse_phase_argument(phase_argument: str) -> dict[str, Any]:
    """Parse a phase string in the format mode:percent:start:end[:enabled]."""
    parts = [part.strip() for part in phase_argument.split(":")]
    if len(parts) not in {4, 5}:
        raise ValueError("Phase must use mode:percent:start_month:end_month[:enabled].")

    mode, percent_text, start_text, end_text = parts[:4]
    enabled_text = parts[4] if len(parts) == 5 else "true"
    enabled = enabled_text.lower() in {"true", "1", "yes", "y"}

    return {
        "enabled": enabled,
        "mode": mode,
        "percent": float(percent_text),
        "start_month": int(start_text),
        "end_month": int(end_text),
    }


def _parse_capacity_phases(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if args.capacity_phases_json:
        loaded_phases = json.loads(args.capacity_phases_json)
        if not isinstance(loaded_phases, list):
            raise ValueError("--capacity_phases_json must decode to a list.")
        return loaded_phases

    if args.phase:
        return [_parse_phase_argument(phase_argument) for phase_argument in args.phase]

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=False, default=None)
    parser.add_argument("--use_synthetic", action="store_true")
    parser.add_argument("--start", default="2020-04-01")
    parser.add_argument("--end", default="2025-03-01")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--department", default="All")
    parser.add_argument("--adjustment_mode", choices=["loss", "add"], default="loss")
    parser.add_argument("--adjustment_percent", type=float, default=0)
    parser.add_argument("--adjustment_start_month", type=int, default=1)
    parser.add_argument("--adjustment_end_month", type=int, default=12)
    parser.add_argument("--capacity_phases_json", default=None)
    parser.add_argument(
        "--phase",
        action="append",
        default=[],
        help=(
            "Repeatable phase definition: mode:percent:start_month:end_month[:enabled]. "
            "Example: --phase add:10:1:3 --phase loss:15:5:7"
        ),
    )
    parser.add_argument("--changepoint", type=float, default=0.05)
    parser.add_argument("--interval_width", type=float, default=0.90)
    parser.add_argument(
        "--model",
        choices=["auto", "prophet", "ets", "theta", "arima", "seasonal_naive"],
        default="prophet",
        help="Forecast model. 'auto' backtests all models and picks the best.",
    )

    parser.add_argument(
        "--workflow",
        choices=["forecast", "departments", "scenarios"],
        default="forecast",
    )
    parser.add_argument("--calendar-covariates", action="store_true")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Department to exclude; repeat as needed.",
    )
    parser.add_argument(
        "--uncertainty", choices=["prophet", "conformal", "auto"], default="conformal"
    )
    parser.add_argument("--tuning", choices=["manual", "auto"], default="manual")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=["prophet", "ets", "theta", "arima", "seasonal_naive"],
    )
    parser.add_argument(
        "--output-dir", help="Write CSV, Excel, and a run manifest here."
    )
    parser.add_argument(
        "--fiscal-view", choices=["month", "quarter", "year"], default="year"
    )
    parser.add_argument("--scenario-a-json", default="[]")
    parser.add_argument("--scenario-b-json", default="[]")
    args = parser.parse_args()
    capacity_phases = _parse_capacity_phases(args)

    common = dict(
        use_synthetic=args.use_synthetic,
        filename=args.filename,
        timeframe_start=args.start,
        timeframe_end=args.end,
        forecast_periods=args.months,
        model_choice=args.model,
        exclude_departments=args.exclude,
        use_calendar_covariates=args.calendar_covariates,
        uncertainty_method=args.uncertainty,
        auto_candidates=args.candidates,
        changepoint_prior_scale=args.changepoint,
        interval_width=args.interval_width,
    )
    if args.workflow == "departments":
        bundle = forecast_departments(
            **common,
            departments=None if args.department == "All" else [args.department],
        )
        tables = {
            "Model_Summary": bundle.summary_df,
            "Forecasts_By_Department": bundle.combined_future_df,
            "Failures": pd.DataFrame(
                [{"Department": k, "Reason": v} for k, v in bundle.failures.items()],
                columns=["Department", "Reason"],
            ),
            "Diagnostics": pd.concat(
                [
                    r.tuning_diagnostics_df.assign(Department=d)
                    for d, r in bundle.results.items()
                    if r.tuning_diagnostics_df is not None
                ],
                ignore_index=True,
            )
            if any(r.tuning_diagnostics_df is not None for r in bundle.results.values())
            else pd.DataFrame(),
        }
        manifest = build_run_manifest(bundle.combined_future_df, vars(args))
        manifest["department_runs"] = {
            d: r.run_manifest for d, r in bundle.results.items()
        }
        print(bundle.summary_df.to_string(index=False))
        if bundle.failures:
            print("Partial forecast; failures:", bundle.failures)
    else:
        result = forecast_visits(
            **common,
            department=args.department,
            tuning_mode=args.tuning,
            adjustment_mode=args.adjustment_mode,
            adjustment_percent=args.adjustment_percent,
            adjustment_start_month=args.adjustment_start_month,
            adjustment_end_month=args.adjustment_end_month,
            capacity_phases=capacity_phases if args.workflow == "forecast" else [],
        )
        manifest = result.run_manifest
        if args.workflow == "scenarios":
            a, b = json.loads(args.scenario_a_json), json.loads(args.scenario_b_json)
            monthly, fiscal, definitions = scenario_tables(result, a, b)
            tables = {
                "Monthly_Comparison": monthly,
                "FY_Summary": fiscal,
                "Scenario_Definitions": definitions,
            }
            manifest = {**manifest, "scenarios": {"A": a, "B": b}}
        else:
            tables = forecast_tables(result, args.fiscal_view)
        print("Model:", result.selected_model)
        print("Note:", result.tuning_note)
        print("Departments:", result.department_info)
        print("Metrics:", result.performance_metrics)
        print("Evaluation:", result.evaluation_summary)
        print(list(tables.values())[-1].to_string(index=False))
    if args.output_dir:
        write_workflow_export(args.output_dir, tables, manifest)
        print("Exports written to", args.output_dir)


if __name__ == "__main__":
    main()
