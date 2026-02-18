from __future__ import annotations

import argparse
import json
from typing import Any

from visit_forecast import forecast_visits


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

    args = parser.parse_args()
    capacity_phases = _parse_capacity_phases(args)

    result = forecast_visits(
        use_synthetic=args.use_synthetic,
        filename=args.filename,
        timeframe_start=args.start,
        timeframe_end=args.end,
        forecast_periods=args.months,
        department=args.department,
        adjustment_mode=args.adjustment_mode,
        adjustment_percent=args.adjustment_percent,
        adjustment_start_month=args.adjustment_start_month,
        adjustment_end_month=args.adjustment_end_month,
        capacity_phases=capacity_phases,
        changepoint_prior_scale=args.changepoint,
        interval_width=args.interval_width,
    )

    print("Departments:", result.department_info)
    print("Metrics:", result.performance_metrics)
    print(result.fiscal_summary)


if __name__ == "__main__":
    main()
