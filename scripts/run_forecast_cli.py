from __future__ import annotations

import argparse
from visit_forecast import forecast_visits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filename", required=False, default=None)
    p.add_argument("--use_synthetic", action="store_true")
    p.add_argument("--start", default="2020-04-01")
    p.add_argument("--end", default="2025-03-01")
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--department", default="All")
    p.add_argument("--adjustment_mode", choices=["loss","add"], default="loss")
    p.add_argument("--adjustment_percent", type=float, default=0)
    p.add_argument("--adjustment_start_month", type=int, default=1)
    p.add_argument("--adjustment_end_month", type=int, default=12)
    p.add_argument("--changepoint", type=float, default=0.05)
    p.add_argument("--interval_width", type=float, default=0.90)

    args = p.parse_args()

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
        changepoint_prior_scale=args.changepoint,
        interval_width=args.interval_width,
    )

    print("Departments:", result.department_info)
    print("Metrics:", result.performance_metrics)
    print(result.fiscal_summary)


if __name__ == "__main__":
    main()
