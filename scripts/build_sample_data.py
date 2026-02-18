from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

def generate_synthetic_data(
    timeframe_start: str,
    timeframe_end: str,
    departments: list[str] | None = None,
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

def main(out_dir: str = "data/sample"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data("2020-04-01", "2025-03-01")
    df.to_csv(out_path / "synthetic_visits.csv", index=False)

    df_small = df[df["Date"] <= "2021-03-01"].copy()
    df_small.to_csv(out_path / "synthetic_visits_small.csv", index=False)

    print(f"Wrote: {out_path/'synthetic_visits.csv'}")
    print(f"Wrote: {out_path/'synthetic_visits_small.csv'}")

if __name__ == "__main__":
    main()
