from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from darts import TimeSeries


@dataclass(frozen=True)
class ForecastResult:
    series: TimeSeries
    forecast: TimeSeries
    future_forecast_df: pd.DataFrame
    fiscal_summary: pd.DataFrame
    performance_metrics: Dict[str, float]
    department_info: str

    # Prophet-native artifacts (optional)
    prophet_forecast_df: Optional[pd.DataFrame] = None   # output of Prophet.predict on future dataframe
    prophet_cv_metrics_df: Optional[pd.DataFrame] = None # output of prophet.diagnostics.performance_metrics
    prophet_cv_raw_df: Optional[pd.DataFrame] = None     # output of prophet.diagnostics.cross_validation
