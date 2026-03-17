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
    tuning_mode: str = "manual"
    selected_changepoint_prior_scale: Optional[float] = None
    selected_interval_width: Optional[float] = None
    tuning_primary_metric: Optional[str] = None
    tuning_primary_score: Optional[float] = None
    tuning_note: Optional[str] = None
    tuning_diagnostics_df: Optional[pd.DataFrame] = None
