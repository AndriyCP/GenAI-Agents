import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_growth_rate(current: float, previous: float) -> float:
    """Calculate growth rate between two values."""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate moving average for a time series."""
    return data.rolling(window=window).mean()

def calculate_seasonal_strength(decomposition) -> float:
    """Calculate the strength of seasonality in a time series."""
    return 1 - (decomposition.resid.var() / 
               (decomposition.trend + decomposition.seasonal).var())

def format_number(number: float) -> str:
    """Format number for display."""
    if number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    else:
        return f"{number:.1f}"

def get_top_n_items(data: pd.DataFrame, 
                    group_col: str, 
                    value_col: str, 
                    n: int = 5) -> List[Dict[str, Any]]:
    """Get top N items by value from a DataFrame."""
    return (data.groupby(group_col)[value_col]
            .sum()
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
            .to_dict('records'))

def calculate_percentile_rank(value: float, 
                            data: pd.Series) -> float:
    """Calculate percentile rank of a value in a series."""
    return (data < value).mean() * 100

def detect_anomalies(data: pd.Series, 
                    window: int = 7, 
                    threshold: float = 2.0) -> pd.Series:
    """Detect anomalies in time series data using rolling statistics."""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    z_scores = (data - rolling_mean) / rolling_std
    return abs(z_scores) > threshold 