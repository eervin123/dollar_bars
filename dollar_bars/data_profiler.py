"""
Data profiling functionality for dollar bars.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from datetime import timedelta


def data_profiler(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Profile OHLCV data to help determine appropriate dollar bar sizes.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary containing data profile information
    """
    # Ensure all column names are lowercase
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Validate required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Calculate basic statistics
    dollar_values = df["close"] * df["volume"]
    total_dollar_value = dollar_values.sum()
    avg_dollar_value = dollar_values.mean()
    std_dollar_value = dollar_values.std()

    # Calculate time-based statistics
    if isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index.to_series().diff()
        avg_time_diff = time_diff.mean()
        total_duration = df.index[-1] - df.index[0]
    else:
        time_diff = pd.to_datetime(df["open time"]).diff()
        avg_time_diff = time_diff.mean()
        total_duration = pd.to_datetime(df["open time"].iloc[-1]) - pd.to_datetime(
            df["open time"].iloc[0]
        )

    # Calculate suggested bar sizes
    conservative_size = avg_dollar_value * 2
    moderate_size = avg_dollar_value * 5
    aggressive_size = avg_dollar_value * 10

    # Calculate volume profile
    volume_profile = {
        "total_volume": df["volume"].sum(),
        "avg_volume": df["volume"].mean(),
        "std_volume": df["volume"].std(),
        "min_volume": df["volume"].min(),
        "max_volume": df["volume"].max(),
    }

    # Calculate price profile
    price_profile = {
        "avg_price": df["close"].mean(),
        "std_price": df["close"].std(),
        "min_price": df["low"].min(),
        "max_price": df["high"].max(),
    }

    # Calculate time profile
    time_profile = {
        "total_duration": total_duration,
        "avg_time_diff": avg_time_diff,
        "total_periods": len(df),
    }

    # Calculate dollar value profile
    dollar_value_profile = {
        "total_dollar_value": total_dollar_value,
        "avg_dollar_value": avg_dollar_value,
        "std_dollar_value": std_dollar_value,
        "min_dollar_value": dollar_values.min(),
        "max_dollar_value": dollar_values.max(),
    }

    # Calculate suggested bar sizes based on different criteria
    suggested_sizes = {
        "conservative": conservative_size,
        "moderate": moderate_size,
        "aggressive": aggressive_size,
    }

    # Calculate expected number of bars for each suggested size
    expected_bars = {
        "conservative": total_dollar_value / conservative_size,
        "moderate": total_dollar_value / moderate_size,
        "aggressive": total_dollar_value / aggressive_size,
    }

    # Calculate expected bar duration for each suggested size
    expected_duration = {
        "conservative": total_duration / expected_bars["conservative"],
        "moderate": total_duration / expected_bars["moderate"],
        "aggressive": total_duration / expected_bars["aggressive"],
    }

    return {
        "volume_profile": volume_profile,
        "price_profile": price_profile,
        "time_profile": time_profile,
        "dollar_value_profile": dollar_value_profile,
        "recommendations": {
            "suggested_dollar_bar_sizes": suggested_sizes,
            "expected_bars": expected_bars,
            "expected_duration": expected_duration,
        },
    }


def describe_dollar_bars(
    df: pd.DataFrame,
    dollar_bar_size: float,
) -> Dict[str, Any]:
    """
    Generate descriptive statistics for dollar bars.

    Args:
        df: DataFrame with dollar bars
        dollar_bar_size: Target dollar value for each bar

    Returns:
        Dictionary containing descriptive statistics
    """
    # Ensure all column names are lowercase
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Calculate basic statistics
    total_bars = len(df)
    total_volume = df["volume"].sum()
    total_dollar_value = (df["close"] * df["volume"]).sum()

    # Calculate bar duration statistics
    if "open time" in df.columns and "close time" in df.columns:
        durations = pd.to_datetime(df["close time"]) - pd.to_datetime(df["open time"])
        duration_stats = {
            "mean_minutes": durations.mean().total_seconds() / 60,
            "std_minutes": durations.std().total_seconds() / 60,
            "min_minutes": durations.min().total_seconds() / 60,
            "max_minutes": durations.max().total_seconds() / 60,
        }
    else:
        duration_stats = None

    # Calculate dollar value statistics
    dollar_values = df["close"] * df["volume"]
    dollar_value_stats = {
        "mean": dollar_values.mean(),
        "std": dollar_values.std(),
        "min": dollar_values.min(),
        "max": dollar_values.max(),
    }

    # Calculate volume statistics
    volume_stats = {
        "mean": df["volume"].mean(),
        "std": df["volume"].std(),
        "min": df["volume"].min(),
        "max": df["volume"].max(),
    }

    # Calculate price statistics
    price_stats = {
        "mean": df["close"].mean(),
        "std": df["close"].std(),
        "min": df["low"].min(),
        "max": df["high"].max(),
    }

    return {
        "total_bars": total_bars,
        "total_volume": total_volume,
        "total_dollar_value": total_dollar_value,
        "bar_duration_stats": duration_stats,
        "dollar_value_stats": dollar_value_stats,
        "volume_stats": volume_stats,
        "price_stats": price_stats,
    }
