"""
Tests for the dollar_bars package.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest
from dollar_bars import generate_dollar_bars, data_profiler, describe_dollar_bars


def create_test_data(n_rows: int) -> pd.DataFrame:
    """Create test data for dollar bars."""
    np.random.seed(42)  # For reproducibility

    # Create timestamps
    timestamps = pd.date_range(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        periods=n_rows,
        freq="1min",
        tz="UTC",
    )

    # Generate random price data with some trend and volatility
    base_price = 100.0
    trend = np.linspace(0, 20, n_rows)  # Add a slight upward trend
    noise = np.random.normal(0, 2, n_rows)  # Add some noise
    close_prices = base_price + trend + noise

    # Generate volume with some spikes
    base_volume = 100
    volume = np.random.gamma(2, 2, n_rows) * base_volume
    volume[np.random.randint(0, n_rows, 20)] *= 5  # Add some volume spikes

    # Create OHLCV data
    df = pd.DataFrame(
        {
            "Open": close_prices * (1 + np.random.normal(0, 0.001, n_rows)),
            "High": close_prices * (1 + abs(np.random.normal(0, 0.002, n_rows))),
            "Low": close_prices * (1 - abs(np.random.normal(0, 0.002, n_rows))),
            "Close": close_prices,
            "Volume": volume,
        },
        index=timestamps,
    )

    # Ensure High is highest and Low is lowest
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


def test_basic_functionality():
    """Test basic dollar bar creation with simple data."""
    # Create test data
    df = create_test_data(1000)
    dollar_bar_size = 1000.0  # Each bar should contain ~10 rows

    # Test raw bar creation
    raw_bars = generate_dollar_bars(
        df, dollar_bar_size, return_aligned_to_original_time=False, use_numba=True
    )

    # Verify basic properties
    assert not raw_bars.empty, "Should create some bars"
    assert len(raw_bars) > 0, "Should create at least one bar"
    assert all(
        col in raw_bars.columns
        for col in [
            "Open time",
            "Close time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    # Verify bar properties
    for _, bar in raw_bars.iterrows():
        assert bar["High"] >= bar["Low"], "High should be >= Low"
        assert bar["Open"] <= bar["High"], "Open should be <= High"
        assert bar["Open"] >= bar["Low"], "Open should be >= Low"
        assert bar["Close"] <= bar["High"], "Close should be <= High"
        assert bar["Close"] >= bar["Low"], "Close should be >= Low"
        assert bar["Volume"] > 0, "Volume should be positive"
        assert (
            bar["Open time"] <= bar["Close time"]
        ), "Open time should be <= Close time"


def test_dollar_bar_size():
    """Test that dollar bars are mathematically correct."""
    # Create test data
    df = create_test_data(1000)
    dollar_bar_size = 1000.0

    raw_bars = generate_dollar_bars(
        df, dollar_bar_size, return_aligned_to_original_time=False, use_numba=True
    )

    # Calculate dollar value for each bar
    bar_dollar_values = raw_bars["Close"] * raw_bars["Volume"]

    # Test mathematical correctness
    assert all(raw_bars["Volume"] > 0), "All bars should have positive volume"
    assert all(
        raw_bars["High"] >= raw_bars["Low"]
    ), "High should be >= Low for all bars"
    assert all(raw_bars["Open"] <= raw_bars["High"]), "Open should be <= High"
    assert all(raw_bars["Open"] >= raw_bars["Low"]), "Open should be >= Low"
    assert all(raw_bars["Close"] <= raw_bars["High"]), "Close should be <= High"
    assert all(raw_bars["Close"] >= raw_bars["Low"]), "Close should be >= Low"
    assert all(
        raw_bars["Open time"] <= raw_bars["Close time"]
    ), "Open time should be <= Close time"

    # Test that bars are continuous (no gaps)
    for i in range(len(raw_bars) - 1):
        expected_close = raw_bars["Open time"].iloc[i + 1] - pd.Timedelta(
            microseconds=1
        )
        actual_close = raw_bars["Close time"].iloc[i]
        assert (
            actual_close == expected_close
        ), f"Bar {i} close time {actual_close} does not match next bar open time {expected_close}"

    # Test that bars don't overlap (except for the last bar which might be partial)
    for i in range(len(raw_bars) - 1):
        assert (
            raw_bars["Close time"].iloc[i] <= raw_bars["Open time"].iloc[i + 1]
        ), f"Bar {i} ends after bar {i+1} starts"

    # Test volume conservation
    total_input_volume = df["Volume"].sum()
    total_bar_volume = raw_bars["Volume"].sum()
    volume_diff_pct = (
        abs(total_input_volume - total_bar_volume) / total_input_volume * 100
    )
    assert volume_diff_pct < 5.0, "Volume should be reasonably conserved (within 5%)"


def test_alignment():
    """Test that dollar bars can be aligned to the original timeseries."""
    # Create test data with timezone-aware timestamps
    df = create_test_data(1000)
    df.index = pd.date_range(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        periods=len(df),
        freq="1min",
        tz="UTC",
    )

    # Test alignment with raw bars
    aligned_bars = generate_dollar_bars(
        df,
        dollar_bar_size=1000.0,
        return_aligned_to_original_time=True,
        use_numba=True,
    )

    # Test basic properties
    assert len(aligned_bars) == len(
        df
    ), "Aligned DataFrame should have same length as input"
    assert all(
        aligned_bars.index == df.index
    ), "Aligned DataFrame should have same index as input"

    # Test that dollar bar data is properly forward-filled
    dollar_bar_cols = [col for col in aligned_bars.columns if col.startswith("db_")]
    assert len(dollar_bar_cols) > 0, "No dollar bar columns found"

    # Test that NewDBFlag correctly identifies new bars
    flag_col = [col for col in dollar_bar_cols if col.endswith("NewDBFlag")][0]
    assert (
        aligned_bars[flag_col].sum() > 0
    ), "No new dollar bars found in aligned DataFrame"

    # Test that all dollar bar columns are filled
    for col in dollar_bar_cols:
        if not col.endswith("NewDBFlag"):
            assert (
                not aligned_bars[col].isna().any()
            ), f"Column {col} has missing values"


def test_edge_cases():
    """Test edge cases like empty DataFrames and extreme values."""
    # Test empty DataFrame
    empty_df = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    empty_bars = generate_dollar_bars(empty_df, 1000.0, use_numba=True)
    assert len(empty_bars) == 0, "Empty DataFrame should produce empty dollar bars"

    # Test DataFrame with all zeros
    zero_df = pd.DataFrame(
        {
            "Open": [0.0] * 10,
            "High": [0.0] * 10,
            "Low": [0.0] * 10,
            "Close": [0.0] * 10,
            "Volume": [0.0] * 10,
        },
        index=pd.date_range(start="2024-01-01", periods=10, freq="1min", tz="UTC"),
    )
    zero_bars = generate_dollar_bars(zero_df, 1000.0, use_numba=True)
    assert len(zero_bars) == 0, "Zero values should produce no dollar bars"

    # Test DataFrame with one valid row
    single_row_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.0],
            "Volume": [10.0],
        },
        index=pd.date_range(start="2024-01-01", periods=1, freq="1min", tz="UTC"),
    )
    single_bars = generate_dollar_bars(single_row_df, 1000.0, use_numba=True)
    assert len(single_bars) == 1, "Single valid row should produce one bar"

    # Test DataFrame with extreme values
    extreme_df = pd.DataFrame(
        {
            "Open": [1e9],
            "High": [1e9],
            "Low": [1e9],
            "Close": [1e9],
            "Volume": [1e9],
        },
        index=pd.date_range(start="2024-01-01", periods=1, freq="1min", tz="UTC"),
    )
    extreme_bars = generate_dollar_bars(extreme_df, 1e6, use_numba=True)
    assert len(extreme_bars) > 0, "Extreme values should still produce bars"


def test_consistency():
    """Test consistency between Numba and non-Numba versions."""
    df = create_test_data(1000)
    dollar_bar_size = 1000.0

    # Get results from both versions
    numba_bars = generate_dollar_bars(
        df, dollar_bar_size, return_aligned_to_original_time=False, use_numba=True
    )
    regular_bars = generate_dollar_bars(
        df, dollar_bar_size, return_aligned_to_original_time=False, use_numba=False
    )

    # Compare results
    pd.testing.assert_frame_equal(
        numba_bars.reset_index(drop=True),
        regular_bars.reset_index(drop=True),
        check_dtype=False,
        atol=1e-5,
    ), "Numba and regular versions should produce same results"
