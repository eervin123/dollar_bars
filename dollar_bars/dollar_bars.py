import pandas as pd
import numpy as np
from numba import njit


def _create_raw_dollar_bars(
    ohlc_df: pd.DataFrame, dollar_bar_size: float
) -> pd.DataFrame:
    """
    Creates raw dollar bars from OHLCV data.
    Focuses on 'Open', 'High', 'Low', 'Close', 'Volume'.
    Includes 'Open time' and 'Close time' for each bar.
    All timestamps are in UTC.
    """
    if not isinstance(ohlc_df, pd.DataFrame) or ohlc_df.empty:
        cols = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
        return pd.DataFrame(columns=cols)

    if dollar_bar_size <= 0:
        raise ValueError("dollar_bar_size must be positive.")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in ohlc_df.columns:
            raise ValueError(f"ohlc_df must contain column: {col}")

    has_datetime_index = isinstance(ohlc_df.index, pd.DatetimeIndex)
    has_open_time_col = "Open time" in ohlc_df.columns
    if not has_datetime_index and not has_open_time_col:
        raise ValueError(
            "ohlc_df must have a DatetimeIndex or an 'Open time' column to determine bar timestamps."
        )

    df = ohlc_df.copy()  # Avoid modifying original DataFrame
    df["DollarValue"] = df["Close"] * df["Volume"]

    bar_ranges = []  # Stores tuples of (start_idx, end_idx_exclusive)
    current_bar_start_idx = 0
    current_bar_dollar_sum = 0.0

    for i in range(len(df)):
        current_bar_dollar_sum += df["DollarValue"].iloc[i]
        if current_bar_dollar_sum >= dollar_bar_size:
            bar_ranges.append((current_bar_start_idx, i + 1))
            current_bar_start_idx = i + 1
            current_bar_dollar_sum = 0.0

    # If the last portion of data forms a partial bar, it's currently ignored.
    # To include it:
    # if current_bar_start_idx < len(df) and current_bar_dollar_sum > 0 :
    #     bar_ranges.append((current_bar_start_idx, len(df)))

    if not bar_ranges:
        cols = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
        return pd.DataFrame(columns=cols)

    dollar_bars_list = []
    open_times = []
    for start_idx, end_idx_exclusive in bar_ranges:
        bar_slice = df.iloc[start_idx:end_idx_exclusive]

        open_price = df["Open"].iloc[start_idx]
        high_price = bar_slice["High"].max()
        low_price = bar_slice["Low"].min()
        close_price = df["Close"].iloc[end_idx_exclusive - 1]
        volume_sum = bar_slice["Volume"].sum()

        if has_datetime_index:
            open_time = _ensure_utc_timestamp(df.index[start_idx])
            open_times.append(open_time)
            close_time = _ensure_utc_timestamp(df.index[end_idx_exclusive - 1])
        else:  # has_open_time_col is True
            open_time = _ensure_utc_timestamp(
                pd.to_datetime(df["Open time"].iloc[start_idx])
            )
            open_times.append(open_time)
            close_time = _ensure_utc_timestamp(
                pd.to_datetime(df["Open time"].iloc[end_idx_exclusive - 1])
            )

        dollar_bars_list.append(
            {
                "Open time": open_time,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume_sum,
                "Close time": close_time,  # will adjust below
            }
        )

    # Adjust close times for continuity
    for i in range(len(dollar_bars_list) - 1):
        next_open = open_times[i + 1]
        dollar_bars_list[i]["Close time"] = next_open - pd.Timedelta(microseconds=1)
    # For the last bar, keep as end of last included period + 59.999s
    if len(dollar_bars_list) > 0:
        last_idx = len(dollar_bars_list) - 1
        last_close = dollar_bars_list[last_idx]["Close time"]
        dollar_bars_list[last_idx]["Close time"] = last_close + pd.Timedelta(
            seconds=59.999
        )

    raw_bars_df = pd.DataFrame(dollar_bars_list)
    if not raw_bars_df.empty:
        # Ensure standard column order
        raw_bars_df = raw_bars_df[
            ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
        ]
    return raw_bars_df


def _simplify_number(num: float) -> str:
    """
    Simplifies a large number by converting it to a shorter representation with a suffix (K, M, B),
    rounding to the nearest integer for each suffix.
    E.g., 95535298.67 -> 96M
    """
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{int(round(num / 1_000_000_000))}B"
    elif abs_num >= 1_000_000:
        return f"{int(round(num / 1_000_000))}M"
    elif abs_num >= 1_000:
        return f"{int(round(num / 1_000))}K"
    else:
        return str(int(round(num)))


def _align_dollar_bars_to_timeseries(
    original_df: pd.DataFrame, raw_dollar_bars_df: pd.DataFrame, dollar_bar_size: float
) -> pd.DataFrame:
    """
    Aligns raw dollar bars to the original DataFrame's timeseries, forward-filling the data.
    original_df must have a DatetimeIndex for this alignment.
    All timestamps are in UTC.
    """
    if not isinstance(original_df.index, pd.DatetimeIndex):
        raise ValueError(
            "_align_dollar_bars_to_timeseries requires original_df to have a DatetimeIndex."
        )

    # Determine prefix string using simplified number
    prefix_val = _simplify_number(dollar_bar_size)
    dollar_bar_prefix = f"db_{prefix_val}_"

    if raw_dollar_bars_df.empty:
        # If no dollar bars, create empty prefixed columns on the original_df
        merged_df = original_df.copy()
        # Define typical columns that would come from dollar bars
        example_dollar_bar_cols = [
            "Open time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close time",
        ]
        for col_name in example_dollar_bar_cols:
            merged_df[dollar_bar_prefix + col_name] = np.nan
        merged_df[dollar_bar_prefix + "NewDBFlag"] = False
        return merged_df

    # Ensure both DataFrames have UTC timezone
    original_df = original_df.copy()
    if original_df.index.tz is None:
        original_df.index = original_df.index.tz_localize("UTC")
    else:
        original_df.index = original_df.index.tz_convert("UTC")

    dollar_bars_df_renamed = raw_dollar_bars_df.copy()
    dollar_bars_df_renamed.columns = [
        dollar_bar_prefix + col for col in dollar_bars_df_renamed.columns
    ]

    # Set index for the dollar bars df for merging, using its own (prefixed) 'Open time'
    open_time_col = dollar_bar_prefix + "Open time"
    if pd.api.types.is_datetime64_any_dtype(dollar_bars_df_renamed[open_time_col]):
        if dollar_bars_df_renamed[open_time_col].dt.tz is None:
            dollar_bars_df_renamed[open_time_col] = dollar_bars_df_renamed[
                open_time_col
            ].dt.tz_localize("UTC")
        else:
            dollar_bars_df_renamed[open_time_col] = dollar_bars_df_renamed[
                open_time_col
            ].dt.tz_convert("UTC")
    else:
        dollar_bars_df_renamed[open_time_col] = pd.to_datetime(
            dollar_bars_df_renamed[open_time_col]
        ).dt.tz_localize("UTC")

    # Handle potential duplicate timestamps in dollar bars
    if dollar_bars_df_renamed[open_time_col].duplicated().any():
        # Keep the first occurrence of each timestamp
        dollar_bars_df_renamed = dollar_bars_df_renamed.loc[
            ~dollar_bars_df_renamed[open_time_col].duplicated()
        ]

    # Set the index and merge
    dollar_bars_df_renamed = dollar_bars_df_renamed.set_index(open_time_col, drop=False)
    merged_df = original_df.merge(
        dollar_bars_df_renamed, how="left", left_index=True, right_index=True
    )

    # Create NewDBFlag before forward-filling data columns
    merged_df[dollar_bar_prefix + "NewDBFlag"] = ~merged_df[
        dollar_bar_prefix + "Close"
    ].isna()

    # Identify only the newly added (prefixed) dollar bar columns to forward-fill
    dollar_bar_derived_cols = [
        col
        for col in merged_df.columns
        if col.startswith(dollar_bar_prefix) and col != dollar_bar_prefix + "NewDBFlag"
    ]

    # Use ffill() instead of fillna(method='ffill')
    merged_df[dollar_bar_derived_cols] = merged_df[dollar_bar_derived_cols].ffill()

    # Fill any remaining NaNs in NewDBFlag (e.g., at the start before any bar) with False
    merged_df[dollar_bar_prefix + "NewDBFlag"] = merged_df[
        dollar_bar_prefix + "NewDBFlag"
    ].fillna(False)

    return merged_df


@njit
def _calculate_dollar_bars_nb_core(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    dollar_value_arr: np.ndarray,
    dollar_bar_size: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Core Numba function for calculating dollar bars.
    Returns arrays of indices and values, keeping timestamps separate from the calculation.
    Respects minute boundaries - never splits a minute into multiple bars.
    """
    n_rows = len(open_arr)
    if n_rows == 0:
        return (
            np.empty(0, dtype=np.int64),  # start indices
            np.empty(0, dtype=np.int64),  # end indices
            np.empty(0, dtype=np.float64),  # open prices
            np.empty(0, dtype=np.float64),  # high prices
            np.empty(0, dtype=np.float64),  # low prices
            np.empty(0, dtype=np.float64),  # close prices
            np.empty(0, dtype=np.float64),  # volumes
        )

    # Pre-allocate arrays with maximum possible size
    max_bars = n_rows  # We'll never have more bars than rows
    out_start_idx = np.empty(max_bars, dtype=np.int64)
    out_end_idx = np.empty(max_bars, dtype=np.int64)
    out_open = np.empty(max_bars, dtype=np.float64)
    out_high = np.empty(max_bars, dtype=np.float64)
    out_low = np.empty(max_bars, dtype=np.float64)
    out_close = np.empty(max_bars, dtype=np.float64)
    out_volume = np.empty(max_bars, dtype=np.float64)

    current_bar_start_idx = 0
    current_bar_dollar_sum = 0.0
    bar_count = 0

    # Skip initial zero-value minutes
    while (
        current_bar_start_idx < n_rows and dollar_value_arr[current_bar_start_idx] == 0
    ):
        current_bar_start_idx += 1

    if current_bar_start_idx == n_rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    for i in range(current_bar_start_idx, n_rows):
        # Skip zero-value minutes
        if dollar_value_arr[i] == 0:
            continue

        # Add current minute's dollar value to the running sum
        current_bar_dollar_sum += dollar_value_arr[i]

        # If we've reached or exceeded the target size, create a bar
        if current_bar_dollar_sum >= dollar_bar_size:
            # Create bar
            out_start_idx[bar_count] = current_bar_start_idx
            out_end_idx[bar_count] = i
            out_open[bar_count] = open_arr[current_bar_start_idx]

            # Calculate high and low for the bar
            bar_high = high_arr[current_bar_start_idx]
            bar_low = low_arr[current_bar_start_idx]
            for j in range(current_bar_start_idx, i + 1):
                if dollar_value_arr[j] > 0:  # Only include non-zero minutes
                    bar_high = max(bar_high, high_arr[j])
                    bar_low = min(bar_low, low_arr[j])

            out_high[bar_count] = bar_high
            out_low[bar_count] = bar_low
            out_close[bar_count] = close_arr[i]
            out_volume[bar_count] = sum(volume_arr[current_bar_start_idx : i + 1])

            bar_count += 1

            # Find next non-zero minute to start new bar
            current_bar_start_idx = i + 1
            while (
                current_bar_start_idx < n_rows
                and dollar_value_arr[current_bar_start_idx] == 0
            ):
                current_bar_start_idx += 1
            current_bar_dollar_sum = 0.0

    # Handle the last bar if it has any non-zero data
    if current_bar_start_idx < n_rows and current_bar_dollar_sum > 0:
        out_start_idx[bar_count] = current_bar_start_idx
        out_end_idx[bar_count] = n_rows - 1
        out_open[bar_count] = open_arr[current_bar_start_idx]

        # Calculate high and low for the last bar
        bar_high = high_arr[current_bar_start_idx]
        bar_low = low_arr[current_bar_start_idx]
        for j in range(current_bar_start_idx, n_rows):
            if dollar_value_arr[j] > 0:  # Only include non-zero minutes
                bar_high = max(bar_high, high_arr[j])
                bar_low = min(bar_low, low_arr[j])

        out_high[bar_count] = bar_high
        out_low[bar_count] = bar_low
        out_close[bar_count] = close_arr[n_rows - 1]
        out_volume[bar_count] = sum(volume_arr[current_bar_start_idx:n_rows])
        bar_count += 1

    # Trim arrays to actual size
    return (
        out_start_idx[:bar_count],
        out_end_idx[:bar_count],
        out_open[:bar_count],
        out_high[:bar_count],
        out_low[:bar_count],
        out_close[:bar_count],
        out_volume[:bar_count],
    )


def _create_raw_dollar_bars_nb(
    ohlc_df: pd.DataFrame, dollar_bar_size: float
) -> pd.DataFrame:
    """
    Creates raw dollar bars from OHLCV data using a Numba JIT compiled core.
    Focuses on 'Open', 'High', 'Low', 'Close', 'Volume'.
    Includes 'Open time' and 'Close time' for each bar.
    All timestamps are in UTC.
    """
    if not isinstance(ohlc_df, pd.DataFrame) or ohlc_df.empty:
        cols = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
        return pd.DataFrame(columns=cols)

    if dollar_bar_size <= 0:
        raise ValueError("dollar_bar_size must be positive.")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in ohlc_df.columns:
            raise ValueError(f"ohlc_df must contain column: {col}")

    has_datetime_index = isinstance(ohlc_df.index, pd.DatetimeIndex)
    has_open_time_col = "Open time" in ohlc_df.columns

    if not has_datetime_index and not has_open_time_col:
        raise ValueError(
            "ohlc_df must have a DatetimeIndex or an 'Open time' column to determine bar timestamps."
        )

    # Convert data to numpy arrays
    df_copy = ohlc_df.copy()
    dollar_value_np = (df_copy["Close"].values * df_copy["Volume"].values).astype(
        np.float64
    )
    open_np = df_copy["Open"].values.astype(np.float64)
    high_np = df_copy["High"].values.astype(np.float64)
    low_np = df_copy["Low"].values.astype(np.float64)
    close_np = df_copy["Close"].values.astype(np.float64)
    volume_np = df_copy["Volume"].values.astype(np.float64)

    # Get timestamps (we'll use these later)
    if has_datetime_index:
        timestamps = df_copy.index
    else:
        timestamps = pd.to_datetime(df_copy["Open time"])

    # Run Numba function
    (
        start_indices,
        end_indices,
        out_open,
        out_high,
        out_low,
        out_close,
        out_volume,
    ) = _calculate_dollar_bars_nb_core(
        open_np,
        high_np,
        low_np,
        close_np,
        volume_np,
        dollar_value_np,
        dollar_bar_size,
    )

    if len(out_open) == 0:
        cols = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
        return pd.DataFrame(columns=cols)

    # Create DataFrame with the results
    open_times = list(timestamps[start_indices])
    close_times = list(timestamps[end_indices])
    # Adjust close times for continuity
    for i in range(len(close_times) - 1):
        close_times[i] = open_times[i + 1] - pd.Timedelta(microseconds=1)
    if len(close_times) > 0:
        close_times[-1] = close_times[-1] + pd.Timedelta(seconds=59.999)

    raw_bars_df = pd.DataFrame(
        {
            "Open time": open_times,
            "Open": out_open,
            "High": out_high,
            "Low": out_low,
            "Close": out_close,
            "Volume": out_volume,
            "Close time": close_times,
        }
    )

    # Ensure standard column order
    raw_bars_df = raw_bars_df[
        ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
    ]
    return raw_bars_df


def data_profiler(ohlc_df: pd.DataFrame) -> dict:
    """
    Analyze OHLCV data to help users choose appropriate dollar bar sizes.
    Returns a dictionary with key statistics about the data.
    """
    if not isinstance(ohlc_df, pd.DataFrame) or ohlc_df.empty:
        return {
            "error": "Empty or invalid DataFrame provided",
            "recommendations": {
                "warnings": ["Empty DataFrame provided"],
                "suggested_dollar_bar_sizes": {
                    "conservative": 0.0,
                    "moderate": 0.0,
                    "aggressive": 0.0,
                },
            },
        }

    # Calculate dollar volumes
    dollar_volumes = ohlc_df["Close"] * ohlc_df["Volume"]

    # Calculate key statistics
    stats = {
        "total_rows": len(ohlc_df),
        "date_range": {
            "start": ohlc_df.index.min(),
            "end": ohlc_df.index.max(),
            "days": (ohlc_df.index.max() - ohlc_df.index.min()).days,
        },
        "price_stats": {
            "mean": ohlc_df["Close"].mean(),
            "median": ohlc_df["Close"].median(),
            "min": ohlc_df["Close"].min(),
            "max": ohlc_df["Close"].max(),
            "std": ohlc_df["Close"].std(),
        },
        "volume_stats": {
            "mean": ohlc_df["Volume"].mean(),
            "median": ohlc_df["Volume"].median(),
            "min": ohlc_df["Volume"].min(),
            "max": ohlc_df["Volume"].max(),
            "std": ohlc_df["Volume"].std(),
        },
        "dollar_volume_stats": {
            "mean": dollar_volumes.mean(),
            "median": dollar_volumes.median(),
            "min": dollar_volumes.min(),
            "max": dollar_volumes.max(),
            "std": dollar_volumes.std(),
            "percentiles": {
                "50": dollar_volumes.quantile(0.50),
                "75": dollar_volumes.quantile(0.75),
                "90": dollar_volumes.quantile(0.90),
                "95": dollar_volumes.quantile(0.95),
                "99": dollar_volumes.quantile(0.99),
            },
        },
        "high_volume_analysis": {
            "over_100M": (dollar_volumes > 100_000_000).sum(),
            "over_500M": (dollar_volumes > 500_000_000).sum(),
            "over_1B": (dollar_volumes > 1_000_000_000).sum(),
        },
    }

    # Add recommendations
    stats["recommendations"] = {
        "suggested_dollar_bar_sizes": {
            "conservative": stats["dollar_volume_stats"]["percentiles"]["75"],
            "moderate": stats["dollar_volume_stats"]["percentiles"]["90"],
            "aggressive": stats["dollar_volume_stats"]["percentiles"]["95"],
        },
        "warnings": [],
    }

    # Add warnings if needed
    if stats["high_volume_analysis"]["over_100M"] > 0:
        stats["recommendations"]["warnings"].append(
            f"Found {stats['high_volume_analysis']['over_100M']:,} minutes with volume > $100M. "
            "Consider using a larger dollar bar size or splitting high-volume minutes."
        )
    if stats["high_volume_analysis"]["over_500M"] > 0:
        stats["recommendations"]["warnings"].append(
            f"Found {stats['high_volume_analysis']['over_500M']:,} minutes with volume > $500M. "
            "These extreme volumes may affect dollar bar consistency."
        )

    return stats


def describe_dollar_bars(dollar_bars_df: pd.DataFrame, dollar_bar_size: float) -> dict:
    """
    Analyze the results of dollar bar creation.
    Returns a dictionary with key statistics about the created bars.
    """
    if not isinstance(dollar_bars_df, pd.DataFrame) or dollar_bars_df.empty:
        return {"error": "Empty or invalid DataFrame provided"}

    # Determine prefix if columns are prefixed using simplified number
    prefix_val = _simplify_number(dollar_bar_size)
    prefix = f"db_{prefix_val}_"

    # Check if we have prefixed columns
    has_prefix = any(col.startswith(prefix) for col in dollar_bars_df.columns)

    # Get the correct column names based on whether they're prefixed
    if has_prefix:
        open_time_col = f"{prefix}Open time"
        close_time_col = f"{prefix}Close time"
        close_col = f"{prefix}Close"
        volume_col = f"{prefix}Volume"
    else:
        open_time_col = "Open time"
        close_time_col = "Close time"
        close_col = "Close"
        volume_col = "Volume"

    # Calculate dollar values for each bar
    bar_dollar_values = dollar_bars_df[close_col] * dollar_bars_df[volume_col]

    # Calculate deviations from target size
    deviations = (bar_dollar_values - dollar_bar_size) / dollar_bar_size * 100

    stats = {
        "total_bars": len(dollar_bars_df),
        "date_range": {
            "start": dollar_bars_df[open_time_col].min(),
            "end": dollar_bars_df[close_time_col].max(),
            "days": (
                dollar_bars_df[close_time_col].max()
                - dollar_bars_df[open_time_col].min()
            ).days,
        },
        "dollar_value_stats": {
            "mean": bar_dollar_values.mean(),
            "median": bar_dollar_values.median(),
            "min": bar_dollar_values.min(),
            "max": bar_dollar_values.max(),
            "std": bar_dollar_values.std(),
        },
        "deviation_stats": {
            "mean": deviations.mean(),
            "median": deviations.median(),
            "min": deviations.min(),
            "max": deviations.max(),
            "std": deviations.std(),
        },
        "bar_duration_stats": {
            "mean_minutes": (
                dollar_bars_df[close_time_col] - dollar_bars_df[open_time_col]
            )
            .mean()
            .total_seconds()
            / 60,
            "median_minutes": (
                dollar_bars_df[close_time_col] - dollar_bars_df[open_time_col]
            )
            .median()
            .total_seconds()
            / 60,
            "min_minutes": (
                dollar_bars_df[close_time_col] - dollar_bars_df[open_time_col]
            )
            .min()
            .total_seconds()
            / 60,
            "max_minutes": (
                dollar_bars_df[close_time_col] - dollar_bars_df[open_time_col]
            )
            .max()
            .total_seconds()
            / 60,
        },
    }

    # Add warnings if needed
    stats["warnings"] = []
    if abs(stats["deviation_stats"]["max"]) > 50:
        stats["warnings"].append(
            f"Maximum deviation from target size is {stats['deviation_stats']['max']:.1f}%. "
            "Consider using a different dollar bar size."
        )
    if stats["bar_duration_stats"]["max_minutes"] > 60:
        stats["warnings"].append(
            f"Some bars span more than an hour ({stats['bar_duration_stats']['max_minutes']:.1f} minutes). "
            "This might affect your analysis."
        )

    return stats


def generate_dollar_bars(
    ohlc_df: pd.DataFrame,
    dollar_bar_size: float,
    return_aligned_to_original_time: bool = True,
    use_numba: bool = False,
) -> pd.DataFrame:
    """
    Generates dollar bars from OHLCV data.

    Args:
        ohlc_df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
                                Must have a pd.DatetimeIndex or an 'Open time' column.
                                If return_aligned_to_original_time is True, ohlc_df must have a
                                pd.DatetimeIndex for the alignment step.
                                All timestamps are assumed to be in UTC.
        dollar_bar_size (float): The target dollar value for each bar.
        return_aligned_to_original_time (bool, optional):
            If True (default), returns a DataFrame aligned with the original ohlc_df's index,
            with dollar bar data forward-filled. This requires ohlc_df to have a DatetimeIndex.
            If False, returns the raw dollar bars DataFrame.
        use_numba (bool, optional):
            If True, uses the Numba-optimized version for creating raw dollar bars. Defaults to False.

    Returns:
        pd.DataFrame: Dollar bars data, either raw or aligned and forward-filled.
        All timestamps in the returned DataFrame will be in UTC.
    """
    # Normalize column names first
    ohlc_df = _normalize_column_names(ohlc_df)

    # Check for all zero values
    if (ohlc_df["Close"] * ohlc_df["Volume"]).sum() == 0:
        cols = ["Open time", "Open", "High", "Low", "Close", "Volume", "Close time"]
        return pd.DataFrame(columns=cols)

    # Profile the data and print warnings if needed
    profile = data_profiler(ohlc_df)
    if "warnings" in profile["recommendations"]:
        for warning in profile["recommendations"]["warnings"]:
            print(f"\nWarning: {warning}")

    if return_aligned_to_original_time and not isinstance(
        ohlc_df.index, pd.DatetimeIndex
    ):
        # Try to set index if 'Open time' column exists and is Datetime compatible
        if "Open time" in ohlc_df.columns:
            try:
                # Make a copy before modifying
                temp_ohlc_df = ohlc_df.copy()
                # Convert to UTC when setting index
                temp_ohlc_df = temp_ohlc_df.set_index(
                    pd.to_datetime(temp_ohlc_df["Open time"]).dt.tz_localize("UTC")
                )
                print(
                    "Used 'Open time' column to set UTC DatetimeIndex on ohlc_df for alignment."
                )
                ohlc_df_for_alignment = temp_ohlc_df
            except Exception as e:
                raise ValueError(
                    "For alignment, ohlc_df must have a DatetimeIndex, or an 'Open time' column "
                    f"convertible to DatetimeIndex. Error during conversion: {e}"
                )
        else:
            raise ValueError(
                "For alignment, ohlc_df must have a DatetimeIndex. "
                "Alternatively, provide an 'Open time' column."
            )
    else:
        ohlc_df_for_alignment = ohlc_df

    # Ensure timezone consistency for alignment (convert to UTC)
    if return_aligned_to_original_time:
        ohlc_df_for_alignment = _ensure_timezone_consistency(ohlc_df_for_alignment)

    if use_numba:
        print("Using Numba-optimized version for raw dollar bar creation.")
        raw_bars = _create_raw_dollar_bars_nb(ohlc_df, dollar_bar_size)
    else:
        print("Using original Pandas version for raw dollar bar creation.")
        raw_bars = _create_raw_dollar_bars(ohlc_df, dollar_bar_size)

    if return_aligned_to_original_time:
        if raw_bars.empty:
            print(
                "Warning: No dollar bars were formed. Aligning an empty set of bars to the original timeseries."
            )
        # Ensure timezone consistency for raw bars before alignment (convert to UTC)
        raw_bars = _ensure_timezone_consistency(raw_bars)
        aligned_bars = _align_dollar_bars_to_timeseries(
            ohlc_df_for_alignment, raw_bars, dollar_bar_size
        )

        # Describe the results
        description = describe_dollar_bars(raw_bars, dollar_bar_size)
        if "warnings" in description:
            for warning in description["warnings"]:
                print(f"\nWarning: {warning}")

        return aligned_bars
    else:
        # Describe the results
        description = describe_dollar_bars(raw_bars, dollar_bar_size)
        if "warnings" in description:
            for warning in description["warnings"]:
                print(f"\nWarning: {warning}")
        # Set index to 'Open time' if present
        if "Open time" in raw_bars.columns:
            raw_bars = raw_bars.set_index("Open time")
        return raw_bars


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to handle case sensitivity.
    Specifically looks for 'Open time' or 'Open Time' and standardizes to 'Open time'.
    """
    df = df.copy()

    # Create a mapping of lowercase column names to their original names
    col_map = {col.lower(): col for col in df.columns}

    # Standard column names we want to normalize
    standard_cols = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "open time": "Open time",
        "close time": "Close time",
    }

    # Rename columns to standard format
    rename_map = {}
    for std_lower, std_name in standard_cols.items():
        if std_lower in col_map:
            original_name = col_map[std_lower]
            if original_name != std_name:
                rename_map[original_name] = std_name

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _ensure_timezone_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame index is in UTC timezone.
    - If timezone-naive, assume UTC
    - If timezone-aware, convert to UTC
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            # If naive, localize to UTC
            df.index = df.index.tz_localize("UTC")
        else:
            # If already has timezone, convert to UTC
            df.index = df.index.tz_convert("UTC")
    return df


def _ensure_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Ensure a timestamp is in UTC.
    - If timezone-naive, assume UTC
    - If timezone-aware, convert to UTC
    """
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _run_tests():
    """
    Run tests to verify dollar bar functionality.
    Tests both raw bar creation and alignment with various scenarios.
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    import pytest

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
        # We might get more bars than input rows if splitting rows for precise dollar values
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
        # Load real BTC data
        data_path = "/Users/ericervin/Documents/Coding/data-repository/data/archive/BTCUSDT_1m_futures.pkl"
        try:
            import vectorbtpro as vbt

            data_vbt = vbt.BinanceData.load(data_path)
            df = data_vbt.get()  # This should be your OHLCV DataFrame
            if (
                not isinstance(df.index, pd.DatetimeIndex)
                and "Open time" not in df.columns
            ):
                if "opentime" in df.columns:
                    df.rename(columns={"opentime": "Open time"}, inplace=True)
                elif df.index.name == "Open time" or df.index.name == "opentime":
                    df = df.reset_index()
            print(f"\nUsing real BTC data with {len(df):,} rows for testing")

            # Calculate and print dollar volume statistics for the raw data
            dollar_volumes = df["Close"] * df["Volume"]
            print("\nRaw Data Dollar Volume Statistics:")
            print(f"Mean dollar volume per minute: ${dollar_volumes.mean():,.2f}")
            print(f"Median dollar volume per minute: ${dollar_volumes.median():,.2f}")
            print(f"Min dollar volume per minute: ${dollar_volumes.min():,.2f}")
            print(f"Max dollar volume per minute: ${dollar_volumes.max():,.2f}")
            print(f"95th percentile: ${dollar_volumes.quantile(0.95):,.2f}")
            print(f"99th percentile: ${dollar_volumes.quantile(0.99):,.2f}")

            # Count minutes with extreme volume
            high_volume_minutes = (dollar_volumes > 100_000_000).sum()
            print(
                f"\nMinutes with >$100M volume: {high_volume_minutes:,} ({high_volume_minutes/len(df)*100:.1f}% of total)"
            )
            very_high_volume_minutes = (dollar_volumes > 500_000_000).sum()
            print(
                f"Minutes with >$500M volume: {very_high_volume_minutes:,} ({very_high_volume_minutes/len(df)*100:.2f}% of total)"
            )

        except Exception as e:
            print(f"\nFailed to load BTC data: {e}. Using synthetic data instead.")
            df = create_test_data(1000)

        # Use $100M as the target size for BTC
        dollar_bar_size = 100_000_000  # $100M bars for BTC

        raw_bars = generate_dollar_bars(
            df, dollar_bar_size, return_aligned_to_original_time=False, use_numba=True
        )

        # Calculate dollar value for each bar
        bar_dollar_values = raw_bars["Close"] * raw_bars["Volume"]

        # Print debugging information
        print("\nDollar Bar Size Test Debug Info:")
        print(f"Target dollar bar size: ${dollar_bar_size:,.2f}")
        print(f"Number of bars created: {len(raw_bars):,}")
        print(f"Average dollar value: ${bar_dollar_values.mean():,.2f}")
        print(f"Min dollar value: ${bar_dollar_values.min():,.2f}")
        print(f"Max dollar value: ${bar_dollar_values.max():,.2f}")
        print(f"Dollar value std dev: ${bar_dollar_values.std():,.2f}")

        # Print first few bars for inspection
        print("\nFirst few bars:")
        for i in range(min(5, len(raw_bars))):
            dv = bar_dollar_values.iloc[i]
            pct_diff = abs(dv - dollar_bar_size) / dollar_bar_size * 100
            print(f"Bar {i}: ${dv:,.2f} ({pct_diff:.1f}% from target)")

        # Test mathematical correctness
        print("\nTesting mathematical correctness...")

        # 1. Test that all bars have positive volume
        assert all(raw_bars["Volume"] > 0), "All bars should have positive volume"

        # 2. Test that High >= Low for all bars
        assert all(
            raw_bars["High"] >= raw_bars["Low"]
        ), "High should be >= Low for all bars"

        # 3. Test that Open and Close are within High-Low range
        assert all(raw_bars["Open"] <= raw_bars["High"]), "Open should be <= High"
        assert all(raw_bars["Open"] >= raw_bars["Low"]), "Open should be >= Low"
        assert all(raw_bars["Close"] <= raw_bars["High"]), "Close should be <= High"
        assert all(raw_bars["Close"] >= raw_bars["Low"]), "Close should be >= Low"

        # 4. Test that timestamps are in correct order
        assert all(
            raw_bars["Open time"] <= raw_bars["Close time"]
        ), "Open time should be <= Close time"

        # 5. Test that bars don't overlap (except for the last bar which might be partial)
        for i in range(len(raw_bars) - 1):
            assert (
                raw_bars["Close time"].iloc[i] <= raw_bars["Open time"].iloc[i + 1]
            ), f"Bar {i} ends after bar {i+1} starts"

        # 6. Test that volume is reasonably conserved
        # Note: Some volume loss is expected due to partial bars at the end of high-volume minutes
        total_input_volume = df["Volume"].sum()
        total_bar_volume = raw_bars["Volume"].sum()
        volume_diff_pct = (
            abs(total_input_volume - total_bar_volume) / total_input_volume * 100
        )
        print(f"\nVolume conservation check:")
        print(f"Total input volume: {total_input_volume:,.2f}")
        print(f"Total bar volume: {total_bar_volume:,.2f}")
        print(f"Difference: {volume_diff_pct:.2f}%")
        # Allow for volume loss due to partial bars
        assert (
            volume_diff_pct < 5.0
        ), "Volume should be reasonably conserved (within 5%)"

        # 7. Test that no data is lost in the process
        input_timespan = (df.index.max() - df.index.min()).total_seconds()
        bar_timespan = (
            raw_bars["Close time"].max() - raw_bars["Open time"].min()
        ).total_seconds()
        timespan_diff_pct = abs(input_timespan - bar_timespan) / input_timespan * 100
        print(f"\nTimespan conservation check:")
        print(f"Input timespan: {input_timespan/3600:.2f} hours")
        print(f"Bar timespan: {bar_timespan/3600:.2f} hours")
        print(f"Difference: {timespan_diff_pct:.2f}%")
        assert timespan_diff_pct < 0.01, "Timespan should be conserved within 0.01%"

        # 8. Test that dollar values are reasonable
        print("\nDollar value check:")
        print(f"Total input dollar value: ${dollar_volumes.sum():,.2f}")
        print(f"Total bar dollar value: ${bar_dollar_values.sum():,.2f}")
        dollar_value_diff_pct = (
            abs(dollar_volumes.sum() - bar_dollar_values.sum())
            / dollar_volumes.sum()
            * 100
        )
        print(f"Difference: {dollar_value_diff_pct:.2f}%")
        # Allow for dollar value loss due to partial bars
        assert (
            dollar_value_diff_pct < 5.0
        ), "Dollar values should be reasonably conserved (within 5%)"

        print("\nAll mathematical correctness tests passed! ✅")

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

        print("All alignment tests passed! ✅")

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

        print("All edge case tests passed! ✅")

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

    # Run all tests
    print("\nRunning tests...")
    test_basic_functionality()
    test_dollar_bar_size()
    test_alignment()
    test_edge_cases()
    test_consistency()
    print("All tests passed! ✅")


if __name__ == "__main__":
    print("Running dollar bars tests...")
    _run_tests()
    print("\nAll tests completed!")
