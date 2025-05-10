# Dollar Bars

A Python package for creating dollar bars from OHLCV (Open, High, Low, Close, Volume) data. Dollar bars are created by aggregating price-volume data until a specified dollar value is reached, providing a more consistent way to analyze financial time series data.

## Features

- Create dollar bars from OHLCV data
- Numba-optimized implementation for high performance
- Timezone-aware processing
- Comprehensive data profiling and bar analysis
- Support for both raw bars and time-aligned bars

## Installation

You can install the package from source:

```bash
git clone https://github.com/eervin123/dollar-bars.git
cd dollar-bars
pip install -e .
```

## Quick Start

```python
from dollar_bars import generate_dollar_bars
import pandas as pd

# Load your OHLCV data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
}, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))

# Create dollar bars
dollar_bars = generate_dollar_bars(
    df,
    dollar_bar_size=100_000_000,  # $100M bars
    return_aligned_to_original_time=True,  # Align to original timestamps
    use_numba=True  # Use Numba optimization
)
```

## Usage Guide

### 1. Data Requirements

Your input DataFrame must have:

- Required columns: 'Open', 'High', 'Low', 'Close', 'Volume'
- Either a DatetimeIndex or an 'Open Time' column
- All timestamps should be in UTC

### 2. Creating Dollar Bars

The main function `generate_dollar_bars` has several options:

```python
from dollar_bars import generate_dollar_bars, data_profiler

# First, profile your data to choose an appropriate bar size
profile = data_profiler(df)
print(f"Suggested bar sizes:")
print(f"Conservative: ${profile['recommendations']['suggested_dollar_bar_sizes']['conservative']:,.2f}")
print(f"Moderate: ${profile['recommendations']['suggested_dollar_bar_sizes']['moderate']:,.2f}")
print(f"Aggressive: ${profile['recommendations']['suggested_dollar_bar_sizes']['aggressive']:,.2f}")

# Create dollar bars
dollar_bars = generate_dollar_bars(
    df,
    dollar_bar_size=1_000_000,
    return_aligned_to_original_time=True,
    use_numba=True
)
```

### 3. Analyzing Dollar Bars

```python
from dollar_bars import describe_dollar_bars

# Get statistics about your dollar bars
stats = describe_dollar_bars(dollar_bars, dollar_bar_size=1_000_000)
print(f"Number of bars: {stats['total_bars']}")
print(f"Average dollar value: ${stats['dollar_value_stats']['mean']:,.2f}")
print(f"Average bar duration: {stats['bar_duration_stats']['mean_minutes']:.1f} minutes")
```

### 4. Understanding the Output

When `return_aligned_to_original_time=True`, the output DataFrame will have columns prefixed with `db_[size]_`, where `[size]` is your dollar bar size, simplified and rounded for readability (e.g., `db_96M_` for a $96,000,000 bar). For example:

- `db_96M_Open`: Open price for the $96M dollar bar
- `db_96M_High`: High price
- `db_96M_Low`: Low price
- `db_96M_Close`: Close price
- `db_96M_Volume`: Volume
- `db_96M_Open time`: Bar start time (first included minute)
- `db_96M_Close time`: Bar end time (last included minute, or the open time of the next bar minus one microsecond)
- `db_96M_NewDBFlag`: True when a new dollar bar starts

When `return_aligned_to_original_time=False`, the returned DataFrame is indexed by `Open time` and contains only the actual dollar bars (no forward-filling). This is ideal for event-based analysis and indicator application.

### Column Prefix Simplification

Column prefixes use the `_simplify_number` logic, which rounds the bar size to the nearest integer and adds a suffix (K, M, B). For example:
- 95,535,298.67 becomes `96M`
- 1,234,567,890 becomes `1B`
- 12,345 becomes `12K`

This makes the output much more readable and user-friendly.

## Profiling and Analysis

- Use `data_profiler(df)` to get a profile of your OHLCV data, including suggested dollar bar sizes and volume statistics.
- Use `describe_dollar_bars(dollar_bars, dollar_bar_size)` to get detailed statistics about your generated dollar bars, such as average value, duration, and deviation from target size.

## Example Scripts

See the `examples/` directory for practical usage:
- `analyze_btc_data.py`: End-to-end profiling, bar creation, and statistical analysis.
- `vbt_db_w_indicators.py`: Shows how to create dollar bars and apply vectorbt indicators (like BBands) on top of them, including step plotting for event-based bars.

These examples demonstrate best practices for using the package in research and production workflows.

## Important Notes

1. Dollar Bar Behavior:
   - A minute with high dollar volume (e.g., $500M) will create a single $500M dollar bar
   - If accumulated value is $99M and next minute has $20M, it creates a $119M bar
   - Zero-value minutes are skipped
   - Minutes are never split (respecting time boundaries)

2. Performance:
   - Standard 1min dataframe ~1.4mm rows takes over 90 seconds without numba.
   - The Numba-optimized version is significantly faster
   - Processing 1.4M rows takes ~0.23 seconds
   - With alignment, processing takes ~0.36 seconds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 