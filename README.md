# Dollar Bars

A Python package for creating dollar bars from OHLCV (Open, High, Low, Close, Volume) data. Dollar bars are created by aggregating price-volume data until a specified dollar value is reached, providing a more consistent way to analyze financial time series data.

## Features

- Create dollar bars from OHLCV data
- Numba-optimized implementation for high performance
- Timezone-aware processing
- Comprehensive data profiling and bar analysis
- Support for both raw bars and time-aligned bars

## Installation

You can install the package using pip:

```bash
pip install dollar-bars
```

Or install from source:

```bash
git clone https://github.com/yourusername/dollar-bars.git
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

When `return_aligned_to_original_time=True`, the output DataFrame will have columns prefixed with `db_[size]_`, where `[size]` is your dollar bar size. For example:

- `db_1000000_Open`: Open price for the $1M dollar bar
- `db_1000000_High`: High price
- `db_1000000_Low`: Low price
- `db_1000000_Close`: Close price
- `db_1000000_Volume`: Volume
- `db_1000000_Open Time`: Bar start time
- `db_1000000_Close Time`: Bar end time
- `db_1000000_NewDBFlag`: True when a new dollar bar starts

## Important Notes

1. Dollar Bar Behavior:
   - A minute with high dollar volume (e.g., $500M) will create a single $500M dollar bar
   - If accumulated value is $99M and next minute has $20M, it creates a $119M bar
   - Zero-value minutes are skipped
   - Minutes are never split (respecting time boundaries)

2. Performance:
   - The Numba-optimized version is significantly faster
   - Processing 1.4M rows takes ~0.23 seconds
   - With alignment, processing takes ~0.36 seconds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 