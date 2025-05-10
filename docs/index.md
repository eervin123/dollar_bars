# Dollar Bars Documentation

## Overview

Dollar Bars is a Python package for creating dollar bars from OHLCV (Open, High, Low, Close, Volume) data. Dollar bars are created by aggregating price-volume data until a specified dollar value is reached, providing a more consistent way to analyze financial time series data.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from dollar_bars import generate_dollar_bars, data_profiler
import pandas as pd

# Load your OHLCV data
df = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
}, index=pd.date_range('2024-01-01', periods=1000, freq='1min'))

# Profile your data to choose an appropriate bar size
profile = data_profiler(df)
print(f"Suggested bar sizes:")
print(f"Conservative: ${profile['recommendations']['suggested_dollar_bar_sizes']['conservative']:,.2f}")
print(f"Moderate: ${profile['recommendations']['suggested_dollar_bar_sizes']['moderate']:,.2f}")
print(f"Aggressive: ${profile['recommendations']['suggested_dollar_bar_sizes']['aggressive']:,.2f}")

# Create dollar bars
dollar_bars = generate_dollar_bars(
    df,
    dollar_bar_size=1_000_000,  # $1M bars
    return_aligned_to_original_time=True,  # Align to original timestamps
    use_numba=True  # Use Numba optimization
)
```

## API Reference

### generate_dollar_bars

```python
generate_dollar_bars(
    ohlc_df: pd.DataFrame,
    dollar_bar_size: float,
    return_aligned_to_original_time: bool = True,
    use_numba: bool = False
) -> pd.DataFrame
```

Creates dollar bars from OHLCV data.

#### Parameters:
- `ohlc_df`: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
- `dollar_bar_size`: Target dollar value for each bar
- `return_aligned_to_original_time`: If True, returns aligned bars
- `use_numba`: If True, uses Numba optimization

### data_profiler

```python
data_profiler(ohlc_df: pd.DataFrame) -> dict
```

Analyzes OHLCV data to help choose appropriate dollar bar sizes.

### describe_dollar_bars

```python
describe_dollar_bars(dollar_bars_df: pd.DataFrame, dollar_bar_size: float) -> dict
```

Generates descriptive statistics for dollar bars.

## Examples

See the [examples](examples.md) page for more detailed usage examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 