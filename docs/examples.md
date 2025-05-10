# Dollar Bars Examples

## Basic Usage

```python
import pandas as pd
from dollar_bars import generate_dollar_bars, data_profiler

# Create sample OHLCV data
df = pd.DataFrame({
    'Open': [100, 101, 102, 103, 104],
    'High': [102, 103, 104, 105, 106],
    'Low': [99, 100, 101, 102, 103],
    'Close': [101, 102, 103, 104, 105],
    'Volume': [1000, 2000, 1500, 3000, 2500]
}, index=pd.date_range('2024-01-01', periods=5, freq='1min'))

# Profile the data
profile = data_profiler(df)
print(f"Suggested bar sizes:")
print(f"Conservative: ${profile['recommendations']['suggested_dollar_bar_sizes']['conservative']:,.2f}")
print(f"Moderate: ${profile['recommendations']['suggested_dollar_bar_sizes']['moderate']:,.2f}")
print(f"Aggressive: ${profile['recommendations']['suggested_dollar_bar_sizes']['aggressive']:,.2f}")

# Create dollar bars
dollar_bars = generate_dollar_bars(
    df,
    dollar_bar_size=100_000,  # $100K bars
    return_aligned_to_original_time=True,
    use_numba=True
)
```

## Working with Real Data

```python
import pandas as pd
from dollar_bars import generate_dollar_bars, data_profiler, describe_dollar_bars

# Load your OHLCV data (example with CSV)
df = pd.read_csv('your_data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Profile the data to choose appropriate bar size
profile = data_profiler(df)
dollar_bar_size = profile['recommendations']['suggested_dollar_bar_sizes']['moderate']

# Create dollar bars
dollar_bars = generate_dollar_bars(
    df,
    dollar_bar_size=dollar_bar_size,
    return_aligned_to_original_time=True,
    use_numba=True
)

# Analyze the results
stats = describe_dollar_bars(dollar_bars, dollar_bar_size)
print(f"Number of bars: {stats['total_bars']}")
print(f"Average dollar value: ${stats['dollar_value_stats']['mean']:,.2f}")
print(f"Average bar duration: {stats['bar_duration_stats']['mean_minutes']:.1f} minutes")
```

## Handling Different Timezones

```python
import pandas as pd
from dollar_bars import generate_dollar_bars

# Create data with different timezone
df = pd.DataFrame({
    'Open': [100, 101, 102],
    'High': [102, 103, 104],
    'Low': [99, 100, 101],
    'Close': [101, 102, 103],
    'Volume': [1000, 2000, 1500]
}, index=pd.date_range('2024-01-01', periods=3, freq='1min', tz='America/New_York'))

# The package will automatically convert to UTC
dollar_bars = generate_dollar_bars(
    df,
    dollar_bar_size=100_000,
    return_aligned_to_original_time=True
)

# All timestamps in the result will be in UTC
print(dollar_bars.index.tz)  # UTC
```

## Using Numba Optimization

```python
import pandas as pd
from dollar_bars import generate_dollar_bars
import time

# Create large dataset
df = pd.DataFrame({
    'Open': np.random.normal(100, 1, 1000000),
    'High': np.random.normal(102, 1, 1000000),
    'Low': np.random.normal(98, 1, 1000000),
    'Close': np.random.normal(100, 1, 1000000),
    'Volume': np.random.gamma(2, 2, 1000000) * 1000
}, index=pd.date_range('2024-01-01', periods=1000000, freq='1min'))

# Compare performance
start = time.time()
bars_without_numba = generate_dollar_bars(df, 100_000, use_numba=False)
print(f"Without Numba: {time.time() - start:.2f} seconds")

start = time.time()
bars_with_numba = generate_dollar_bars(df, 100_000, use_numba=True)
print(f"With Numba: {time.time() - start:.2f} seconds")
```

## Working with Aligned Bars

```python
import pandas as pd
from dollar_bars import generate_dollar_bars

# Create sample data
df = pd.DataFrame({
    'Open': [100, 101, 102, 103, 104],
    'High': [102, 103, 104, 105, 106],
    'Low': [99, 100, 101, 102, 103],
    'Close': [101, 102, 103, 104, 105],
    'Volume': [1000, 2000, 1500, 3000, 2500]
}, index=pd.date_range('2024-01-01', periods=5, freq='1min'))

# Create aligned dollar bars
aligned_bars = generate_dollar_bars(
    df,
    dollar_bar_size=100_000,
    return_aligned_to_original_time=True
)

# The result will have the same index as the input
print(f"Input shape: {df.shape}")
print(f"Output shape: {aligned_bars.shape}")

# New dollar bars are marked with a flag
new_bar_col = [col for col in aligned_bars.columns if col.endswith('NewDBFlag')][0]
print(f"Number of new bars: {aligned_bars[new_bar_col].sum()}")
``` 