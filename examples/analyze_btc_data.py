"""
Example script demonstrating dollar bars with BTC data.
"""

import pandas as pd
import numpy as np
from dollar_bars import (
    generate_dollar_bars,
    data_profiler,
    describe_dollar_bars,
    _simplify_number,
)
import time
from scipy import stats
import vectorbtpro as vbt


def load_btc_data():
    """Load BTC data from your data repository."""
    data_path = "/Users/ericervin/Documents/Coding/data-repository/data/archive/BTCUSDT_1m_futures.pkl"
    try:
        data_vbt = vbt.BinanceData.load(data_path)
        df = data_vbt.get()

        # Ensure proper column names and index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "opentime" in df.columns:
                df.rename(columns={"opentime": "Open time"}, inplace=True)
            elif df.index.name == "Open time" or df.index.name == "opentime":
                df = df.reset_index()

        print(f"\nLoaded BTC data with {len(df):,} rows")
        return df
    except Exception as e:
        print(f"Error loading BTC data: {e}")
        return None


def calculate_returns(df, price_col):
    """Calculate log returns for a price series."""
    return np.log(df[price_col] / df[price_col].shift(1))


def analyze_distribution(returns, name):
    """Analyze the distribution of returns."""
    # Remove NaN values
    returns = returns.dropna()

    # Basic statistics
    stats_dict = {
        "mean": returns.mean(),
        "std": returns.std(),
        "skew": stats.skew(returns),
        "kurtosis": stats.kurtosis(returns),
        "jarque_bera_stat": stats.jarque_bera(returns)[0],
        "jarque_bera_pvalue": stats.jarque_bera(returns)[1],
        "normality_stat": stats.normaltest(returns)[0],
        "normality_pvalue": stats.normaltest(returns)[1],
    }

    # # Create histogram using vectorbt
    # hist = vbt.Histogram(
    #     returns,
    #     trace_kwargs=dict(name=name, histnorm="probability density", nbinsx=10),
    #     title=f"Return Distribution - {name}",
    #     xaxis_title="Log Returns",
    #     yaxis_title="Density",
    # )
    # hist.fig.show()

    return stats_dict


def compare_time_vs_dollar_bars(df, dollar_bar_size):
    """Compare the distribution of returns between time bars and dollar bars."""
    # Create time bars (using 1-minute bars as they are in the original data)
    time_bars = df.copy()
    time_returns = calculate_returns(time_bars, "Close")

    # Create dollar bars
    dollar_bars = generate_dollar_bars(
        df,
        dollar_bar_size=dollar_bar_size,
        return_aligned_to_original_time=True,
        use_numba=True,
    )

    # Get the correct column name for dollar bar close prices
    prefix_val = _simplify_number(dollar_bar_size)
    dollar_close_col = f"db_{prefix_val}_Close"
    dollar_returns = calculate_returns(dollar_bars, dollar_close_col)

    # Analyze distributions
    print("\nDistribution Analysis:")
    print("\nTime Bars Statistics:")
    time_stats = analyze_distribution(time_returns, "Time Bars")
    for stat, value in time_stats.items():
        print(f"{stat}: {value:.6f}")

    print("\nDollar Bars Statistics:")
    dollar_stats = analyze_distribution(
        dollar_returns, f"Dollar Bars (${dollar_bar_size:,.2f})"
    )
    for stat, value in dollar_stats.items():
        print(f"{stat}: {value:.6f}")

    # # Compare QQ plots using vectorbt
    # time_hist = vbt.Histogram(
    #     time_returns,
    #     trace_kwargs=dict(
    #         name="Time Bars Q-Q Plot", histnorm="probability density", nbinsx=10
    #     ),
    #     title="Time Bars Q-Q Plot",
        
    # )
    # time_hist.fig.show()

    # dollar_hist = vbt.Histogram(
    #     dollar_returns,
    #     trace_kwargs=dict(
    #         name="Dollar Bars Q-Q Plot", histnorm="probability density", nbinsx=10
    #     ),
    #     title="Dollar Bars Q-Q Plot",
    # )
    # dollar_hist.fig.show()


def analyze_dollar_bars(df, dollar_bar_size, use_numba=True):
    """Create and analyze dollar bars of a specific size."""
    start_time = time.time()

    print("\nInput DataFrame columns:", df.columns.tolist())

    # Create dollar bars
    dollar_bars = generate_dollar_bars(
        df,
        dollar_bar_size=dollar_bar_size,
        return_aligned_to_original_time=True,
        use_numba=use_numba,
    )

    print("\nDollar bars DataFrame columns:", dollar_bars.columns.tolist())

    # Get statistics
    stats = describe_dollar_bars(dollar_bars, dollar_bar_size)

    # Count actual number of dollar bars formed
    prefix_val = _simplify_number(dollar_bar_size)
    flag_col = f"db_{prefix_val}_NewDBFlag"
    actual_bars = dollar_bars[flag_col].sum()

    # Print results
    print(f"\nDollar Bar Size: ${dollar_bar_size:,.2f}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Total minutes in dataset: {len(df):,}")
    print(f"Actual dollar bars formed: {actual_bars:,}")
    print(f"Average dollar value: ${stats['dollar_value_stats']['mean']:,.2f}")
    print(
        f"Average bar duration: {stats['bar_duration_stats']['mean_minutes']:.1f} minutes"
    )
    print(f"Min bar duration: {stats['bar_duration_stats']['min_minutes']:.1f} minutes")
    print(f"Max bar duration: {stats['bar_duration_stats']['max_minutes']:.1f} minutes")

    if "warnings" in stats:
        for warning in stats["warnings"]:
            print(f"Warning: {warning}")

    return dollar_bars, stats


def main():
    # Load data
    df = load_btc_data()
    if df is None:
        return

    # Profile the data
    print("\nProfiling data...")
    profile = data_profiler(df)

    # Print data profile
    print("\nData Profile:")
    print(
        f"Date range: {profile['date_range']['start']} to {profile['date_range']['end']}"
    )
    print(f"Total days: {profile['date_range']['days']}")

    print("\nPrice Statistics:")
    print(f"Mean price: ${profile['price_stats']['mean']:,.2f}")
    print(
        f"Price range: ${profile['price_stats']['min']:,.2f} to ${profile['price_stats']['max']:,.2f}"
    )

    print("\nVolume Statistics:")
    print(f"Mean volume: {profile['volume_stats']['mean']:,.2f}")
    print(
        f"Volume range: {profile['volume_stats']['min']:,.2f} to {profile['volume_stats']['max']:,.2f}"
    )

    print("\nDollar Volume Statistics:")
    print(f"Mean dollar volume: ${profile['dollar_volume_stats']['mean']:,.2f}")
    print(f"Median dollar volume: ${profile['dollar_volume_stats']['median']:,.2f}")
    print(
        f"95th percentile: ${profile['dollar_volume_stats']['percentiles']['95']:,.2f}"
    )
    print(
        f"99th percentile: ${profile['dollar_volume_stats']['percentiles']['99']:,.2f}"
    )

    print("\nHigh Volume Analysis:")
    print(
        f"Minutes with >$100M volume: {profile['high_volume_analysis']['over_100M']:,}"
    )
    print(
        f"Minutes with >$500M volume: {profile['high_volume_analysis']['over_500M']:,}"
    )
    print(f"Minutes with >$1B volume: {profile['high_volume_analysis']['over_1B']:,}")

    # Calculate optimal dollar bar size for 150 bars per day
    total_days = profile["date_range"]["days"]
    total_dollar_volume = profile["dollar_volume_stats"]["mean"] * len(df)
    optimal_bar_size = total_dollar_volume / (total_days * 150)  # 150 bars per day

    # Calculate 25th percentile if not present
    if "25" not in profile["dollar_volume_stats"]["percentiles"]:
        profile["dollar_volume_stats"]["percentiles"]["25"] = (
            df["Close"] * df["Volume"]
        ).quantile(0.25)

    print("\nSuggested Dollar Bar Sizes:")
    print(
        f"25th percentile: ${profile['dollar_volume_stats']['percentiles']['25']:,.2f}"
    )
    print(
        f"50th percentile: ${profile['dollar_volume_stats']['percentiles']['50']:,.2f}"
    )
    print(
        f"75th percentile: ${profile['dollar_volume_stats']['percentiles']['75']:,.2f}"
    )
    print(
        f"90th percentile: ${profile['dollar_volume_stats']['percentiles']['90']:,.2f}"
    )
    print(
        f"95th percentile: ${profile['dollar_volume_stats']['percentiles']['95']:,.2f}"
    )
    print(f"Optimal (~150 bars/day): ${optimal_bar_size:,.2f}")

    if "warnings" in profile["recommendations"]:
        print("\nWarnings:")
        for warning in profile["recommendations"]["warnings"]:
            print(f"- {warning}")

    # Create dollar bars with different sizes
    print("\nCreating dollar bars with different sizes...")

    # Define all sizes to analyze
    sizes = {
        "25th percentile": profile["dollar_volume_stats"]["percentiles"]["25"],
        "50th percentile": profile["dollar_volume_stats"]["percentiles"]["50"],
        "75th percentile": profile["dollar_volume_stats"]["percentiles"]["75"],
        "90th percentile": profile["dollar_volume_stats"]["percentiles"]["90"],
        "95th percentile": profile["dollar_volume_stats"]["percentiles"]["95"],
        "Optimal": optimal_bar_size,
    }

    # Store results for comparison
    results = {}
    for name, size in sizes.items():
        print(f"\nAnalyzing {name}...")
        bars, stats = analyze_dollar_bars(df, size)
        results[name] = {
            "bars": bars,
            "stats": stats,
            "actual_bars": bars[f"db_{_simplify_number(size)}_NewDBFlag"].sum(),
        }

    # Compare results
    print("\nComparison of Different Dollar Bar Sizes:")
    print(
        f"{'Size':<20} {'Total Minutes':<15} {'Actual Bars':<15} {'Bars/Day':<15} {'Avg Duration':<15} {'Avg Value':<15}"
    )
    print("-" * 95)

    for name, result in results.items():
        bars_per_day = result["actual_bars"] / total_days
        print(
            f"{name:<20} {len(df):<15,} {result['actual_bars']:<15,} {bars_per_day:<15.1f} "
            f"{result['stats']['bar_duration_stats']['mean_minutes']:<15.1f} "
            f"${result['stats']['dollar_value_stats']['mean']:<15,.2f}"
        )

    # Compare time bars vs dollar bars distribution
    print("\nComparing Time Bars vs Dollar Bars Distribution...")
    compare_time_vs_dollar_bars(df, optimal_bar_size)


if __name__ == "__main__":
    main()
