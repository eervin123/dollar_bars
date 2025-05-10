import pandas as pd
import vectorbtpro as vbt
from dollar_bars import generate_dollar_bars


# Load BTC data (adjust path as needed)
def load_btc_data():
    data_path = "/Users/ericervin/Documents/Coding/data-repository/data/archive/BTCUSDT_1m_futures.pkl"
    data_vbt = vbt.BinanceData.load(data_path)
    df = data_vbt.get()
    print(f"Loaded BTC data with {len(df):,} rows")
    return df


def main():
    df = load_btc_data()
    dollar_bar_size = 100_000_000  # $100M
    # Use the aligned DataFrame (no lookahead bias)
    aligned_df = generate_dollar_bars(
        df,
        dollar_bar_size=dollar_bar_size,
        use_numba=True,
        return_aligned_to_original_time=True,
    )
    prefix = f"db_{int(dollar_bar_size//1_000_000)}M_"
    time_slice = slice("2021-01-09 18:00", "2021-01-09 20:00")
    pd.set_option("display.max_rows", 50)
    print(f"\nSample mapping for time slice {time_slice.start} to {time_slice.stop}:")
    print(
        aligned_df.loc[
            time_slice,
            [
                "Close",
                f"{prefix}Open",
                f"{prefix}Close",
                f"{prefix}Open time",
                f"{prefix}Close time",
            ],
        ]
    )
    # Show more rows around a bar transition for inspection
    transition_time = pd.Timestamp("2021-01-09 18:03:00+00:00")
    window = aligned_df.loc[
        transition_time
        - pd.Timedelta(minutes=5) : transition_time
        + pd.Timedelta(minutes=5)
    ]
    print(
        window[
            [
                "Close",
                f"{prefix}Open",
                f"{prefix}Close",
                f"{prefix}Open time",
                f"{prefix}Close time",
            ]
        ]
    )
    # Plot using vectorbtpro's vbt.plot()
    fig = (
        aligned_df["Close"]
        .loc[time_slice]
        .vbt.plot(
            trace_kwargs=dict(
                name="Minutely Close", line=dict(color="purple", width=1)
            ),
            title=f"Minutely vs Dollar Bar Alignment\n{time_slice.start} to {time_slice.stop}",
            xaxis_title="Time",
            yaxis_title="Price",
        )
    )
    fig.add_scatter(
        x=aligned_df.loc[time_slice].index,
        y=aligned_df[f"{prefix}Close"].loc[time_slice],
        mode="lines",
        name="Dollar Bar Close (step)",
        line=dict(color="brown", width=2, shape="hv"),
        connectgaps=False,
    )
    # Find where the minutely bar's Close time matches the dollar bar's Close time
    close_times = aligned_df["Close time"].loc[time_slice]
    db_close_times = aligned_df[f"{prefix}Close time"].loc[time_slice]
    marker_mask = close_times == db_close_times

    fig.add_scatter(
        x=aligned_df.loc[time_slice].index[marker_mask],
        y=aligned_df[f"{prefix}Close"].loc[time_slice][marker_mask],
        mode="markers",
        name="Dollar Bar Close (on minutely grid)",
        marker=dict(color="red", symbol="x", size=10),
    )
    fig.show()


if __name__ == "__main__":
    main()
