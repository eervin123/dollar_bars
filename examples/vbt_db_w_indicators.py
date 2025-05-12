import pandas as pd
import vectorbtpro as vbt
from dollar_bars import generate_dollar_bars


# Step 1: Load BTC data (adjust path as needed)
def load_btc_data():
    data_path = "/Users/ericervin/Documents/Coding/data-repository/data/archive/BTCUSDT_1m_futures.pkl"
    data_vbt = vbt.BinanceData.load(data_path)
    df = data_vbt.get()
    print(f"Loaded BTC data with {len(df):,} rows")
    return df


def main():
    df = load_btc_data()
    # Step 2: Generate raw (unaligned) dollar bars
    dollar_bar_size = 1_000_000_000  # $100M
    dollarbar_df = generate_dollar_bars(
        df,
        dollar_bar_size=dollar_bar_size,
        use_numba=True,
        return_aligned_to_original_time=False,
    )
    print(f"Generated {len(dollarbar_df):,} raw dollar bars.")
    print(dollarbar_df.head())
    # Step 3: Create a vectorbt Data object from the dollar bars
    data = vbt.Data.from_data(dollarbar_df)
    print("Created vbt.Data object from dollar bars. Now creating BBands...")
    time_slice = slice("2021-01-01", "2021-01-20")
    bbands = data[time_slice].run("bbands")
    
    fig = bbands.plot(title="BTC BBands Based on 1B Dollar Bars")
    fig = df.Close.loc[time_slice].vbt.plot(
        title="BTC Close Price",
        fig=fig,
    )
    # Show dollar bars with a step line 
    fig = dollarbar_df.Close.loc[time_slice].vbt.plot(
        title="BTC Dollar BarClose Price",
        fig=fig,
        trace_kwargs=dict(line_shape="hv"),
    )
    fig.show()


if __name__ == "__main__":
    main()
