import pandas as pd # for number handling
import yfinance as yf # for extracting live market data
import matplotlib.pyplot as plt

from metrics import calculate_metrics, print_metrics

pd.options.mode.chained_assignment = None

# RSI smoothing span - uncomment exactly one, comment out the rest
# RSI_SPAN = 14  # Wilder-equivalent
# RSI_SPAN = 17
RSI_SPAN = 19  # baseline
# RSI_SPAN = 21
# RSI_SPAN = 25

# RSI entry/exit thresholds (oversold-entry / recovery-exit) - uncomment exactly one pair, comment out the rest
# RSI_ENTRY, RSI_EXIT = 20, 35
# RSI_ENTRY, RSI_EXIT = 25, 40
RSI_ENTRY, RSI_EXIT = 30, 40  # baseline
# RSI_ENTRY, RSI_EXIT = 35, 45
# RSI_ENTRY, RSI_EXIT = 30, 50


def RSIcalc(asset):
    try:
        df = yf.download(asset, start='2016-01-01', auto_adjust=False, multi_level_index=False)
    except Exception:
        return None
    if df.empty or len(df) < 200:
        return None
    df['MA200'] = df['Adj Close'].rolling(window=200).mean() # MA 200
    df['price change'] = df['Adj Close'].pct_change() # relative returns
    df['Upmove'] = df['price change'].apply(lambda x:x if x>0 else 0) # upmove trend
    df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x<0 else 0) # downmove
    df['avg Up'] = df['Upmove'].ewm(span=RSI_SPAN).mean()
    df['avg Down'] = df['Downmove'].ewm(span=RSI_SPAN).mean()
    df = df.dropna()
    if df.empty:
        return None
    df['RS'] = df['avg Up']/df['avg Down']
    df['RSI'] = df['RS'].apply(lambda x: 100-(100/(x+1)))
    df.loc[(df['Adj Close'] > df['MA200']) & (df['RSI'] <RSI_ENTRY), 'Buy']= 'Yes' # mentioning when to buy
    df.loc[(df['Adj Close'] < df['MA200']) | (df['RSI'] >RSI_ENTRY), 'Buy']= 'No' # mentioning when not to buy
    return df


def getSignals(df):
    Buying_dates = []
    Selling_dates = []
    for i in range(len(df)-11):
        if "Yes" in df['Buy'].iloc[i]:
            Buying_dates.append(df.iloc[i+1].name)
            for j in range(1,11):
                if df['RSI'].iloc[i + j] >RSI_EXIT:
                    Selling_dates.append(df.iloc[i+j+1].name)
                    break
                elif j == 10:
                    Selling_dates.append(df.iloc[i+j+1].name)
    return Buying_dates, Selling_dates


if __name__ == "__main__":
    # deferred: backtest_engine.py imports RSIcalc/getSignals from this file, so importing
    # it back at module level here would be circular. Deferring avoids that.
    from backtest_engine import BacktestEngine, get_sp500_tickers, prompt_for_ticker

    ticker = prompt_for_ticker(get_sp500_tickers())

    frame = RSIcalc(ticker)
    if frame is None:
        raise SystemExit(f"Not enough price history to run the RSI strategy on {ticker}.")

    buy, sell = getSignals(frame)
    if not buy or not sell:
        raise SystemExit(f"No RSI buy/sell signals were generated for {ticker} over this period.")

    # translate the RSI strategy's buy/sell dates into the (date, close, signal) shape
    # BacktestEngine expects, so this runs through the same Portfolio as every other strategy
    price_data = frame.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'date', 'Adj Close': 'close'})
    price_data['signal'] = 'HOLD'
    price_data.loc[price_data['date'].isin(buy), 'signal'] = 'BUY'
    price_data.loc[price_data['date'].isin(sell), 'signal'] = 'SELL'

    engine = BacktestEngine(price_data, initial_capital=10000)
    portfolio = engine.run()

    print_metrics(calculate_metrics(portfolio.get_trades(), initial_capital=10000),
                  label=f"RSI Strategy (span={RSI_SPAN}, entry/exit={RSI_ENTRY}/{RSI_EXIT}) - {ticker}")

    plt.figure(figsize=(12,5))
    plt.scatter(frame.loc[buy].index, frame.loc[buy]['Adj Close'], marker='^', c='g', label='Buy')
    plt.scatter(frame.loc[sell].index, frame.loc[sell]['Adj Close'], marker='v', c='r', label='Sell')
    plt.plot(frame['Adj Close'], alpha=0.7, label='Adj Close')
    plt.title(f"RSI Strategy (span={RSI_SPAN}, entry/exit={RSI_ENTRY}/{RSI_EXIT}) - {ticker}")
    plt.legend()
    plt.show()
