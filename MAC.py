import yfinance as yf
import matplotlib.pyplot as plt

from metrics import calculate_metrics, print_metrics


def MACD(df):
    df['EMA12'] = df.Close.ewm(span=12).mean()
    df['EMA26'] = df.Close.ewm(span=26).mean()
    df['MACD'] = df.EMA12 - df.EMA26
    df['signal_line'] = df.MACD.ewm(span=9).mean() # the MACD indicator's signal line, not the BUY/SELL/HOLD trade signal
    print('Indicators added')


def macd_signals(df):
    """Returns (buy_dates, sell_dates): the next-day execution dates, mirroring
    RSI's getSignals() shape so both strategies feed BacktestEngine identically."""
    Buy, Sell = [], []
    for i in range(2, len(df)):
        if df.MACD.iloc[i] > df.signal_line.iloc[i] and df.MACD.iloc[i-1] < df.signal_line.iloc[i-1]:
            Buy.append(i)
        elif df.MACD.iloc[i] < df.signal_line.iloc[i] and df.MACD.iloc[i-1] > df.signal_line.iloc[i-1]:
            Sell.append(i)

    Realbuys = [i+1 for i in Buy if i+1 < len(df)]
    Realsells = [i+1 for i in Sell if i+1 < len(df)]
    return list(df.iloc[Realbuys].index), list(df.iloc[Realsells].index)


if __name__ == "__main__":
    # deferred: backtest_engine.py imports MACD/macd_signals from this file, so importing
    # it back at module level here would be circular. Deferring avoids that.
    from backtest_engine import BacktestEngine, get_sp500_tickers, prompt_for_ticker

    ticker = prompt_for_ticker(get_sp500_tickers())

    df = yf.download(ticker, start='2016-01-01', auto_adjust=False, multi_level_index=False)
    if df.empty:
        raise SystemExit(f"No price data returned for {ticker}.")
    # Standardize on the same price basis as RSI.py (split/dividend-adjusted): downloading
    # with auto_adjust=False gives an explicit 'Adj Close' column, then aliased onto 'Close' so
    # everything below (MACD(), macd_signals(), plotting) keeps working unchanged. Previously
    # this used the implicit yfinance default (auto_adjust=True, whose 'Close' happens to equal
    # 'Adj Close') - numerically identical today, but relying on an unstated library default to
    # keep two strategies on the same price series was fragile, not a deliberate, explained choice.
    df['Close'] = df['Adj Close']

    MACD(df)
    buy_dates, sell_dates = macd_signals(df)
    if not buy_dates or not sell_dates:
        raise SystemExit(f"No MACD buy/sell signals were generated for {ticker} over this period.")

    # translate the MACD crossover dates into the (date, close, signal) shape
    # BacktestEngine expects, so this runs through the same Portfolio as every other strategy
    price_data = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'close'})
    price_data['signal'] = 'HOLD'
    price_data.loc[price_data['date'].isin(buy_dates), 'signal'] = 'BUY'
    price_data.loc[price_data['date'].isin(sell_dates), 'signal'] = 'SELL'

    engine = BacktestEngine(price_data, initial_capital=10000)
    portfolio = engine.run()

    print_metrics(calculate_metrics(portfolio.get_trades(), initial_capital=10000), label=f"MACD Strategy - {ticker}")

    plt.figure(figsize=(12,5))
    plt.plot(df.MACD, label='MACD', color='green')
    plt.plot(df.signal_line, label='Signal', color='red')
    plt.legend()
    plt.title(f"MACD indicator - {ticker}")

    plt.figure(figsize=(12,4))
    plt.scatter(df.loc[buy_dates].index, df.loc[buy_dates].Close, marker="^", color='green', label='Buy signal')
    plt.scatter(df.loc[sell_dates].index, df.loc[sell_dates].Close, marker="v", color='red', label='Sell signal')
    plt.plot(df.Close, label=f'{ticker} Close', color='k')
    plt.legend()
    plt.title(f"MACD Strategy - {ticker}")
    plt.show()
