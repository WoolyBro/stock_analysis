""" This code runs the strategies against price data, day by day, through a Portfolio.

    Assumptions made:
    - Execution price is that day's close (not next-day open)
      since real-world you can't react instantly to a signal seen at close.
    - A BUY while already holding, or a SELL while flat, is ignored. Trade once executed is finalized.
    - Any position still open at the end of the data is force-closed at the final
      day's close, so the last trade isn't left dangling.
"""

import pandas as pd
import requests
import yfinance as yf
from io import StringIO

from metrics import calculate_metrics, Trade
from portfolio import Portfolio


class BacktestEngine:
    def __init__(self, price_data: pd.DataFrame, initial_capital: float = 10000, transaction_cost: float = None):
        # transaction_cost variable has been placed in portfolio.py
        self.price_data = price_data  # must have 'date', 'close', 'signal'
        self.portfolio = Portfolio(initial_capital) if transaction_cost is None else Portfolio(initial_capital, transaction_cost=transaction_cost)

    def run(self) -> Portfolio:
        holding_position = False

        for row in self.price_data.itertuples():
            date, price, signal = row.date, row.close, row.signal

            if signal == "BUY" and not holding_position:
                self.portfolio.buy(date, price)
                holding_position = True

            elif signal == "SELL" and holding_position:
                self.portfolio.sell(date, price)
                holding_position = False

            self.portfolio.update_equity(date, price)

        if holding_position:
            last_row = self.price_data.iloc[-1]
            self.portfolio.sell(last_row.date, last_row.close)

        return self.portfolio


def get_sp500_tickers():
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
    table = pd.read_html(StringIO(resp.text))[0]
    return [t.replace('.', '-') for t in table['Symbol'].tolist()]


def prompt_for_ticker(tickers):
    valid = {t.upper() for t in tickers}
    while True:
        choice = input("Enter an S&P 500 ticker (e.g. TSLA): ").strip().upper()
        if choice in valid:
            return choice
        print(f"'{choice}' is not one of the S&P 500 tickers. Try again.")


def prompt_for_scope():
    options = {"1": "index", "2": "stock"}
    while True:
        choice = input(
            "What do you want to analyze - "
            "1) The S&P 500 index over the last 10 years  "
            "2) A specific S&P 500 stock: "
        ).strip()
        if choice in options:
            return options[choice]
        print("Please enter 1 or 2.")


def print_strategy_report(name: str, trades: list, initial_capital: float, equity_curve: list = None):
    final_capital = initial_capital + sum(t.pnl for t in trades)

    print("\n" + "-" * 60)
    print(f"\nStrategy used: {name}\n")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}\n")

    if not trades:
        print("No trades were generated over this period.")
        return

    trades_df = pd.DataFrame([{
        'Entry Date': t.entry_date.date(),
        'Entry Price': round(float(t.entry_price), 2),
        'Exit Date': t.exit_date.date(),
        'Exit Price': round(float(t.exit_price), 2),
        'Quantity': round(float(t.quantity), 4),
        'Return %': round(t.return_pct * 100, 2),
    } for t in trades])
    print("Trades:")
    print(trades_df.to_string(index=False))

    metrics = calculate_metrics(trades, initial_capital=initial_capital, equity_curve=equity_curve)
    metrics_df = pd.DataFrame(
        [(k, f"{v:.2f}" if isinstance(v, float) else str(v)) for k, v in metrics.items()],
        columns=['Metric', 'Value'],
    )
    print("\nResults:")
    print(metrics_df.to_string(index=False))


def run_rsi(ticker: str, initial_capital: float = 10000, transaction_cost: float = None):
    """Returns of portfolio is None if RSI couldn't run (bad ticker, no
    signals, etc); otherwise it's the Portfolio after a full run - use .get_trades() and
    .equity_curve as needed."""
    from RSI import RSIcalc, getSignals

    frame = RSIcalc(ticker)
    if frame is None:
        return None, None
    buy, sell = getSignals(frame)
    if not buy or not sell:
        return frame, None
    price_data = frame.reset_index()[['Date', 'Adj Close']].rename(columns={'Date': 'date', 'Adj Close': 'close'})
    price_data['signal'] = 'HOLD'
    price_data.loc[price_data['date'].isin(buy), 'signal'] = 'BUY'
    price_data.loc[price_data['date'].isin(sell), 'signal'] = 'SELL'
    portfolio = BacktestEngine(price_data, initial_capital=initial_capital, transaction_cost=transaction_cost).run()
    return frame, portfolio


def run_macd(ticker: str, initial_capital: float = 10000, transaction_cost: float = None):
    """Returns (df, portfolio), same contract as run_rsi()."""
    from MAC import MACD, macd_signals

    try:
        df = yf.download(ticker, start='2016-01-01', auto_adjust=False, multi_level_index=False)
    except Exception:
        return None, None
    if df.empty:
        return None, None
    # same price-basis standardization as MAC.py's __main__ - see the comment there for why
    df['Close'] = df['Adj Close']
    MACD(df)
    buy, sell = macd_signals(df)
    if not buy or not sell:
        return df, None
    price_data = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'close'})
    price_data['signal'] = 'HOLD'
    price_data.loc[price_data['date'].isin(buy), 'signal'] = 'BUY'
    price_data.loc[price_data['date'].isin(sell), 'signal'] = 'SELL'
    portfolio = BacktestEngine(price_data, initial_capital=initial_capital, transaction_cost=transaction_cost).run()
    return df, portfolio


def buy_and_hold(price_series: pd.Series, initial_capital: float = 10000, transaction_cost: float = None):
    """price_series - Buys at the first day, holds to the last.
    Returns (trade, equity_curve) - a real day-by-day equity curve (not just start/end), so
    Max Drawdown/Volatility/Sharpe reflect the actual price path instead of looking like the
    instrument never dipped.

    transaction_cost: same round-trip cost concept as Portfolio.sell(), applied once since
    Buy & Hold is only ever one round trip. None (the default) defers to portfolio.py's
    TRANSACTION_COST toggle, same as everywhere else. Only the final entry of equity_curve is
    adjusted for it - the cost is a one-time exit friction, not a daily drag, so it shouldn't
    distort the day-by-day Volatility/Sharpe computed from this curve, just the final value.""" 
    if transaction_cost is None:
        from portfolio import TRANSACTION_COST as transaction_cost

    first_price = float(price_series.iloc[0])
    quantity = initial_capital / first_price
    exit_price = float(price_series.iloc[-1]) * (1 - transaction_cost)
    trade = Trade(
        entry_date=price_series.index[0],
        entry_price=first_price,
        exit_date=price_series.index[-1],
        exit_price=exit_price,
        quantity=quantity,
    )
    equity_curve = [(date, quantity * float(close)) for date, close in price_series.items()]
    if transaction_cost:
        last_date, _ = equity_curve[-1]
        equity_curve[-1] = (last_date, quantity * exit_price)
    return trade, equity_curve


def print_comparison_table(results: dict, initial_capital: float, ticker: str, equity_curves: dict = None):
    """results: {strategy_name: list[Trade]}, equity_curves: optional {strategy_name: list[(date, value)]}"""
    equity_curves = equity_curves or {}
    rows = []
    for name, trades in results.items():
        m = calculate_metrics(trades, initial_capital=initial_capital, equity_curve=equity_curves.get(name))
        rows.append({
            'Strategy': name,
            'CAGR (%)': m.get('CAGR (%)', float('nan')),
            'Win Rate (%)': m.get('Win Rate (%)', float('nan')),
            'Sharpe Ratio': m.get('Sharpe Ratio', float('nan')),
            'Max Drawdown (%)': m.get('Max Drawdown (%)', float('nan')),
        })
    comp_df = pd.DataFrame(rows)
    for col in comp_df.columns[1:]:
        comp_df[col] = comp_df[col].apply(lambda v: f"{v:.2f}" if isinstance(v, float) else v)

    print("\n" + "=" * 60)
    print(f"Comparison - {ticker}")
    print("=" * 60)
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    from RSI import RSI_SPAN, RSI_ENTRY, RSI_EXIT

    INITIAL_CAPITAL = 10000

    scope = prompt_for_scope()

    if scope == "index":
        ticker = "SPY"  # tracks the S&P 500 index, used here as the index's tradable proxy
        label = "S&P 500 Index (SPY)"
    else:
        ticker = prompt_for_ticker(get_sp500_tickers())
        label = ticker

    _, rsi_portfolio = run_rsi(ticker, INITIAL_CAPITAL)
    rsi_trades = rsi_portfolio.get_trades() if rsi_portfolio else []
    if not rsi_trades:
        print(f"No RSI (span={RSI_SPAN}, entry/exit={RSI_ENTRY}/{RSI_EXIT}) trades were generated for {label}.")

    df, macd_portfolio = run_macd(ticker, INITIAL_CAPITAL)
    if df is None:
        raise SystemExit(f"No price data returned for {label}.")
    macd_trades = macd_portfolio.get_trades() if macd_portfolio else []
    if not macd_trades:
        print(f"No MACD trades were generated for {label}.")

    bh_trade, bh_equity_curve = buy_and_hold(df['Close'], INITIAL_CAPITAL)

    all_results = {
        f"RSI (span={RSI_SPAN}, entry/exit={RSI_ENTRY}/{RSI_EXIT})": rsi_trades,
        "MACD": macd_trades,
        "Buy & Hold": [bh_trade],
    }
    equity_curves = {"Buy & Hold": bh_equity_curve}

    for name, trades in all_results.items():
        print_strategy_report(f"{name} on {label}", trades, INITIAL_CAPITAL, equity_curve=equity_curves.get(name))

    print_comparison_table(all_results, INITIAL_CAPITAL, label, equity_curves=equity_curves)
