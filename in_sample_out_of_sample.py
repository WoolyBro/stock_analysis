"""In-sample vs. out-of-sample split: Parameters are frozen
using only 2016-2020 data (in-sample) and then applied, unmodified, to 2021-present data not
used in that selection (out-of-sample).

In this project RSI's/MACD's parameters (span, entry/exit thresholds, 12/26/9) are fixed
conventional constants, not fitted/optimized from data at all - so there is no actual "freezing" 
step to perform here. What this checks instead is whether the SAME already-fixed strategy's 
performance holds up on 2021-present data versus its 2016-2020 performance: the standard proxy 
for "is this curve-fit to one period" when there was never an explicit fitting step to begin with.

it has same reasoning as regime_analysis.py/rolling_window_stability.py: a robustness check
layered on top of already-computed equity curves. It deliberately does NOT re-download or
re-run each strategy separately per period with a truncated start date - RSI needs its 200-day
MA and MACD its 26-day EMA to warm up and a real trader on 2021-01-01 would have had that
2016-2020 history available, not a blank slate. So each strategy is run ONCE across the full
2016-present history and the resulting trades/equity curve are split by date after the fact.
"""
import pandas as pd

from metrics import calculate_metrics
from backtest_engine import run_rsi, run_macd, buy_and_hold

IN_SAMPLE_START = pd.Timestamp('2016-01-01')
IN_SAMPLE_END = pd.Timestamp('2020-12-31')
OUT_OF_SAMPLE_START = pd.Timestamp('2021-01-01')


def split_trades_by_period(trades: list, start: pd.Timestamp, end: pd.Timestamp = None) -> list:
    """Trades whose ENTRY falls within [start, end] (end=None means open-ended)."""
    return [t for t in trades if pd.Timestamp(t.entry_date) >= start
            and (end is None or pd.Timestamp(t.entry_date) <= end)]


def split_equity_by_period(equity_curve: list, start: pd.Timestamp, end: pd.Timestamp = None) -> list:
    """Equity curve points within [start, end] (end=None means open-ended)."""
    return [(d, v) for d, v in equity_curve if pd.Timestamp(d) >= start
            and (end is None or pd.Timestamp(d) <= end)]


def in_sample_out_of_sample_table(ticker: str, initial_capital: float = 10000) -> pd.DataFrame:
    """Buy & Hold's single trade only "enters" once, on day one of the full history, so it would
    never appear in the out-of-sample trades slice at all if scored by trades alone."""
    _, rsi_portfolio = run_rsi(ticker, initial_capital)
    df, macd_portfolio = run_macd(ticker, initial_capital)
    if df is None:
        return pd.DataFrame()
    bh_trade, bh_equity_curve = buy_and_hold(df['Close'], initial_capital)

    strategies = {
        "RSI": (rsi_portfolio.get_trades() if rsi_portfolio else [], rsi_portfolio.equity_curve if rsi_portfolio else []),
        "MACD": (macd_portfolio.get_trades() if macd_portfolio else [], macd_portfolio.equity_curve if macd_portfolio else []),
        "Buy & Hold": ([bh_trade], bh_equity_curve),
    }

    periods = [
        ("in-sample (2016-2020)", IN_SAMPLE_START, IN_SAMPLE_END),
        ("out-of-sample (2021-present)", OUT_OF_SAMPLE_START, None),
    ]

    rows = []
    for name, (trades, equity_curve) in strategies.items():
        for period_label, start, end in periods:
            period_trades = split_trades_by_period(trades, start, end)
            period_equity = split_equity_by_period(equity_curve, start, end)
            m = calculate_metrics(period_trades, initial_capital=initial_capital,
                                   equity_curve=period_equity if len(period_equity) >= 2 else None)
            rows.append({
                'Strategy / Sample': f"{name} - {period_label}",
                'CAGR (%)': m.get('CAGR (%)', float('nan')),
                'Sharpe Ratio': m.get('Sharpe Ratio', float('nan')),
                'Max Drawdown (%)': m.get('Max Drawdown (%)', float('nan')),
                'Win Rate (%)': m.get('Win Rate (%)', float('nan')),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from backtest_engine import get_sp500_tickers, prompt_for_ticker, prompt_for_scope

    scope = prompt_for_scope()
    if scope == "index":
        ticker = "SPY"
        label = "S&P 500 Index (SPY)"
    else:
        ticker = prompt_for_ticker(get_sp500_tickers())
        label = ticker

    table = in_sample_out_of_sample_table(ticker)

    print(f"\nTable 8 - In-sample vs. out-of-sample performance - {label}\n")
    display = table.copy()
    for col in display.columns[1:]:
        display[col] = display[col].apply(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    print(display.to_string(index=False))

    print("\nObservation: [ state how much each strategy's Sharpe/CAGR degrades from in-sample "
          "to out-of-sample, and whether the degradation is larger for RSI/MACD than for Buy & "
          "Hold ]")
