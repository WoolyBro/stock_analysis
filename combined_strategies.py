"""Combined-strategy experiments: tests whether requiring RSI
and MACD to agree or blending them, improves on either alone - and whether the two are
independent enough for combination to plausibly help in the first place.
"""
import pandas as pd

from metrics import calculate_metrics
from backtest_engine import BacktestEngine, run_rsi, run_macd


def _in_position_state(dates_index: pd.DatetimeIndex, trades: list) -> pd.Series:
    """Boolean series aligned to dates_index, True on every day between a trade's entry and
    exit (inclusive)."""
    state = pd.Series(False, index=dates_index)
    for t in trades:
        state.loc[(state.index >= pd.Timestamp(t.entry_date)) & (state.index <= pd.Timestamp(t.exit_date))] = True
    return state


def combined_signal_portfolio(df: pd.DataFrame, rsi_portfolio, macd_portfolio, rule: str, initial_capital: float = 10000):
    price_data = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'close'})
    dates_index = pd.DatetimeIndex(price_data['date'])

    rsi_state = _in_position_state(dates_index, rsi_portfolio.get_trades() if rsi_portfolio else [])
    macd_state = _in_position_state(dates_index, macd_portfolio.get_trades() if macd_portfolio else [])

    if rule == 'AND':
        combined = rsi_state.values & macd_state.values
    elif rule == 'OR':
        combined = rsi_state.values | macd_state.values
    else:
        raise ValueError(f"Unknown rule: {rule!r}, expected 'AND' or 'OR'")

    prev = pd.Series(combined).shift(1, fill_value=False).values
    price_data['signal'] = 'HOLD'
    price_data.loc[combined & ~prev, 'signal'] = 'BUY'
    price_data.loc[~combined & prev, 'signal'] = 'SELL'

    return BacktestEngine(price_data, initial_capital=initial_capital).run()


def blend_50_50(ticker: str, initial_capital: float = 10000):
    """Runs RSI and MACD each with half the capital, independently, then sums their equity
    curves day by day and pools their trades. Days before a strategy's own equity curve has
    started (RSI needs 200 days of history before its first reading) are treated as that half
    of capital sitting untouched in cash - a realistic assumption, since the strategy genuinely
    hasn't started operating yet. Returns (trades, equity_curve)."""
    half = initial_capital / 2
    _, rsi_portfolio = run_rsi(ticker, half)
    df, macd_portfolio = run_macd(ticker, half)
    if df is None:
        return [], []

    rsi_curve = dict(rsi_portfolio.equity_curve) if rsi_portfolio else {}
    macd_curve = dict(macd_portfolio.equity_curve) if macd_portfolio else {}

    all_dates = sorted(set(rsi_curve) | set(macd_curve))
    blended_curve = [(d, rsi_curve.get(d, half) + macd_curve.get(d, half)) for d in all_dates]

    blended_trades = (rsi_portfolio.get_trades() if rsi_portfolio else []) + \
                      (macd_portfolio.get_trades() if macd_portfolio else [])
    return blended_trades, blended_curve


def daily_returns_correlation(rsi_portfolio, macd_portfolio) -> float:
    rsi_curve = dict(rsi_portfolio.equity_curve) if rsi_portfolio else {}
    macd_curve = dict(macd_portfolio.equity_curve) if macd_portfolio else {}
    common_dates = sorted(set(rsi_curve) & set(macd_curve))
    if len(common_dates) < 3:
        return float('nan')

    rsi_vals = [rsi_curve[d] for d in common_dates]
    macd_vals = [macd_curve[d] for d in common_dates]
    rsi_returns = [(rsi_vals[i] / rsi_vals[i - 1]) - 1 for i in range(1, len(rsi_vals))]
    macd_returns = [(macd_vals[i] / macd_vals[i - 1]) - 1 for i in range(1, len(macd_vals))]

    n = len(rsi_returns)
    mean_r = sum(rsi_returns) / n
    mean_m = sum(macd_returns) / n
    cov = sum((r - mean_r) * (m - mean_m) for r, m in zip(rsi_returns, macd_returns)) / n
    std_r = (sum((r - mean_r) ** 2 for r in rsi_returns) / n) ** 0.5
    std_m = (sum((m - mean_m) ** 2 for m in macd_returns) / n) ** 0.5
    if std_r == 0 or std_m == 0:
        return float('nan')
    return cov / (std_r * std_m)


if __name__ == "__main__":
    from backtest_engine import get_sp500_tickers, prompt_for_ticker, prompt_for_scope

    INITIAL_CAPITAL = 10000

    scope = prompt_for_scope()
    if scope == "index":
        ticker = "SPY"
        label = "S&P 500 Index (SPY)"
    else:
        ticker = prompt_for_ticker(get_sp500_tickers())
        label = ticker

    # run each strategy at full capital ONCE, reused for the baseline rows, AND, OR, and the
    # correlation figure - only the 50/50 blend needs its own separate half-capital runs,
    # since position sizing depends on how much capital each strategy actually gets
    _, rsi_portfolio = run_rsi(ticker, INITIAL_CAPITAL)
    df, macd_portfolio = run_macd(ticker, INITIAL_CAPITAL)
    if df is None:
        raise SystemExit(f"No price data returned for {label}.")

    and_portfolio = combined_signal_portfolio(df, rsi_portfolio, macd_portfolio, 'AND', INITIAL_CAPITAL)
    or_portfolio = combined_signal_portfolio(df, rsi_portfolio, macd_portfolio, 'OR', INITIAL_CAPITAL)
    blend_trades, blend_curve = blend_50_50(ticker, INITIAL_CAPITAL)

    rows_data = {
        "RSI alone (baseline)": (rsi_portfolio.get_trades() if rsi_portfolio else [], None),
        "MACD alone (baseline)": (macd_portfolio.get_trades() if macd_portfolio else [], None),
        "AND - both must agree": (and_portfolio.get_trades(), None),
        "OR - either signal": (or_portfolio.get_trades(), None),
        "50/50 capital-split blend": (blend_trades, blend_curve),
    }

    print(f"\nTable 11 - Combined-signal performance vs. each strategy alone - {label}\n")
    rows = []
    for name, (trades, equity_curve) in rows_data.items():
        m = calculate_metrics(trades, initial_capital=INITIAL_CAPITAL,
                               equity_curve=equity_curve if equity_curve else None)
        rows.append({
            'Rule': name,
            'CAGR (%)': m.get('CAGR (%)', float('nan')),
            'Sharpe Ratio': m.get('Sharpe Ratio', float('nan')),
            'Max Drawdown (%)': m.get('Max Drawdown (%)', float('nan')),
            'Win Rate (%)': m.get('Win Rate (%)', float('nan')),
            '# Trades': len(trades),
        })
    table_df = pd.DataFrame(rows)
    display = table_df.copy()
    for col in ['CAGR (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']:
        display[col] = display[col].apply(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    print(display.to_string(index=False))

    corr = daily_returns_correlation(rsi_portfolio, macd_portfolio)
    corr_text = f"{corr:.3f}" if corr == corr else "nan (not enough overlapping history)"
    print(f"\nCorrelation between RSI's and MACD's period returns: {corr_text}")
