import pandas as pd

from metrics import calculate_metrics


def classify_regimes(price_series: pd.Series, return_window: int = 126, vol_window: int = 21,
                      bull_threshold: float = 0.10, bear_threshold: float = -0.10) -> pd.Series:
    trailing_return = price_series.pct_change(return_window)
    daily_returns = price_series.pct_change()
    rolling_vol = daily_returns.rolling(vol_window).std() * (252 ** 0.5)

    trend = pd.Series(index=price_series.index, dtype=object)
    trend[trailing_return > bull_threshold] = 'Bull'
    trend[trailing_return < bear_threshold] = 'Bear'
    trend[(trailing_return >= bear_threshold) & (trailing_return <= bull_threshold)] = 'Sideways'

    vol_median = rolling_vol.median()
    vol_level = pd.Series(index=price_series.index, dtype=object)
    vol_level[rolling_vol > vol_median] = 'High Vol'
    vol_level[rolling_vol <= vol_median] = 'Low Vol'

    regime = trend + " / " + vol_level
    regime[trailing_return.isna() | rolling_vol.isna()] = None
    return regime


def summarize_regime_segments(regime_series: pd.Series, min_days: int = 0) -> list:
    """Collapses day-by-day regime labels into contiguous (start, end, label, days) segments -
    purely to check the classifier against the market. """
    segments = []
    current_label = None
    seg_start = None
    prev_date = None

    for date, label in regime_series.items():
        if label != current_label:
            if current_label is not None:
                segments.append((seg_start, prev_date, current_label))
            current_label = label
            seg_start = date
        prev_date = date
    if current_label is not None:
        segments.append((seg_start, prev_date, current_label))

    all_segments = [(s, e, l, (e - s).days + 1) for s, e, l in segments if l is not None]
    return [seg for seg in all_segments if seg[3] >= min_days]


def split_trades_by_regime(trades: list, regime_series: pd.Series) -> dict:
    """Buckets trades by the regime in effect on each trade's ENTRY date. Good for trade-level
    stats (win rate, profit factor, avg win/loss by regime).
    Trades entered during the unclassified warm-up window are dropped from the breakdown -
    they're still counted in the strategy's overall metrics elsewhere, just not regime-attributed.
    """
    regime_by_date = regime_series.dropna().to_dict()
    buckets = {}
    for trade in trades:
        label = regime_by_date.get(trade.entry_date)
        if label is None:
            continue
        buckets.setdefault(label, []).append(trade)
    return buckets


def split_equity_by_regime(equity_curve: list, regime_series: pd.Series) -> dict:
    regime_by_date = regime_series.dropna().to_dict()
    buckets = {}
    for i in range(1, len(equity_curve)):
        date, value = equity_curve[i]
        _, prev_value = equity_curve[i - 1]
        label = regime_by_date.get(date)
        if label is None or prev_value == 0:
            continue
        buckets.setdefault(label, []).append((value / prev_value) - 1)
    return buckets


def _compound(daily_returns: list) -> float:
    total = 1.0
    for r in daily_returns:
        total *= (1 + r)
    return (total - 1) * 100


def build_regime_performance_matrix(regime_series: pd.Series, strategy_equity_curves: dict,
                                     initial_capital: float = 10000) -> pd.DataFrame:
    """strategy_equity_curves: Every strategy is attributed to regimes via its day-by-day equity 
    curve (not trade entry dates).
    """
    all_regimes = sorted(regime_series.dropna().unique())
    strategy_names = list(strategy_equity_curves.keys())

    rows = []
    for regime in all_regimes:
        row = {'Regime': regime}
        for name, curve in strategy_equity_curves.items():
            daily_returns = split_equity_by_regime(curve, regime_series).get(regime, [])
            row[name] = f"{_compound(daily_returns):.1f}% ({len(daily_returns)}d)" if daily_returns else "-"
        rows.append(row)
    return pd.DataFrame(rows)[['Regime'] + strategy_names]


def build_regime_trade_stats(regime_series: pd.Series, strategy_trades: dict,
                              initial_capital: float = 10000) -> pd.DataFrame:
    """strategy_trades: {strategy_name: list[Trade]}. Trade-level view (win rate, trade count)
    by regime, for strategies that actually take discrete trades - not meaningful for Buy & Hold
    (see split_trades_by_regime's docstring), so leave that one out when calling this."""
    all_regimes = sorted(regime_series.dropna().unique())
    strategy_names = list(strategy_trades.keys())

    rows = []
    for regime in all_regimes:
        row = {'Regime': regime}
        for name, trades in strategy_trades.items():
            bucket = split_trades_by_regime(trades, regime_series).get(regime, [])
            if not bucket:
                row[f"{name} Trades"] = 0
                row[f"{name} Win Rate"] = "-"
            else:
                m = calculate_metrics(bucket, initial_capital=initial_capital)
                row[f"{name} Trades"] = len(bucket)
                row[f"{name} Win Rate"] = f"{m['Win Rate (%)']:.0f}%"
        rows.append(row)
    cols = ['Regime'] + [c for name in strategy_names for c in (f"{name} Trades", f"{name} Win Rate")]
    return pd.DataFrame(rows)[cols]


if __name__ == "__main__":
    from backtest_engine import get_sp500_tickers, prompt_for_ticker, prompt_for_scope, run_rsi, run_macd, buy_and_hold

    INITIAL_CAPITAL = 10000

    scope = prompt_for_scope()
    if scope == "index":
        ticker = "SPY"
        label = "S&P 500 Index (SPY)"
    else:
        ticker = prompt_for_ticker(get_sp500_tickers())
        label = ticker

    _, rsi_portfolio = run_rsi(ticker, INITIAL_CAPITAL)
    df, macd_portfolio = run_macd(ticker, INITIAL_CAPITAL)
    if df is None:
        raise SystemExit(f"No price data returned for {label}.")
    bh_trade, bh_equity_curve = buy_and_hold(df['Close'], INITIAL_CAPITAL)

    regime_series = classify_regimes(df['Close'])

    all_segments = summarize_regime_segments(regime_series)
    shown_segments = summarize_regime_segments(regime_series, min_days=10)
    hidden = len(all_segments) - len(shown_segments)
    print(f"\nRegime segments detected for {label} (segments under 10 days hidden, {hidden} of {len(all_segments)} total):\n")
    print(pd.DataFrame(shown_segments, columns=['Start', 'End', 'Regime', 'Days']).to_string(index=False))

    rsi_name = "RSI"
    equity_curves = {
        rsi_name: rsi_portfolio.equity_curve if rsi_portfolio else [],
        "MACD": macd_portfolio.equity_curve if macd_portfolio else [],
        "Buy & Hold": bh_equity_curve,
    }

    print(f"\nStrategy x Regime performance matrix - {label}\n")
    matrix = build_regime_performance_matrix(regime_series, equity_curves, INITIAL_CAPITAL)
    print(matrix.to_string(index=False))

    trade_strategies = {
        rsi_name: rsi_portfolio.get_trades() if rsi_portfolio else [],
        "MACD": macd_portfolio.get_trades() if macd_portfolio else [],
    }
    print(f"\nTrade count and win rate by regime - {label}\n")
    trade_stats = build_regime_trade_stats(regime_series, trade_strategies, INITIAL_CAPITAL)
    print(trade_stats.to_string(index=False))
