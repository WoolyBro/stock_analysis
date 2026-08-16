"""Rolling-window stability: does RSI/MACD's Sharpe
ratio hold up consistently across rolling 2-year windows, stepped every 6 months, across the
whole backtest - or does it swing between clearly positive and clearly negative, meaning the
edge is regime-dependent rather than a stable property of the strategy?"""
import pandas as pd
import matplotlib.pyplot as plt

from metrics import calculate_metrics


def rolling_window_stability(equity_curve: list, window_years: int = 2, step_months: int = 6) -> pd.DataFrame:
    """Slides a window_years-long window across the equity curve, stepping step_months at a
    time, and computes CAGR/Sharpe from each window's slice via metrics.calculate_metrics()
    (trades=[], equity_curve=slice - a 2-year window can easily contain zero discrete RSI/MACD
    trades since they trade rarely, but calculate_metrics() only needs trades to be non-empty
    when there's no equity_curve; with one, it computes CAGR/Sharpe from that alone)."""
    if not equity_curve:
        return pd.DataFrame()

    dates = [pd.Timestamp(d) for d, _ in equity_curve]
    start, end = dates[0], dates[-1]
    window = pd.DateOffset(years=window_years)
    window_starts = pd.date_range(start=start, end=end - window, freq=pd.DateOffset(months=step_months))

    rows = []
    for window_start in window_starts:
        window_end = window_start + window
        slice_curve = [(d, v) for d, v in equity_curve if window_start <= pd.Timestamp(d) <= window_end]
        m = calculate_metrics([], equity_curve=slice_curve) if len(slice_curve) >= 2 else {}
        rows.append({
            'Window Start': window_start.date(),
            'Window End': window_end.date(),
            'CAGR (%)': m.get('CAGR (%)', float('nan')),
            'Sharpe Ratio': m.get('Sharpe Ratio', float('nan')),
        })
    return pd.DataFrame(rows)


def plot_rolling_sharpe(strategy_windows: dict, label: str, save_path: str = None):
    plt.figure(figsize=(12, 5))
    for name, df in strategy_windows.items():
        if df.empty:
            continue
        # Buy & Hold is extra context beyond the paper's Section 3.3.2 spec (which only asks
        # for RSI and MACD), so it's drawn as a light dashed reference line, not a main series
        style = '--' if name == 'Buy & Hold' else '-'
        plt.plot(df['Window Start'], df['Sharpe Ratio'], marker='o', linestyle=style, label=name)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.title(f"Figure 2 - Rolling 2-Year Sharpe Ratio (stepped every 6 months) - {label}")
    plt.xlabel("Window Start")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved chart to {save_path}")
    plt.show()


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

    strategies = {
        "RSI": rsi_portfolio.equity_curve if rsi_portfolio else [],
        "MACD": macd_portfolio.equity_curve if macd_portfolio else [],
        "Buy & Hold": bh_equity_curve,
    }

    windows = {name: rolling_window_stability(curve) for name, curve in strategies.items()}

    for name, wdf in windows.items():
        print(f"\n--- Rolling 2-year Sharpe, stepped 6 months - {name} on {label} ---\n")
        if wdf.empty:
            print("Not enough history for even one full 2-year window.")
            continue
        print(wdf.to_string(index=False))
        valid = wdf['Sharpe Ratio'].dropna()
        if len(valid) > 1:
            print(f"\nSharpe across windows: min={valid.min():.2f}, max={valid.max():.2f}, "
                  f"mean={valid.mean():.2f}, std={valid.std():.2f}")

    print("\nObservation: [ describe whether the rolling Sharpe line is roughly flat (stable "
          "edge) or swings between clearly positive and clearly negative (regime-dependent "
          "edge) ]")

    plot_rolling_sharpe(windows, label, save_path=f"figure2_rolling_sharpe_{ticker}.png")
