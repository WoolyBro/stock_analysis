from dataclasses import dataclass
from datetime import datetime
from statistics import stdev


@dataclass
class Trade:
    """One completed buy-then-sell cycle. The common format every strategy (RSI, MACD, ...)
    should output, so a single calculate_metrics() can score all of them the same way."""
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: float

    @property
    def pnl(self):
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def return_pct(self):
        return (self.exit_price - self.entry_price) / self.entry_price


def _max_drawdown_pct(values: list[float]) -> float:
    running_max = values[0]
    max_drawdown = 0.0
    for value in values:
        running_max = max(running_max, value)
        drawdown = (value - running_max) / running_max
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown * 100


def calculate_metrics(trades: list[Trade], initial_capital: float = 10000, risk_free_rate: float = 0.0,
                       equity_curve: list[tuple] = None) -> dict:
    """equity_curve: optional list of (date, value) covering every day in the backtest, not just
    trade entry/exit points. This is needed for strategies like Buy & Hold that hold a position
    continuously — without it, drawdown or volatility can only be approximated by compounding
    the discrete trade returns which does not take into account anything that happens between trades (e.g. a
    single buy-and-hold "trade" would otherwise show 0% drawdown, since it only has a start and
    end point and no visibility into the price path in between)."""
    if not trades and not equity_curve:
        return {}

    trades = sorted(trades, key=lambda t: t.entry_date)
    returns = [t.return_pct for t in trades]
    n = len(trades)

    if equity_curve:
        dates = [d for d, _ in equity_curve]
        values = [v for _, v in equity_curve]
        final_value = values[-1]
        years = (dates[-1] - dates[0]).days / 365.25

        max_drawdown = _max_drawdown_pct(values)

        period_returns = [(values[i] / values[i - 1]) - 1 for i in range(1, len(values))]
        periods_per_year = len(period_returns) / years if years > 0 else float('nan')
        std_period_return = stdev(period_returns) if len(period_returns) > 1 else 0.0
        mean_period_return = sum(period_returns) / len(period_returns) if period_returns else 0.0
    else:
        # simulate an equity curve by compounding each trade's return on the full account,
        # since we only have discrete round-trip trades and not a continuous daily equity series
        equity = [initial_capital]
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        final_value = equity[-1]
        years = (trades[-1].exit_date - trades[0].entry_date).days / 365.25

        max_drawdown = _max_drawdown_pct(equity)

        # trade-level returns annualized by trade frequency, since we have no daily equity curve
        period_returns = returns
        periods_per_year = n / years if years > 0 else float('nan')
        std_period_return = stdev(period_returns) if len(period_returns) > 1 else 0.0
        mean_period_return = sum(period_returns) / len(period_returns)

    # --- Return metrics ---
    total_return = (final_value - initial_capital) / initial_capital
    cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else float('nan')
    avg_return_per_trade = (final_value - initial_capital) / n if n > 0 else float('nan')

    # --- Trading performance metrics --- (all NaN when n == 0: no trades in this slice to average)
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    win_rate = (len(wins) / n * 100) if n > 0 else float('nan')
    avg_win = (sum(wins) / len(wins)) * 100 if wins else (float('nan') if n == 0 else 0.0)
    avg_loss = (sum(losses) / len(losses)) * 100 if losses else (float('nan') if n == 0 else 0.0)

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float('inf')
    else:
        profit_factor = float('nan')

    # --- Risk metrics ---
    annual_volatility = std_period_return * (periods_per_year ** 0.5) * 100 if periods_per_year == periods_per_year else float('nan')
    sharpe_ratio = ((mean_period_return - risk_free_rate) / std_period_return) * (periods_per_year ** 0.5) if std_period_return > 0 else float('nan')

    return {
        'Number of Trades': n,
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr * 100,
        'Average Return per Trade ($)': avg_return_per_trade,
        'Win Rate (%)': win_rate,
        'Average Winning Trade (%)': avg_win,
        'Average Losing Trade (%)': avg_loss,
        'Profit Factor': profit_factor,
        'Max Drawdown (%)': max_drawdown,
        'Annualized Volatility (%)': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
    }


def print_metrics(metrics: dict, label: str = ""):
    if label:
        print(f"--- {label} ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    # sanity check using the sample trade from the notes, plus a couple more so the
    # win/loss and Sharpe/volatility math has more than one data point to work with
    sample_trades = [
        Trade(datetime(2019, 3, 4), 2743, datetime(2019, 4, 18), 2905, 10),
        Trade(datetime(2019, 5, 1), 2900, datetime(2019, 5, 20), 2750, 10),
        Trade(datetime(2019, 6, 10), 2800, datetime(2019, 7, 2), 3050, 10),
    ]
    print_metrics(calculate_metrics(sample_trades), label="Sample Trades")
