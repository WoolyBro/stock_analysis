from datetime import datetime, timedelta

from metrics import Trade, calculate_metrics, print_metrics

# Per-round-trip transaction cost (brokerage + slippage combined, e.g. 0.001 = 0.1%),
# charged once per completed trade by shaving it off the exit price in Portfolio.sell()
# below. This is what "profits after brokerage" means in this project - every result
# By default, charges are set to 0.0


TRANSACTION_COST = 0.0       # baseline - no costs (frictionless, matches every result so far)
# TRANSACTION_COST = 0.001   # 0.1% per round trip

class Portfolio:
    """Tracks cash, holdings, and a full equity curve over time. Every position is
    all-in / all-out: buy() spends 100% of cash, sell() liquidates 100% of holdings,
    and only one position can be open at a time."""

    def __init__(self, initial_capital: float, transaction_cost: float = TRANSACTION_COST):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # per-round-trip cost used by sell() below
        self.cash = initial_capital
        self.holdings = 0
        self.equity_curve = []  # list of (date, total_value)
        self.trades = []        # list of completed Trade objects
        self.open_trade = None  # entry_date/entry_price/quantity until sell() fills it

    def buy(self, date, price):
        if self.open_trade is not None or self.cash <= 0:
            return
        self.holdings = self.cash / price
        self.open_trade = {'entry_date': date, 'entry_price': price, 'quantity': self.holdings}
        self.cash = 0

    def sell(self, date, price):
        if self.open_trade is None:
            return
        # the whole round-trip's cost is taken off the exit proceeds here, in one place, so
        # it flows through correctly into: the reported Trade's exit_price/return, the cash
        # available to compound into the NEXT trade, and every day's equity curve from here on
        exit_price = price * (1 - self.transaction_cost)
        self.trades.append(Trade(
            entry_date=self.open_trade['entry_date'],
            entry_price=self.open_trade['entry_price'],
            exit_date=date,
            exit_price=exit_price,
            quantity=self.open_trade['quantity'],
        ))
        self.cash = self.holdings * exit_price
        self.holdings = 0
        self.open_trade = None

    def update_equity(self, date, current_price):
        total_value = self.cash + self.holdings * current_price
        self.equity_curve.append((date, total_value))

    def get_total_return(self):
        if not self.equity_curve:
            return 0.0
        final_value = self.equity_curve[-1][1]
        return (final_value - self.initial_capital) / self.initial_capital

    def get_trades(self):
        return self.trades


if __name__ == "__main__":
    # sanity check: walk a tiny synthetic price series, buy partway through, sell later,
    # and confirm cash/holdings/equity_curve/trades all update as expected
    prices = [100, 102, 105, 103, 108, 112, 110, 115, 120, 118]
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(len(prices))]

    portfolio = Portfolio(initial_capital=10000)

    for i, (date, price) in enumerate(zip(dates, prices)):
        if i == 2:
            portfolio.buy(date, price)
        if i == 7:
            portfolio.sell(date, price)
        portfolio.update_equity(date, price)

    print("Cash:", portfolio.cash)
    print("Holdings:", portfolio.holdings)
    print("Equity curve (last 3):", portfolio.equity_curve[-3:])
    print("Total return:", portfolio.get_total_return())
    print("Trades:", portfolio.get_trades())

    print_metrics(calculate_metrics(portfolio.get_trades(), initial_capital=10000), label="Portfolio Trades")
