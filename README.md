# RSI vs. MACD vs. Buy & Hold — Backtesting Project

This folder holds the code behind *"Do Simple Technical Indicators Survive Scrutiny? A
Custom-Built Backtesting Study of RSI and MACD versus Buy-and-Hold"*. The
question the paper asks is simple: do RSI and MACD actually beat just buying and holding, or
does their apparent edge fall apart once you test it properly? Everything here — the signal
logic, the portfolio simulation, the execution rules, the metrics — is written from scratch in
Python, on purpose, so every assumption in the paper is something you can point to in code
rather than something borrowed from a library you have to take on faith.

## How to run it

Every script asks the same first question — analyze the S&P 500 index (SPY) as a whole, or one
specific stock — then downloads live data through `yfinance` and prints its results straight to
the console. Nothing needs arguments; just run the file for whatever question you're answering.

```
python backtest_engine.py              # baseline: RSI vs MACD vs Buy & Hold on one instrument
python regime_analysis.py              # performance broken down by market regime
python rolling_window_stability.py     # rolling 2-year Sharpe, is the edge stable over time?
python in_sample_out_of_sample.py      # 2016-2020 vs 2021-present, is it curve-fit to one period?
python combined_strategies.py          # what if RSI and MACD are combined instead of run alone?
```

`RSI.py` and `MAC.py` also run standalone if you want to see one strategy on its own, plotted.

## Project structure

**Core engine** — the part everything else is built on:

| File | Job |
|---|---|
| `metrics.py` | Defines `Trade` and one `calculate_metrics()` function — CAGR, Sharpe, max drawdown, win rate, profit factor, etc. Every strategy in this project is scored through this one function, on purpose, so the numbers are never computed two different ways. |
| `portfolio.py` | Tracks cash, holdings, and a day-by-day equity curve. All-in / all-out only — no partial positions, no shorting. Also where the transaction-cost toggle lives (see below). |
| `backtest_engine.py` | Walks price data day by day and turns a `BUY`/`SELL`/`HOLD` signal column into real trades through a `Portfolio`. Same engine runs RSI, MACD, and Buy & Hold — no strategy-specific code inside it. |
| `RSI.py` / `MAC.py` | The two strategies being tested. Each just produces buy/sell signals; neither knows or cares how a trade gets executed — that's the engine's job. |

**Robustness checks** — each one re-uses the engine above instead of re-implementing anything,
and maps to a specific section of the paper:

| File | Paper section | Question it answers |
|---|---|---|
| `regime_analysis.py` |  Does the edge only exist in certain market conditions (bull/bear, high/low vol)? |
| `rolling_window_stability.py` | Is performance stable over time, or does it swing between clearly good and clearly bad? |
| `in_sample_out_of_sample.py` | Does it hold up on data from *after* the strategy was fixed, or was it just lucky on 2016-2020? |
| `combined_strategies.py` | Does requiring RSI and MACD to agree (or blending them) do any better than either alone? |

Sections not built yet: 3.2.3 (max holding period), 3.2.4 (MACD span sweep), 3.4/3.5
(cross-sectional and geographic), 3.7 (transaction costs, placebo test), 3.8 (statistical
significance). The span and entry/exit sweeps from 3.2.1/3.2.2 are handled differently — as
toggle constants at the top of `RSI.py` (see below), not separate files.

## Design decisions worth knowing about

These aren't bugs — they're deliberate choices, and they matter if a number looks surprising:

- **Execution is same-day close, not next-day open.** Stated explicitly in
  `BacktestEngine`'s docstring. In real life you can't react instantly to a signal you only see
  at the close, so this is a simplification, not an oversight.
- **RSI's and MACD's Max Drawdown/Sharpe are trade-chained approximations**, not a real daily
  equity curve, *unless* an `equity_curve` is passed into `calculate_metrics()` (Buy & Hold
  always gets one; RSI/MACD do too where the analysis needs it, like the regime and
  rolling-window files). Where it isn't passed, those numbers are blind to anything that happens
  *between* trades — worth remembering before comparing them directly to Buy & Hold's.
- **Parameters are fixed conventions, not fitted.** RSI's smoothing span (default 19) and
  entry/exit thresholds (default 30/40) are toggled by commenting/uncommenting constants at the
  top of `RSI.py` — they were never optimized against this data, which is *why* the
  in-sample/out-of-sample check in Section 3.3.3 is a meaningful test rather than a formality.
- **Transaction costs default to zero.** `TRANSACTION_COST` in `portfolio.py` is 0.0 —
  every result in this project so far is frictionless unless that's changed by hand.
- **No shorting, one position at a time.** Every strategy is only ever long or in cash.

## Output

Every script prints full trade logs and results tables. The point of this project is that every 
claim should be traceable back to the actual trades that produced it, not just a headline statistic.
