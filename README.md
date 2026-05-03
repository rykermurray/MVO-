# Portfolio Optimization

A clean Python project for mean-variance portfolio optimization. It downloads historical prices, estimates CAPM expected returns, builds a Ledoit-Wolf shrinkage covariance matrix, solves a long-only max-Sharpe portfolio, and saves the resulting weights and performance metrics to JSON.

The project also includes a monthly rebalance backtester that tracks realized performance over time.

## Features

- Historical adjusted price downloads with `yfinance`
- CAPM-based expected annual returns
- Ledoit-Wolf shrinkage covariance estimation
- Max-Sharpe mean-variance optimization with `PyPortfolioOpt`
- Monthly rebalancing backtest with no look-ahead bias
- Equity curve, daily returns, rebalance weights, turnover, and summary metrics
- Long-only constraints:
  - minimum asset weight: 2%
  - maximum asset weight: 15%
  - no short selling
- JSON output for current weights and backtest results

## Project Structure

```text
portfolio-optimization/
├── data/
├── src/
│   ├── __init__.py
│   ├── backtester.py
│   ├── data_loader.py
│   ├── optimizer.py
│   └── utils.py
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── backtest_results.json
│   └── weights.json
├── requirements.txt
└── README.md
```

## Setup

```bash
cd portfolio-optimization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optimize Current Portfolio

Run this from the project root:

```python
from src.data_loader import download_price_data
from src.optimizer import optimize_portfolio
from src.utils import pretty_print_results, save_optimization_result

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "V",
    "UNH",
    "XOM",
]

start_date = "2021-01-01"
end_date = "2026-01-01"

prices = download_price_data(
    tickers,
    start_date,
    end_date,
    save_path="data/prices.csv",
)
market_prices = download_price_data(["SPY"], start_date, end_date)

result = optimize_portfolio(
    prices,
    market_prices=market_prices,
    risk_free_rate=0.02,
    min_weight=0.02,
    max_weight=0.15,
)

pretty_print_results(result)
save_optimization_result(
    result,
    "results/weights.json",
    metadata={"benchmark": "SPY", "start_date": start_date, "end_date": end_date},
)
```

## Monthly Rebalance Backtest

```python
from src.backtester import backtest_monthly_rebalance
from src.utils import pretty_print_backtest, save_backtest_result

backtest = backtest_monthly_rebalance(
    prices,
    market_prices=market_prices,
    lookback_days=252,
    initial_capital=1.0,
    risk_free_rate=0.02,
    min_weight=0.02,
    max_weight=0.15,
    transaction_cost_bps=5.0,
    optimization_failure_policy="carry_forward",
)

pretty_print_backtest(backtest)
save_backtest_result(
    backtest,
    "results/backtest_results.json",
    metadata={"benchmark": "SPY", "rebalance": "monthly"},
)
```

The backtester re-optimizes on the last available trading day of each month using only historical data up to that date, then applies those weights to the next holding period. The output includes:

- `equity_curve`: portfolio value through time
- `daily_returns`: realized daily strategy returns
- `weights_history`: optimized weights at each rebalance
- `rebalance_metrics`: optimizer metrics and turnover at each rebalance
- `summary`: total return, annualized return, volatility, Sharpe ratio, max drawdown, and final value

If a rebalance window cannot produce a max-Sharpe portfolio, usually because CAPM expected returns are below the risk-free rate, `optimization_failure_policy` controls the behavior:

- `"raise"`: stop immediately with the optimization error
- `"carry_forward"`: reuse the prior weights, or equal weights if no prior weights exist
- `"equal_weight"`: use equal weights for that rebalance

## Notebook

Open the included notebook for a full workflow:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook downloads prices, runs the optimizer, displays weights and performance metrics, runs a monthly rebalance backtest, plots the equity curve, and writes JSON results under `results/`.

## Constraint Notes

The default 2% minimum and 15% maximum weight constraints must be feasible. With these bounds, the portfolio needs at least 7 assets because 6 assets capped at 15% can only allocate 90%. It also supports at most 50 assets because 51 assets at a 2% minimum would require 102% allocation.

This project raises clear errors when the requested tickers, date range, or constraints cannot produce a valid portfolio.
