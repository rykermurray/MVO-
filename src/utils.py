"""Utility helpers for saving and displaying optimization results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from src.backtester import BacktestResult
from src.optimizer import PortfolioResult


def save_optimization_result(
    result: PortfolioResult,
    output_path: Union[str, Path] = "results/weights.json",
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save optimized weights and performance metrics to a JSON file."""

    payload = result.to_dict()
    if metadata:
        payload["metadata"] = dict(metadata)

    return save_json(payload, output_path)


def save_backtest_result(
    result: BacktestResult,
    output_path: Union[str, Path] = "results/backtest_results.json",
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save monthly rebalance backtest results to a JSON file."""

    payload = result.to_dict()
    if metadata:
        payload["metadata"] = dict(metadata)

    return save_json(payload, output_path)


def save_weights_to_json(
    weights: Mapping[str, float],
    output_path: Union[str, Path] = "results/weights.json",
    *,
    performance: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save a weight dictionary, with optional metrics, to JSON."""

    payload: dict[str, Any] = {
        "weights": {ticker: float(weight) for ticker, weight in weights.items()}
    }
    if performance:
        payload["performance"] = {
            metric: float(value) for metric, value in performance.items()
        }
    if metadata:
        payload["metadata"] = dict(metadata)

    return save_json(payload, output_path)


def save_json(payload: Mapping[str, Any], output_path: Union[str, Path]) -> Path:
    """Write JSON to disk, creating parent directories when needed."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")

    return path


def pretty_print_results(result: PortfolioResult) -> None:
    """Print weights and metrics in a readable terminal-friendly format."""

    print("Optimal Portfolio Weights")
    print("-------------------------")
    for ticker, weight in sorted(result.weights.items(), key=lambda item: item[1], reverse=True):
        print(f"{ticker:>8}: {weight:>7.2%}")

    print("\nPortfolio Performance")
    print("---------------------")
    print(f"Expected annual return: {result.expected_annual_return:>7.2%}")
    print(f"Annual volatility:      {result.annual_volatility:>7.2%}")
    print(f"Sharpe ratio:           {result.sharpe_ratio:>7.3f}")


def pretty_print_backtest(result: BacktestResult) -> None:
    """Print backtest summary metrics in a readable format."""

    summary = result.summary

    print("Monthly Rebalance Backtest")
    print("--------------------------")
    print(f"Total return:            {summary['total_return']:>7.2%}")
    print(f"Annualized return:       {summary['annualized_return']:>7.2%}")
    print(f"Annualized volatility:   {summary['annualized_volatility']:>7.2%}")
    print(f"Sharpe ratio:            {summary['sharpe_ratio']:>7.3f}")
    print(f"Max drawdown:            {summary['max_drawdown']:>7.2%}")
    print(f"Final value:             {summary['final_value']:>7.3f}")
    print(f"Rebalances:              {len(result.weights_history):>7}")
