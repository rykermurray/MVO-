"""Monthly portfolio rebalancing and performance backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pypfopt.exceptions import OptimizationError

from src.optimizer import (
    PortfolioResult,
    align_prices_with_market,
    optimize_portfolio,
    prepare_market_prices,
    validate_price_data,
    validate_weight_bounds,
)


@dataclass(frozen=True)
class BacktestResult:
    """Container for rebalance history, equity curve, and summary metrics."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    weights_history: pd.DataFrame
    rebalance_metrics: pd.DataFrame
    summary: dict[str, float]
    initial_capital: float
    rebalance_frequency: str
    lookback_days: int
    transaction_cost_bps: float
    optimization_failure_policy: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the backtest result."""

        return {
            "summary": self.summary,
            "initial_capital": float(self.initial_capital),
            "rebalance_frequency": self.rebalance_frequency,
            "lookback_days": int(self.lookback_days),
            "transaction_cost_bps": float(self.transaction_cost_bps),
            "optimization_failure_policy": self.optimization_failure_policy,
            "equity_curve": {
                date.strftime("%Y-%m-%d"): float(value)
                for date, value in self.equity_curve.items()
            },
            "daily_returns": {
                date.strftime("%Y-%m-%d"): float(value)
                for date, value in self.daily_returns.items()
            },
            "weights_history": _frame_to_nested_dict(self.weights_history),
            "rebalance_metrics": _frame_to_nested_dict(self.rebalance_metrics),
        }


def backtest_monthly_rebalance(
    prices: pd.DataFrame,
    *,
    market_prices: Optional[Union[pd.DataFrame, pd.Series]] = None,
    lookback_days: int = 252,
    initial_capital: float = 1.0,
    risk_free_rate: float = 0.02,
    min_weight: float = 0.02,
    max_weight: float = 0.15,
    transaction_cost_bps: float = 0.0,
    rebalance_frequency: str = "ME",
    min_observations: int = 126,
    optimization_failure_policy: str = "carry_forward",
    solver: Optional[str] = None,
) -> BacktestResult:
    """Run a monthly rebalance backtest using max-Sharpe optimized weights.

    The strategy trains on historical data available at each rebalance date,
    computes fresh CAPM/Ledoit-Wolf max-Sharpe weights, and holds those weights
    until the next rebalance. Returns are applied after the rebalance date, so
    the implementation avoids look-ahead bias.
    """

    clean_prices = validate_price_data(prices)
    clean_market_prices = prepare_market_prices(market_prices)

    if clean_market_prices is not None:
        clean_prices, clean_market_prices = align_prices_with_market(
            clean_prices, clean_market_prices
        )

    validate_weight_bounds(
        num_assets=clean_prices.shape[1],
        min_weight=min_weight,
        max_weight=max_weight,
    )
    _validate_backtest_inputs(
        lookback_days=lookback_days,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        min_observations=min_observations,
        optimization_failure_policy=optimization_failure_policy,
    )

    returns = clean_prices.pct_change().dropna(how="any")
    rebalance_dates = _get_rebalance_dates(
        clean_prices.index,
        frequency=rebalance_frequency,
        lookback_days=lookback_days,
        min_observations=min_observations,
    )

    if len(rebalance_dates) < 2:
        raise ValueError(
            "Not enough data to run a backtest. Reduce lookback_days or extend the "
            "date range."
        )

    portfolio_value = float(initial_capital)
    daily_return_segments: list[pd.Series] = []
    equity_segments: list[pd.Series] = []
    weights_records: list[dict[str, float]] = []
    metrics_records: list[dict[str, Any]] = []
    previous_weights = pd.Series(0.0, index=clean_prices.columns)

    for current_date, next_date in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        training_prices = _slice_lookback_window(
            clean_prices,
            end_date=current_date,
            lookback_days=lookback_days,
        )
        training_market = None
        if clean_market_prices is not None:
            training_market = clean_market_prices.loc[training_prices.index]

        try:
            optimization = optimize_portfolio(
                training_prices,
                market_prices=training_market,
                risk_free_rate=risk_free_rate,
                min_weight=min_weight,
                max_weight=max_weight,
                solver=solver,
            )
            weights = _weights_to_series(optimization, clean_prices.columns)
            expected_annual_return = optimization.expected_annual_return
            annual_volatility = optimization.annual_volatility
            sharpe_ratio = optimization.sharpe_ratio
            optimization_success = 1.0
        except (OptimizationError, ValueError):
            if optimization_failure_policy == "raise":
                raise
            weights = _fallback_weights(
                previous_weights,
                clean_prices.columns,
                policy=optimization_failure_policy,
            )
            expected_annual_return = np.nan
            annual_volatility = np.nan
            sharpe_ratio = np.nan
            optimization_success = 0.0

        holding_returns = returns.loc[
            (returns.index > current_date) & (returns.index <= next_date)
        ]
        if holding_returns.empty:
            continue

        turnover = float((weights - previous_weights).abs().sum())
        cost = turnover * (transaction_cost_bps / 10000)
        first_day = holding_returns.index[0]

        portfolio_returns = holding_returns.dot(weights)
        if cost:
            portfolio_returns.loc[first_day] -= cost

        equity_segment = portfolio_value * (1 + portfolio_returns).cumprod()
        portfolio_value = float(equity_segment.iloc[-1])

        daily_return_segments.append(portfolio_returns)
        equity_segments.append(equity_segment)
        weights_records.append({"date": current_date, **weights.to_dict()})
        metrics_records.append(
            {
                "date": current_date,
                "expected_annual_return": expected_annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "optimization_success": optimization_success,
                "turnover": turnover,
                "transaction_cost": cost,
            }
        )
        previous_weights = weights

    if not daily_return_segments:
        raise ValueError("Backtest produced no holding periods with valid returns.")

    daily_returns = pd.concat(daily_return_segments).sort_index()
    equity_curve = pd.concat(equity_segments).sort_index()
    weights_history = pd.DataFrame(weights_records).set_index("date").sort_index()
    rebalance_metrics = pd.DataFrame(metrics_records).set_index("date").sort_index()
    summary = compute_performance_summary(
        daily_returns,
        equity_curve,
        risk_free_rate=risk_free_rate,
        initial_capital=initial_capital,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        weights_history=weights_history,
        rebalance_metrics=rebalance_metrics,
        summary=summary,
        initial_capital=float(initial_capital),
        rebalance_frequency=rebalance_frequency,
        lookback_days=int(lookback_days),
        transaction_cost_bps=float(transaction_cost_bps),
        optimization_failure_policy=optimization_failure_policy,
    )


def compute_performance_summary(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    *,
    risk_free_rate: float = 0.02,
    initial_capital: float = 1.0,
    frequency: int = 252,
) -> dict[str, float]:
    """Compute annualized backtest performance metrics."""

    if daily_returns.empty or equity_curve.empty:
        raise ValueError("daily_returns and equity_curve must not be empty.")

    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)
    num_days = len(daily_returns)
    annualized_return = float((1 + total_return) ** (frequency / num_days) - 1)
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(frequency))
    sharpe_ratio = _safe_divide(
        annualized_return - risk_free_rate, annualized_volatility
    )
    max_drawdown = float(_compute_drawdown(equity_curve).min())
    positive_days = float((daily_returns > 0).mean())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_value": float(equity_curve.iloc[-1]),
        "positive_day_ratio": positive_days,
        "observations": float(num_days),
    }


def _get_rebalance_dates(
    index: pd.DatetimeIndex,
    *,
    frequency: str,
    lookback_days: int,
    min_observations: int,
) -> list[pd.Timestamp]:
    """Return last available trading dates for each rebalance period."""

    price_index = pd.DatetimeIndex(index).sort_values()
    offset = max(lookback_days, min_observations)
    eligible_index = price_index[offset - 1 :]

    if eligible_index.empty:
        return []

    periods = pd.Series(eligible_index, index=eligible_index).groupby(
        eligible_index.to_period(_normalize_frequency(frequency))
    )

    return [pd.Timestamp(group.iloc[-1]) for _, group in periods]


def _normalize_frequency(frequency: str) -> str:
    """Normalize pandas period frequency aliases."""

    if frequency.upper() in {"M", "ME", "MONTH", "MONTHLY"}:
        return "M"
    raise ValueError("Only monthly rebalance frequency is currently supported.")


def _slice_lookback_window(
    prices: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    """Slice a fixed-length historical training window ending on end_date."""

    end_position = prices.index.get_loc(end_date)
    start_position = max(0, end_position - lookback_days + 1)
    return prices.iloc[start_position : end_position + 1]


def _weights_to_series(result: PortfolioResult, columns: pd.Index) -> pd.Series:
    """Convert result weights to a column-aligned Series."""

    weights = pd.Series(result.weights, dtype=float)
    weights = weights.reindex(columns).fillna(0.0)
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("Optimized weights must have a positive total allocation.")
    return weights / total_weight


def _fallback_weights(
    previous_weights: pd.Series,
    columns: pd.Index,
    *,
    policy: str,
) -> pd.Series:
    """Return fallback weights when a rebalance optimization is infeasible."""

    if policy not in {"carry_forward", "equal_weight"}:
        raise ValueError(
            "optimization_failure_policy must be 'raise', 'carry_forward', or "
            "'equal_weight'."
        )

    if policy == "carry_forward" and previous_weights.sum() > 0:
        return previous_weights.copy()

    equal_weight = 1 / len(columns)
    return pd.Series(equal_weight, index=columns, dtype=float)


def _compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute percentage drawdowns from an equity curve."""

    running_peak = equity_curve.cummax()
    return equity_curve / running_peak - 1


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning 0 when volatility is effectively zero."""

    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _validate_backtest_inputs(
    *,
    lookback_days: int,
    initial_capital: float,
    transaction_cost_bps: float,
    min_observations: int,
    optimization_failure_policy: str,
) -> None:
    """Validate scalar backtest settings."""

    if lookback_days < 2:
        raise ValueError("lookback_days must be at least 2.")
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    if optimization_failure_policy not in {"raise", "carry_forward", "equal_weight"}:
        raise ValueError(
            "optimization_failure_policy must be 'raise', 'carry_forward', or "
            "'equal_weight'."
        )


def _frame_to_nested_dict(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Serialize a date-indexed numeric DataFrame for JSON output."""

    output: dict[str, dict[str, Any]] = {}
    for date, row in frame.iterrows():
        date_key = pd.Timestamp(date).strftime("%Y-%m-%d")
        output[date_key] = {}
        for column, value in row.dropna().items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                output[date_key][str(column)] = float(value)
            else:
                output[date_key][str(column)] = value
    return output
