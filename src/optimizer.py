"""Mean-variance portfolio optimization using PyPortfolioOpt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


@dataclass(frozen=True)
class PortfolioResult:
    """Container for optimized weights and annualized performance metrics."""

    weights: dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    risk_free_rate: float
    min_weight: float
    max_weight: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the result."""

        return asdict(self)


def compute_expected_returns(
    prices: pd.DataFrame,
    *,
    market_prices: Optional[Union[pd.DataFrame, pd.Series]] = None,
    risk_free_rate: float = 0.02,
    frequency: int = 252,
) -> pd.Series:
    """Compute CAPM-based expected annual returns."""

    clean_prices = validate_price_data(prices)
    clean_market_prices = prepare_market_prices(market_prices)

    if clean_market_prices is not None:
        clean_prices, clean_market_prices = align_prices_with_market(
            clean_prices, clean_market_prices
        )

    return expected_returns.capm_return(
        clean_prices,
        market_prices=clean_market_prices,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
    )


def compute_covariance_matrix(
    prices: pd.DataFrame,
    *,
    frequency: int = 252,
    shrinkage_target: str = "constant_variance",
) -> pd.DataFrame:
    """Compute a Ledoit-Wolf shrinkage covariance matrix."""

    clean_prices = validate_price_data(prices)
    return risk_models.CovarianceShrinkage(
        clean_prices,
        frequency=frequency,
    ).ledoit_wolf(shrinkage_target=shrinkage_target)


def optimize_portfolio(
    prices: pd.DataFrame,
    *,
    market_prices: Optional[Union[pd.DataFrame, pd.Series]] = None,
    risk_free_rate: float = 0.02,
    min_weight: float = 0.02,
    max_weight: float = 0.15,
    frequency: int = 252,
    solver: Optional[str] = None,
) -> PortfolioResult:
    """Optimize a long-only max-Sharpe portfolio and return clean weights."""

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

    mu = compute_expected_returns(
        clean_prices,
        market_prices=clean_market_prices,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
    )
    cov_matrix = compute_covariance_matrix(clean_prices, frequency=frequency)

    return optimize_from_estimates(
        mu,
        cov_matrix,
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        solver=solver,
    )


def optimize_from_estimates(
    expected_return_series: pd.Series,
    covariance_matrix: pd.DataFrame,
    *,
    risk_free_rate: float = 0.02,
    min_weight: float = 0.02,
    max_weight: float = 0.15,
    solver: Optional[str] = None,
) -> PortfolioResult:
    """Optimize a max-Sharpe portfolio from precomputed estimates."""

    expected_return_series = expected_return_series.dropna()
    covariance_matrix = covariance_matrix.loc[
        expected_return_series.index, expected_return_series.index
    ]

    validate_weight_bounds(
        num_assets=len(expected_return_series),
        min_weight=min_weight,
        max_weight=max_weight,
    )

    if expected_return_series.max() <= risk_free_rate:
        raise ValueError(
            "At least one asset must have an expected return above the risk-free rate "
            "to compute a max-Sharpe portfolio."
        )

    efficient_frontier = EfficientFrontier(
        expected_return_series,
        covariance_matrix,
        weight_bounds=(min_weight, max_weight),
        solver=solver,
    )
    efficient_frontier.max_sharpe(risk_free_rate=risk_free_rate)
    weights = {
        ticker: float(weight)
        for ticker, weight in efficient_frontier.clean_weights(
            cutoff=0.0, rounding=6
        ).items()
    }

    expected_return, volatility, sharpe_ratio = efficient_frontier.portfolio_performance(
        risk_free_rate=risk_free_rate
    )

    return PortfolioResult(
        weights=weights,
        expected_annual_return=float(expected_return),
        annual_volatility=float(volatility),
        sharpe_ratio=float(sharpe_ratio),
        risk_free_rate=float(risk_free_rate),
        min_weight=float(min_weight),
        max_weight=float(max_weight),
    )


def validate_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a price matrix before estimation or optimization."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    clean_prices = prices.copy()
    clean_prices.index = pd.to_datetime(clean_prices.index)
    clean_prices = clean_prices.sort_index()
    clean_prices = clean_prices.apply(pd.to_numeric, errors="coerce")
    clean_prices = clean_prices.replace([np.inf, -np.inf], np.nan)
    clean_prices = clean_prices.dropna(axis=1, how="all")
    clean_prices = clean_prices.ffill().dropna(axis=0, how="any")
    clean_prices = clean_prices.loc[:, clean_prices.nunique(dropna=True) > 1]

    if clean_prices.shape[0] < 2:
        raise ValueError("At least two rows of price data are required.")
    if clean_prices.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    return clean_prices


def prepare_market_prices(
    market_prices: Optional[Union[pd.DataFrame, pd.Series]],
) -> Optional[pd.DataFrame]:
    """Normalize optional benchmark prices for CAPM expected returns."""

    if market_prices is None:
        return None

    if isinstance(market_prices, pd.Series):
        market_frame = market_prices.to_frame(name=market_prices.name or "MARKET")
    elif isinstance(market_prices, pd.DataFrame):
        market_frame = market_prices.copy()
    else:
        raise TypeError("market_prices must be a pandas DataFrame, Series, or None.")

    if market_frame.shape[1] != 1:
        raise ValueError("market_prices must contain exactly one benchmark column.")

    market_frame.index = pd.to_datetime(market_frame.index)
    market_frame = market_frame.sort_index()
    market_frame = market_frame.apply(pd.to_numeric, errors="coerce")
    market_frame = market_frame.replace([np.inf, -np.inf], np.nan)
    market_frame = market_frame.ffill().dropna(axis=0, how="any")

    if market_frame.shape[0] < 2:
        raise ValueError("At least two rows of market price data are required.")

    return market_frame


def align_prices_with_market(
    prices: pd.DataFrame, market_prices: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align asset and benchmark prices to the same trading dates."""

    common_index = prices.index.intersection(market_prices.index)
    if len(common_index) < 2:
        raise ValueError(
            "Asset prices and market prices must share at least two dates."
        )

    return prices.loc[common_index], market_prices.loc[common_index]


def validate_weight_bounds(
    *, num_assets: int, min_weight: float, max_weight: float
) -> None:
    """Check whether the requested per-asset weight bounds are feasible."""

    if num_assets <= 0:
        raise ValueError("num_assets must be positive.")
    if min_weight < 0:
        raise ValueError("min_weight must be non-negative for long-only portfolios.")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")
    if min_weight > max_weight:
        raise ValueError("min_weight cannot exceed max_weight.")

    min_total = num_assets * min_weight
    max_total = num_assets * max_weight
    tolerance = 1e-9

    if min_total > 1 + tolerance:
        raise ValueError(
            f"Bounds are infeasible: {num_assets} assets at a minimum weight of "
            f"{min_weight:.2%} require {min_total:.2%} total allocation."
        )
    if max_total < 1 - tolerance:
        raise ValueError(
            f"Bounds are infeasible: {num_assets} assets at a maximum weight of "
            f"{max_weight:.2%} allow only {max_total:.2%} total allocation."
        )

