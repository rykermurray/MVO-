"""Data loading utilities for historical market prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import yfinance as yf


DateLike = Union[str, date]


def download_price_data(
    tickers: Iterable[str],
    start_date: DateLike,
    end_date: DateLike,
    *,
    price_field: str = "Close",
    auto_adjust: bool = True,
    min_coverage: float = 0.90,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Download and clean historical price data from Yahoo Finance.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    start_date, end_date:
        Date range passed to yfinance. The end date is exclusive.
    price_field:
        Price column to extract. With ``auto_adjust=True``, "Close" is adjusted.
    auto_adjust:
        Whether yfinance should adjust prices for splits and dividends.
    min_coverage:
        Minimum non-null observation ratio required for each ticker.
    save_path:
        Optional CSV path for saving the cleaned prices.

    Returns
    -------
    pd.DataFrame
        Clean price matrix indexed by date, with tickers as columns.
    """

    ticker_list = normalize_tickers(tickers)
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in the interval (0, 1].")

    raw = yf.download(
        tickers=ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    prices = _extract_price_frame(raw, ticker_list, price_field)
    prices = _clean_price_frame(prices, min_coverage=min_coverage)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(output_path, index=True)

    return prices


def normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize tickers while preserving order and removing duplicates."""

    if isinstance(tickers, str):
        tickers = [tickers]

    normalized: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        clean_ticker = str(ticker).strip().upper()
        if not clean_ticker:
            continue
        if clean_ticker not in seen:
            normalized.append(clean_ticker)
            seen.add(clean_ticker)

    if not normalized:
        raise ValueError("Provide at least one ticker symbol.")

    return normalized


def _extract_price_frame(
    raw: pd.DataFrame, tickers: list[str], price_field: str
) -> pd.DataFrame:
    """Extract a ticker-by-price matrix from yfinance's single or multi-index output."""

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = raw.columns.get_level_values(0)
        level_one = raw.columns.get_level_values(1)

        if price_field in level_zero:
            prices = raw[price_field].copy()
        elif price_field in level_one:
            prices = raw.xs(price_field, axis=1, level=1).copy()
        else:
            available = sorted(set(map(str, level_zero)) | set(map(str, level_one)))
            raise ValueError(
                f"Price field '{price_field}' not found. Available fields: {available}"
            )
    else:
        if price_field not in raw.columns:
            available = [str(column) for column in raw.columns]
            raise ValueError(
                f"Price field '{price_field}' not found. Available fields: {available}"
            )
        prices = raw[[price_field]].copy()
        prices.columns = [tickers[0]]

    prices.columns = [str(column).strip().upper() for column in prices.columns]
    prices = prices.loc[:, ~prices.columns.duplicated()]

    return prices


def _clean_price_frame(prices: pd.DataFrame, *, min_coverage: float) -> pd.DataFrame:
    """Remove unusable columns and rows while avoiding look-ahead filling."""

    clean_prices = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    clean_prices = clean_prices.dropna(axis=1, how="all")

    coverage = clean_prices.notna().mean()
    low_coverage_tickers = coverage[coverage < min_coverage].index
    clean_prices = clean_prices.drop(columns=low_coverage_tickers)

    # Forward-fill isolated missing observations, then remove leading gaps.
    clean_prices = clean_prices.ffill().dropna(axis=0, how="any")
    clean_prices = clean_prices.loc[:, clean_prices.nunique(dropna=True) > 1]

    if clean_prices.empty:
        raise ValueError("No usable price data remains after cleaning.")

    return clean_prices

