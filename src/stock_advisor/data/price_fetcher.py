from __future__ import annotations

import pandas as pd
import yfinance as yf


class TickerNotFoundError(ValueError):
    pass


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history for `ticker` via yfinance.

    `ticker` must carry the yfinance exchange suffix for non-US listings,
    e.g. "RELIANCE.NS" (NSE) or "RELIANCE.BO" (BSE).
    """
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty:
        raise TickerNotFoundError(f"No price data returned for ticker '{ticker}'")

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.index.name = "Date"
    return data
