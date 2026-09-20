import pandas as pd
import pytest

from stock_advisor.data.price_fetcher import TickerNotFoundError, fetch_price_history


class _FakeYFTicker:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def history(self, period: str, interval: str) -> pd.DataFrame:
        return self._frame


def _sample_frame(rows: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": range(rows),
            "High": range(rows),
            "Low": range(rows),
            "Close": range(rows),
            "Volume": range(rows),
            "Dividends": [0.0] * rows,
            "Stock Splits": [0.0] * rows,
        },
        index=dates,
    )


def test_fetch_price_history_selects_ohlcv_columns(monkeypatch):
    monkeypatch.setattr(
        "stock_advisor.data.price_fetcher.yf.Ticker",
        lambda ticker: _FakeYFTicker(_sample_frame()),
    )

    result = fetch_price_history("RELIANCE.NS")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert result.index.name == "Date"
    assert len(result) == 5


def test_fetch_price_history_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(
        "stock_advisor.data.price_fetcher.yf.Ticker",
        lambda ticker: _FakeYFTicker(pd.DataFrame()),
    )

    with pytest.raises(TickerNotFoundError):
        fetch_price_history("NOT_A_TICKER")
