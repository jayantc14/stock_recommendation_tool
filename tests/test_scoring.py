import numpy as np
import pandas as pd
import pytest

from stock_advisor.technical.scoring import compute_technical_signal


def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(close), freq="D")
    close_series = pd.Series(close, index=dates)
    return pd.DataFrame(
        {
            "Open": close_series.shift(1).fillna(close_series.iloc[0]),
            "High": close_series * 1.01,
            "Low": close_series * 0.99,
            "Close": close_series,
            "Volume": np.full(len(close), 1_000_000),
        },
        index=dates,
    )


def test_strong_uptrend_scores_bullish():
    t = np.arange(250)
    close = 100 * (1 + 0.20 * t / 250) + 1.5 * np.sin(t / 5)
    df = _make_ohlcv(close)

    signal = compute_technical_signal(df)

    assert signal.label == "Bullish"
    assert signal.momentum_score > 20


def test_strong_downtrend_scores_bearish():
    t = np.arange(250)
    close = 300 * (1 - 0.20 * t / 250) + 1.5 * np.sin(t / 5)
    df = _make_ohlcv(close)

    signal = compute_technical_signal(df)

    assert signal.label == "Bearish"
    assert signal.momentum_score < -20


def test_entry_range_is_ordered_and_near_support():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, 220))
    df = _make_ohlcv(close)

    signal = compute_technical_signal(df)

    assert signal.entry_range is not None
    low, high = signal.entry_range
    assert low <= high
    assert signal.raw_indicators["support"] == pytest.approx(low)
