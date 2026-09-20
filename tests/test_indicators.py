import numpy as np
import pandas as pd
import pytest

from stock_advisor.technical.indicators import (
    average_true_range,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    support_resistance,
    volume_trend,
)


def test_sma_matches_manual_mean():
    series = pd.Series([1, 2, 3, 4, 5])
    result = sma(series, window=3)
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)
    assert pd.isna(result.iloc[1])


def test_ema_reacts_faster_than_sma_to_a_price_jump():
    series = pd.Series([10.0] * 30 + [20.0] * 5)
    sma_val = sma(series, window=10).iloc[-1]
    ema_val = ema(series, window=10).iloc[-1]
    assert ema_val > sma_val


def test_rsi_is_100_for_strictly_rising_series():
    series = pd.Series(np.arange(1, 40, dtype=float))
    result = rsi(series, window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_for_strictly_falling_series():
    series = pd.Series(np.arange(40, 1, -1, dtype=float))
    result = rsi(series, window=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_macd_histogram_is_zero_for_flat_series():
    series = pd.Series([100.0] * 60)
    result = macd(series)
    assert result["histogram"].iloc[-1] == pytest.approx(0.0)


def test_bollinger_bands_widen_with_volatility():
    calm = pd.Series([100.0] * 19 + [100.5])
    volatile = pd.Series([100.0] * 19 + [150.0])
    calm_bands = bollinger_bands(calm, window=20)
    volatile_bands = bollinger_bands(volatile, window=20)
    calm_width = calm_bands["upper"].iloc[-1] - calm_bands["lower"].iloc[-1]
    volatile_width = volatile_bands["upper"].iloc[-1] - volatile_bands["lower"].iloc[-1]
    assert volatile_width > calm_width


def test_average_true_range_is_positive_for_moving_series():
    high = pd.Series([10, 11, 12, 11, 13, 14, 15, 14, 16, 17, 18, 19, 20, 21, 22], dtype=float)
    low = high - 1
    close = high - 0.5
    result = average_true_range(high, low, close, window=14)
    assert result.iloc[-1] > 0


def test_volume_trend_above_one_when_recent_volume_higher():
    volume = pd.Series([100] * 50 + [300] * 10)
    result = volume_trend(volume, short_window=10, long_window=50)
    assert result.iloc[-1] > 1.0


def test_support_resistance_tracks_rolling_extremes():
    high = pd.Series([10, 12, 15, 11, 9, 8, 20, 7], dtype=float)
    low = high - 2
    result = support_resistance(high, low, window=4)
    assert result["resistance"].iloc[-1] == pytest.approx(20.0)
    assert result["support"].iloc[-1] == pytest.approx(low.iloc[4:8].min())
