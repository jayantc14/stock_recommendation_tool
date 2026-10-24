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


# --- extended indicators ---------------------------------------------------

from stock_advisor.technical import indicators as ind  # noqa: E402


def _ohlcv(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0.05, 1.5, n)))
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    volume = pd.Series(rng.integers(5_000, 50_000, n).astype(float))
    return pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": volume})


def test_volume_trend_is_nan_not_inf_when_baseline_volume_is_zero():
    result = volume_trend(pd.Series([0.0] * 60), short_window=10, long_window=50)
    assert result.iloc[-1] != result.iloc[-1]  # NaN


def test_adx_high_in_clean_trend_and_di_ordering():
    t = np.arange(120, dtype=float)
    close = pd.Series(100 + t)
    result = ind.adx(close + 1, close - 1, close)
    assert result["adx"].iloc[-1] > 40
    assert result["plus_di"].iloc[-1] > result["minus_di"].iloc[-1]


def test_stochastic_and_williams_r_extremes():
    t = np.arange(40, dtype=float)
    high, low, close = pd.Series(t + 1), pd.Series(t), pd.Series(t + 1)  # closes at the high
    assert ind.stochastic(high, low, close)["k"].iloc[-1] == pytest.approx(100.0)
    assert ind.williams_r(high, low, close).iloc[-1] == pytest.approx(0.0)


def test_mfi_is_100_when_price_only_rises():
    t = np.arange(1, 40, dtype=float)
    s = pd.Series(t)
    assert ind.mfi(s + 1, s - 1, s, pd.Series(np.full(len(s), 1000.0))).iloc[-1] == pytest.approx(100.0)


def test_obv_adds_volume_on_up_days_and_subtracts_on_down_days():
    close = pd.Series([10.0, 11.0, 10.0, 10.0, 12.0])
    volume = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    assert list(ind.obv(close, volume)) == [0.0, 200.0, -100.0, -100.0, 400.0]


def test_chaikin_money_flow_matches_plain_loop():
    df = _ohlcv(60)
    window = 20
    got = ind.chaikin_money_flow(df.High, df.Low, df.Close, df.Volume, window).iloc[-1]
    num = den = 0.0
    for i in range(len(df) - window, len(df)):
        h, l, c, v = df.High[i], df.Low[i], df.Close[i], df.Volume[i]
        num += ((c - l) - (h - c)) / (h - l) * v
        den += v
    assert got == pytest.approx(num / den)


def test_roc_distance_and_range_position():
    close = pd.Series([100.0, 110.0, 121.0])
    assert ind.roc(close, 1).iloc[-1] == pytest.approx(0.10)
    assert ind.distance_from_ma(pd.Series([10.0, 10.0, 13.0]), 3).iloc[-1] == pytest.approx(13 / 11 - 1)
    high, low = pd.Series([12.0, 15.0, 14.0]), pd.Series([8.0, 9.0, 10.0])
    pos = ind.range_position(high, low, pd.Series([10.0, 12.0, 12.0]), window=3)
    assert pos["position"].iloc[-1] == pytest.approx((12 - 8) / (15 - 8))
    assert pos["pct_from_high"].iloc[-1] == pytest.approx(12 / 15 - 1)


def test_percent_b_bandwidth_atr_percent_and_realized_vol():
    df = _ohlcv(80)
    pb = ind.percent_b(df.Close)
    assert (pb.dropna().abs() < 5).all()
    assert ind.bollinger_bandwidth(df.Close).iloc[-1] > 0
    assert ind.atr_percent(df.High, df.Low, df.Close).iloc[-1] == pytest.approx(
        average_true_range(df.High, df.Low, df.Close).iloc[-1] / df.Close.iloc[-1]
    )
    flat = pd.Series([100.0] * 40)
    assert ind.realized_volatility(flat).iloc[-1] == pytest.approx(0.0)


def test_aroon_100_at_new_high_and_ties_use_most_recent_bar():
    high = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert ind.aroon(high, high - 1, window=5)["up"].iloc[-1] == pytest.approx(100.0)
    tied = pd.Series([5.0, 5.0, 1.0, 1.0, 1.0, 1.0])  # equal highs at bars 0 and 1
    assert ind.aroon(tied, tied, window=5)["up"].iloc[-1] == pytest.approx(100 * (5 - 4) / 5)


def test_relative_strength_is_stock_minus_benchmark_return():
    stock = pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    bench = pd.Series([100.0, 105.0], index=stock.index)
    assert ind.relative_strength(stock, bench, window=1).iloc[-1] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "name,fn",
    [
        ("adx", lambda d: ind.adx(d.High, d.Low, d.Close)),
        ("stoch", lambda d: ind.stochastic(d.High, d.Low, d.Close)),
        ("willr", lambda d: ind.williams_r(d.High, d.Low, d.Close)),
        ("cci", lambda d: ind.cci(d.High, d.Low, d.Close)),
        ("mfi", lambda d: ind.mfi(d.High, d.Low, d.Close, d.Volume)),
        ("obv", lambda d: ind.obv(d.Close, d.Volume)),
        ("cmf", lambda d: ind.chaikin_money_flow(d.High, d.Low, d.Close, d.Volume)),
        ("roc", lambda d: ind.roc(d.Close, 20)),
        ("dist", lambda d: ind.distance_from_ma(d.Close, 50)),
        ("range", lambda d: ind.range_position(d.High, d.Low, d.Close, 100)),
        ("pctb", lambda d: ind.percent_b(d.Close)),
        ("bw", lambda d: ind.bollinger_bandwidth(d.Close)),
        ("atrp", lambda d: ind.atr_percent(d.High, d.Low, d.Close)),
        ("rvol", lambda d: ind.realized_volatility(d.Close)),
        ("aroon", lambda d: ind.aroon(d.High, d.Low)),
    ],
)
def test_no_lookahead_truncating_future_data_does_not_change_past_values(name, fn):
    df = _ohlcv(300)
    full = fn(df)
    cut = fn(df.iloc[:200])
    pd.testing.assert_frame_equal(
        pd.DataFrame(full).iloc[:200], pd.DataFrame(cut), check_exact=False, rtol=1e-9, atol=1e-9
    )
