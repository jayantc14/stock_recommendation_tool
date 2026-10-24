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


# --- advanced indicators ---------------------------------------------------


def _ohlcv_full(n: int = 400, seed: int = 11) -> pd.DataFrame:
    df = _ohlcv(n, seed)
    rng = np.random.default_rng(seed + 1)
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0]) * (1 + rng.normal(0, 0.003, n))
    df["High"] = df[["High", "Open"]].max(axis=1)
    df["Low"] = df[["Low", "Open"]].min(axis=1)
    return df


def test_wma_weights_recent_bars_more():
    assert ind.wma(pd.Series([1.0, 2.0, 3.0]), 3).iloc[-1] == pytest.approx((1 + 4 + 9) / 6)


def test_hull_ma_has_less_lag_than_sma_on_a_linear_series():
    s = pd.Series(np.arange(100, dtype=float))
    assert abs(ind.hull_ma(s, 21).iloc[-1] - s.iloc[-1]) < abs(ind.sma(s, 21).iloc[-1] - s.iloc[-1])


def test_efficiency_ratio_extremes():
    assert ind.efficiency_ratio(pd.Series(np.arange(30, dtype=float))).iloc[-1] == pytest.approx(1.0)
    assert ind.efficiency_ratio(pd.Series([1.0, 2.0] * 15)).iloc[-1] == pytest.approx(0.0)


def test_kama_survives_a_flat_stretch_and_tracks_a_straight_line():
    # Regression: a flat window makes the efficiency ratio 0/0; that used to turn all later values into NaN.
    s = pd.Series([100.0] * 30 + list(np.arange(100.0, 160.0)))
    out = ind.kama(s)
    assert out.iloc[-1] == out.iloc[-1]  # not NaN
    assert 140 < out.iloc[-1] < 160


def test_linreg_trend_recovers_slope_and_perfect_fit():
    s = pd.Series(2.0 * np.arange(40) + 5)
    out = ind.linreg_trend(s, 20)
    assert out["slope"].iloc[-1] == pytest.approx(2.0)
    assert out["r2"].iloc[-1] == pytest.approx(1.0)
    assert out["slope_pct"].iloc[-1] == pytest.approx(2.0 / s.iloc[-1])


def test_flat_series_gives_zero_for_ppo_kst_coppock_and_trix():
    flat = pd.Series([50.0] * 120)
    assert ind.ppo(flat)["ppo"].iloc[-1] == pytest.approx(0.0)
    assert ind.know_sure_thing(flat)["kst"].iloc[-1] == pytest.approx(0.0)
    assert ind.coppock_curve(flat).iloc[-1] == pytest.approx(0.0)
    assert ind.trix(flat).iloc[-1] == pytest.approx(0.0)


def test_cmo_and_tsi_saturate_on_one_way_series():
    up, down = pd.Series(np.arange(1, 80, dtype=float)), pd.Series(np.arange(80, 1, -1, dtype=float))
    assert ind.cmo(up).iloc[-1] == pytest.approx(100.0)
    assert ind.cmo(down).iloc[-1] == pytest.approx(-100.0)
    assert ind.true_strength_index(up).iloc[-1] == pytest.approx(100.0)


def test_bounded_oscillators_stay_in_range_on_random_data():
    df = _ohlcv_full()
    uo = ind.ultimate_oscillator(df.High, df.Low, df.Close).dropna()
    srsi = ind.stochastic_rsi(df.Close)[["raw", "k", "d"]].dropna()
    assert uo.between(0, 100).all()
    assert srsi.stack().between(-1e-9, 100 + 1e-9).all()


def test_awesome_oscillator_on_linear_median_is_constant():
    t = pd.Series(np.arange(100, dtype=float))
    # SMA5 of a ramp lags 2 bars, SMA34 lags 16.5 bars -> 14.5
    assert ind.awesome_oscillator(t + 1, t - 1).iloc[-1] == pytest.approx(14.5)


def test_candle_anatomy_fractions():
    out = ind.candle_anatomy(pd.Series([11.0]), pd.Series([14.0]), pd.Series([10.0]), pd.Series([13.0]))
    assert out["body"].iloc[0] == pytest.approx(0.5)
    assert out["upper_shadow"].iloc[0] == pytest.approx(0.25)
    assert out["lower_shadow"].iloc[0] == pytest.approx(0.25)
    assert out["close_location"].iloc[0] == pytest.approx(0.75)


def test_gap_percent():
    assert ind.gap_percent(pd.Series([100.0, 105.0]), pd.Series([100.0, 110.0])).iloc[-1] == pytest.approx(0.05)


def test_parabolic_sar_trails_below_in_uptrend_and_above_in_downtrend():
    up = pd.Series(np.arange(100, 160, dtype=float))
    out = ind.parabolic_sar(up + 1, up - 1)
    assert out["direction"].iloc[-1] == 1 and out["sar"].iloc[-1] < (up - 1).iloc[-1]
    down = pd.Series(np.arange(160, 100, -1, dtype=float))
    out = ind.parabolic_sar(down + 1, down - 1)
    assert out["direction"].iloc[-1] == -1 and out["sar"].iloc[-1] > (down + 1).iloc[-1]


def test_supertrend_band_logic_on_random_walk():
    df = _ohlcv_full(500)
    st = ind.supertrend(df.High, df.Low, df.Close).dropna()
    up, down = st["direction"] == 1, st["direction"] == -1
    assert (st["line"][up] < df.Close.loc[st.index][up]).all()
    assert (st["line"][down] > df.Close.loc[st.index][down]).all()
    # inside a run of the same direction the stop only ratchets toward price
    run = (st["direction"] != st["direction"].shift()).cumsum()
    for _, g in st.groupby(run):
        d = g["line"].diff().dropna()
        assert (d >= -1e-9).all() if g["direction"].iloc[0] == 1 else (d <= 1e-9).all()


def test_supertrend_flips_down_after_a_crash():
    close = pd.Series(list(np.linspace(100, 150, 60)) + list(np.linspace(150, 90, 30)))
    st = ind.supertrend(close + 1, close - 1, close)
    assert st["direction"].iloc[59] == 1 and st["direction"].iloc[-1] == -1


def test_ichimoku_lines_and_cloud_use_only_past_data():
    df = _ohlcv_full(200)
    ich = ind.ichimoku(df.High, df.Low, df.Close)
    i = 120
    assert ich["tenkan"].iloc[i] == pytest.approx((df.High.iloc[i - 8 : i + 1].max() + df.Low.iloc[i - 8 : i + 1].min()) / 2)
    j = i - 26
    expected_a = (ich["tenkan"].iloc[j] + ich["kijun"].iloc[j]) / 2
    assert ich["span_a"].iloc[i] == pytest.approx(expected_a)
    assert (ich["cloud_top"].dropna() >= ich["cloud_bottom"].dropna()).all()


def test_keltner_width_and_squeeze_ratio():
    df = _ohlcv_full()
    kc = ind.keltner_channels(df.High, df.Low, df.Close, 20, 10, 2.0)
    atr = average_true_range(df.High, df.Low, df.Close, 10)
    assert (kc["upper"] - kc["lower"]).iloc[-1] == pytest.approx(4 * atr.iloc[-1])
    assert ind.squeeze_ratio(df.High, df.Low, df.Close).dropna().gt(0).all()


def test_vortex_elder_ray_force_index_in_clean_uptrend():
    t = pd.Series(np.arange(100, 180, dtype=float))
    vol = pd.Series(np.full(80, 1000.0))
    v = ind.vortex(t + 1, t - 1, t)
    assert v["plus"].iloc[-1] > v["minus"].iloc[-1]
    er = ind.elder_ray(t + 1, t - 1, t)
    assert er["bull_power"].iloc[-1] > er["bear_power"].iloc[-1]
    assert ind.force_index(t, vol).iloc[-1] > 0


def test_rolling_vwap_distance_is_zero_for_flat_price():
    s = pd.Series([100.0] * 30)
    assert ind.rolling_vwap_distance(s, s, s, pd.Series(np.full(30, 500.0))).iloc[-1] == pytest.approx(0.0)


def test_chaikin_ad_line_accumulates_full_volume_when_closing_at_high():
    high, low = pd.Series([10.0, 11.0, 12.0]), pd.Series([9.0, 10.0, 11.0])
    out = ind.chaikin_ad_line(high, low, high, pd.Series([100.0, 200.0, 300.0]))
    assert list(out) == [100.0, 300.0, 600.0]


def test_choppiness_low_in_trend_high_in_chop():
    t = pd.Series(np.arange(100, 160, dtype=float))
    assert ind.choppiness_index(t + 0.5, t - 0.5, t).iloc[-1] < 38.2
    z = pd.Series([100.0, 101.0] * 30)
    assert ind.choppiness_index(z + 0.5, z - 0.5, z).iloc[-1] > 61.8


def test_ulcer_index_zero_when_always_at_new_highs_and_positive_after_a_fall():
    assert ind.ulcer_index(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1] == pytest.approx(0.0)
    assert ind.ulcer_index(pd.Series(list(np.arange(50, 100, dtype=float)) + [80.0] * 10)).iloc[-1] > 0


def test_range_based_volatilities():
    h, l = pd.Series([101.0] * 40), pd.Series([100.0] * 40)
    expected = np.sqrt(np.log(101 / 100) ** 2 / (4 * np.log(2))) * np.sqrt(252)
    assert ind.parkinson_volatility(h, l).iloc[-1] == pytest.approx(expected)
    df = _ohlcv_full()
    gk = ind.garman_klass_volatility(df.Open, df.High, df.Low, df.Close).dropna()
    assert (gk >= 0).all() and np.isfinite(gk).all()


def test_drawdown_up_day_ratio_and_streak():
    assert ind.drawdown(pd.Series([100.0, 90.0]), 2).iloc[-1] == pytest.approx(-0.10)
    assert ind.up_day_ratio(pd.Series([1.0, 2.0, 3.0, 2.0, 3.0]), 4).iloc[-1] == pytest.approx(0.75)
    assert list(ind.streak(pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0]))) == [0, 1, 2, -1, -2, 0, 1]


def test_beta_and_correlation_recover_a_known_relationship():
    rng = np.random.default_rng(5)
    bench_ret = rng.normal(0, 0.01, 200)
    bench = pd.Series(100 * np.cumprod(1 + bench_ret))
    stock = pd.Series(100 * np.cumprod(1 + 2 * bench_ret))
    assert ind.rolling_beta(stock, bench, 60).iloc[-1] == pytest.approx(2.0, rel=1e-6)
    assert ind.rolling_correlation(stock, bench, 60).iloc[-1] == pytest.approx(1.0)


def test_volume_zscore_and_dollar_volume():
    vol = pd.Series([100.0] * 19 + [1000.0])
    assert ind.volume_zscore(vol, 20).iloc[-1] > 3
    assert ind.average_dollar_volume(pd.Series([10.0] * 5), pd.Series([100.0] * 5), 5).iloc[-1] == pytest.approx(1000.0)
    assert np.isnan(ind.volume_zscore(pd.Series([5.0] * 30), 20).iloc[-1])


@pytest.mark.parametrize(
    "name,fn",
    [
        ("wma", lambda d: ind.wma(d.Close, 20)),
        ("hull", lambda d: ind.hull_ma(d.Close, 21)),
        ("kama", lambda d: ind.kama(d.Close)),
        ("er", lambda d: ind.efficiency_ratio(d.Close)),
        ("linreg", lambda d: ind.linreg_trend(d.Close, 20)),
        ("ppo", lambda d: ind.ppo(d.Close)),
        ("trix", lambda d: ind.trix(d.Close)),
        ("cmo", lambda d: ind.cmo(d.Close)),
        ("uo", lambda d: ind.ultimate_oscillator(d.High, d.Low, d.Close)),
        ("tsi", lambda d: ind.true_strength_index(d.Close)),
        ("kst", lambda d: ind.know_sure_thing(d.Close)),
        ("coppock", lambda d: ind.coppock_curve(d.Close)),
        ("ao", lambda d: ind.awesome_oscillator(d.High, d.Low)),
        ("srsi", lambda d: ind.stochastic_rsi(d.Close)),
        ("candle", lambda d: ind.candle_anatomy(d.Open, d.High, d.Low, d.Close)),
        ("gap", lambda d: ind.gap_percent(d.Open, d.Close)),
        ("sar", lambda d: ind.parabolic_sar(d.High, d.Low)),
        ("supertrend", lambda d: ind.supertrend(d.High, d.Low, d.Close)),
        ("ichimoku", lambda d: ind.ichimoku(d.High, d.Low, d.Close)),
        ("keltner", lambda d: ind.keltner_channels(d.High, d.Low, d.Close)),
        ("squeeze", lambda d: ind.squeeze_ratio(d.High, d.Low, d.Close)),
        ("vortex", lambda d: ind.vortex(d.High, d.Low, d.Close)),
        ("elder", lambda d: ind.elder_ray(d.High, d.Low, d.Close)),
        ("force", lambda d: ind.force_index(d.Close, d.Volume)),
        ("vwap", lambda d: ind.rolling_vwap_distance(d.High, d.Low, d.Close, d.Volume)),
        ("ad", lambda d: ind.chaikin_ad_line(d.High, d.Low, d.Close, d.Volume)),
        ("chaikin_osc", lambda d: ind.chaikin_oscillator(d.High, d.Low, d.Close, d.Volume)),
        ("chop", lambda d: ind.choppiness_index(d.High, d.Low, d.Close)),
        ("ulcer", lambda d: ind.ulcer_index(d.Close)),
        ("parkinson", lambda d: ind.parkinson_volatility(d.High, d.Low)),
        ("gk", lambda d: ind.garman_klass_volatility(d.Open, d.High, d.Low, d.Close)),
        ("skew", lambda d: ind.return_skewness(d.Close)),
        ("kurt", lambda d: ind.return_kurtosis(d.Close)),
        ("dd", lambda d: ind.drawdown(d.Close)),
        ("updays", lambda d: ind.up_day_ratio(d.Close)),
        ("streak", lambda d: ind.streak(d.Close)),
        ("beta", lambda d: ind.rolling_beta(d.Close, d.Close.rolling(5).mean())),
        ("corr", lambda d: ind.rolling_correlation(d.Close, d.Close.rolling(5).mean())),
        ("volz", lambda d: ind.volume_zscore(d.Volume)),
        ("dollarvol", lambda d: ind.average_dollar_volume(d.Close, d.Volume)),
    ],
)
def test_advanced_indicators_have_no_lookahead(name, fn):
    df = _ohlcv_full(400)
    full, cut = pd.DataFrame(fn(df)), pd.DataFrame(fn(df.iloc[:300]))
    pd.testing.assert_frame_equal(full.iloc[:300], cut, check_exact=False, rtol=1e-9, atol=1e-9)
