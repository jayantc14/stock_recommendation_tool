from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    middle = sma(series, window)
    std = series.rolling(window=window).std()
    return pd.DataFrame(
        {
            "upper": middle + num_std * std,
            "middle": middle,
            "lower": middle - num_std * std,
        }
    )


def _wilder(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    return _wilder(true_range(high, low, close), window)


def volume_trend(volume: pd.Series, short_window: int = 10, long_window: int = 50) -> pd.Series:
    """Ratio of recent average volume to longer-term average volume.

    >1 means volume is picking up relative to its baseline (a price move is
    better confirmed); <1 means it's fading (a move is more likely noise).
    """
    baseline = sma(volume, long_window).replace(0, np.nan)
    return sma(volume, short_window) / baseline


def support_resistance(
    high: pd.Series,
    low: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "support": low.rolling(window=window).min(),
            "resistance": high.rolling(window=window).max(),
        }
    )


# --- Extended indicator set -------------------------------------------------
# Every function below is a pure time-series transform: the value at row t uses
# only data up to and including row t (no look-ahead), and early rows without
# enough history are NaN.


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.DataFrame:
    """Wilder's ADX with +DI / -DI. ADX measures trend *strength* (not direction)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = average_true_range(high, low, close, window)
    plus_di = 100 * _wilder(plus_dm, window) / atr
    minus_di = 100 * _wilder(minus_dm, window) / atr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame({"adx": _wilder(dx.dropna(), window).reindex(dx.index), "plus_di": plus_di, "minus_di": minus_di})


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> pd.DataFrame:
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    return pd.DataFrame({"k": k, "d": sma(k, d_window)})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return stochastic(high, low, close, window)["k"] - 100


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    mean = sma(typical, window)
    mean_dev = typical.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - mean) / (0.015 * mean_dev.replace(0, np.nan))


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Money Flow Index: a volume-weighted RSI on the typical price."""
    typical = (high + low + close) / 3
    flow = typical * volume
    direction = typical.diff()
    positive = flow.where(direction > 0, 0.0).rolling(window).sum()
    negative = flow.where(direction < 0, 0.0).rolling(window).sum()
    ratio = positive / negative.replace(0, np.nan)
    out = 100 - 100 / (1 + ratio)
    return out.where(negative != 0, 100.0).where(direction.rolling(window).count() == window)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume. A cumulative level: use its slope/ratio, not its absolute value."""
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


def chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    span = (high - low).replace(0, np.nan)
    multiplier = ((close - low) - (high - close)) / span
    flow = (multiplier * volume).fillna(0.0)
    return flow.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def roc(series: pd.Series, window: int) -> pd.Series:
    """Simple return over `window` bars (0.05 = +5%)."""
    return series.pct_change(window)


def distance_from_ma(series: pd.Series, window: int, kind: str = "sma") -> pd.Series:
    """Percentage distance of price from its moving average (0.05 = 5% above)."""
    ma = sma(series, window) if kind == "sma" else ema(series, window)
    return series / ma - 1


def range_position(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Where price sits in its trailing high/low range (0 = at low, 1 = at high).

    window=252 gives the 52-week position; window=20 is a Donchian-channel position.
    Also returns percentage distance below the range high / above the range low.
    """
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    return pd.DataFrame(
        {
            "position": (close - lowest) / (highest - lowest).replace(0, np.nan),
            "pct_from_high": close / highest - 1,
            "pct_from_low": close / lowest - 1,
        }
    )


def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    bands = bollinger_bands(series, window, num_std)
    return (series - bands["lower"]) / (bands["upper"] - bands["lower"]).replace(0, np.nan)


def bollinger_bandwidth(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    bands = bollinger_bands(series, window, num_std)
    return (bands["upper"] - bands["lower"]) / bands["middle"]


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return average_true_range(high, low, close, window) / close


def realized_volatility(series: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Annualised standard deviation of daily log returns."""
    log_ret = np.log(series).diff()
    return log_ret.rolling(window).std() * np.sqrt(periods_per_year)


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> pd.DataFrame:
    """Aroon up/down: how recently the window's high/low occurred (100 = today)."""
    # On ties the most recent bar counts, so search the reversed window.
    up = high.rolling(window + 1).apply(lambda x: 100.0 * (window - np.argmax(x[::-1])) / window, raw=True)
    down = low.rolling(window + 1).apply(lambda x: 100.0 * (window - np.argmin(x[::-1])) / window, raw=True)
    return pd.DataFrame({"up": up, "down": down, "oscillator": up - down})


def relative_strength(close: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    """Stock return minus benchmark return over `window` bars, aligned on dates."""
    bench = benchmark.reindex(close.index).ffill()
    return roc(close, window) - roc(bench, window)


# --- Advanced indicator set -------------------------------------------------
# Same contract as above: value at row t uses only data up to t. Functions that
# need the open price take it explicitly.


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average (most recent bar has the largest weight)."""
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hull_ma(series: pd.Series, window: int = 21) -> pd.Series:
    half = max(window // 2, 1)
    root = max(int(round(np.sqrt(window))), 1)
    return wma(2 * wma(series, half) - wma(series, window), root)


def kama(series: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman adaptive moving average: smooths hard in chop, tracks closely in trends."""
    er = efficiency_ratio(series, er_window)
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    sc = ((er * (fast_sc - slow_sc) + slow_sc) ** 2).to_numpy()
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    start = er_window - 1
    if len(values) <= start:
        return pd.Series(out, index=series.index)
    out[start] = values[start]
    for i in range(start + 1, len(values)):
        # A perfectly flat window has an undefined efficiency ratio; treat it as zero efficiency
        # (slowest smoothing) so one flat stretch cannot turn the whole recursion into NaN.
        step = sc[i] if not np.isnan(sc[i]) else slow_sc**2
        out[i] = out[i - 1] + step * (values[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def efficiency_ratio(series: pd.Series, window: int = 10) -> pd.Series:
    """|net change| / total path length over `window` bars: 1 = straight line, 0 = pure chop."""
    path = series.diff().abs().rolling(window).sum().replace(0, np.nan)
    return (series - series.shift(window)).abs() / path


def linreg_trend(series: pd.Series, window: int = 20) -> pd.DataFrame:
    """OLS line through the last `window` closes.

    `slope` is in price units per bar, `slope_pct` is slope / price (comparable across stocks),
    `r2` says how straight the trend is (1 = perfectly linear).
    """
    x = np.arange(window, dtype=float)
    x_dev = x - x.mean()
    x_var = (x_dev**2).sum()

    def _slope(y: np.ndarray) -> float:
        return float(np.dot(x_dev, y) / x_var)

    def _r2(y: np.ndarray) -> float:
        total = ((y - y.mean()) ** 2).sum()
        if total == 0:
            return np.nan
        slope = np.dot(x_dev, y) / x_var
        return float((slope**2) * x_var / total)

    slope = series.rolling(window).apply(_slope, raw=True)
    return pd.DataFrame(
        {"slope": slope, "slope_pct": slope / series, "r2": series.rolling(window).apply(_r2, raw=True)}
    )


def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Percentage price oscillator: MACD expressed as % of the slow EMA, so it is comparable across stocks."""
    slow_ema = ema(series, slow)
    line = 100 * (ema(series, fast) - slow_ema) / slow_ema
    sig = ema(line, signal)
    return pd.DataFrame({"ppo": line, "signal": sig, "histogram": line - sig})


def trix(series: pd.Series, window: int = 15) -> pd.Series:
    triple = ema(ema(ema(series, window), window), window)
    return 100 * triple.pct_change()


def cmo(series: pd.Series, window: int = 14) -> pd.Series:
    """Chande momentum oscillator (Wilder-smoothed), range -100..100."""
    delta = series.diff()
    gain = _wilder(delta.clip(lower=0), window)
    loss = _wilder(-delta.clip(upper=0), window)
    return 100 * (gain - loss) / (gain + loss).replace(0, np.nan)


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short: int = 7,
    mid: int = 14,
    long: int = 28,
) -> pd.Series:
    prev_close = close.shift(1)
    buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    def _avg(n: int) -> pd.Series:
        return buying_pressure.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)

    return 100 * (4 * _avg(short) + 2 * _avg(mid) + _avg(long)) / 7


def true_strength_index(series: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    delta = series.diff()
    num = ema(ema(delta, long), short)
    den = ema(ema(delta.abs(), long), short).replace(0, np.nan)
    return 100 * num / den


def know_sure_thing(series: pd.Series) -> pd.DataFrame:
    """Pring's KST: weighted sum of four smoothed rates of change, plus a 9-bar signal line."""
    line = (
        1 * sma(roc(series, 10), 10)
        + 2 * sma(roc(series, 15), 10)
        + 3 * sma(roc(series, 20), 10)
        + 4 * sma(roc(series, 30), 15)
    ) * 100
    return pd.DataFrame({"kst": line, "signal": sma(line, 9)})


def coppock_curve(series: pd.Series, long_roc: int = 14, short_roc: int = 11, window: int = 10) -> pd.Series:
    return wma(100 * (roc(series, long_roc) + roc(series, short_roc)), window)


def awesome_oscillator(high: pd.Series, low: pd.Series, short: int = 5, long: int = 34) -> pd.Series:
    median = (high + low) / 2
    return sma(median, short) - sma(median, long)


def stochastic_rsi(
    series: pd.Series,
    rsi_window: int = 14,
    stoch_window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    r = rsi(series, rsi_window)
    lowest, highest = r.rolling(stoch_window).min(), r.rolling(stoch_window).max()
    raw = 100 * (r - lowest) / (highest - lowest).replace(0, np.nan)
    k = sma(raw, smooth_k)
    return pd.DataFrame({"raw": raw, "k": k, "d": sma(k, smooth_d)})


def candle_anatomy(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Shape of the day's candle, all as fractions of the high-low range (scale-free).

    `body` is signed (balance of power: +1 = closed at the high after opening at the low).
    """
    span = (high - low).replace(0, np.nan)
    top, bottom = pd.concat([open_, close], axis=1).max(axis=1), pd.concat([open_, close], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "body": (close - open_) / span,
            "upper_shadow": (high - top) / span,
            "lower_shadow": (bottom - low) / span,
            "close_location": (close - low) / span,
        }
    )


def gap_percent(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap: today's open vs yesterday's close."""
    return open_ / close.shift(1) - 1


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    step: float = 0.02,
    max_step: float = 0.2,
) -> pd.DataFrame:
    """Wilder's parabolic SAR. `sar` is the stop level; `direction` is +1 (long) / -1 (short)."""
    h, l = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    n = len(h)
    sar = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"sar": sar, "direction": direction}, index=high.index)

    is_long = h[1] >= h[0] and (h[1] - h[0]) >= (l[0] - l[1])
    af = step
    ep = h[0] if is_long else l[0]
    cur = l[0] if is_long else h[0]
    sar[0], direction[0] = np.nan, np.nan  # first bar is only used to seed the state
    for i in range(1, n):
        cur = cur + af * (ep - cur)
        if is_long:
            cur = min(cur, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] <= cur:
                # Flip to short: the new SAR is the old extreme point, but never inside this bar's range.
                is_long, cur, ep, af = False, max(ep, h[i - 1], h[i]), l[i], step
            else:
                if h[i] > ep:
                    ep, af = h[i], min(af + step, max_step)
        else:
            cur = max(cur, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] >= cur:
                is_long, cur, ep, af = True, min(ep, l[i - 1], l[i]), h[i], step
            else:
                if l[i] < ep:
                    ep, af = l[i], min(af + step, max_step)
        sar[i], direction[i] = cur, 1.0 if is_long else -1.0
    return pd.DataFrame({"sar": sar, "direction": direction}, index=high.index)


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """ATR-band trend follower. `direction` is +1 (price above the line) or -1; `line` is the stop level."""
    atr = average_true_range(high, low, close, window).to_numpy()
    mid = ((high + low) / 2).to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        upper, lower = mid[i] + multiplier * atr[i], mid[i] - multiplier * atr[i]
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i], final_lower[i] = upper, lower
            direction[i] = 1.0 if c[i] >= mid[i] else -1.0
        else:
            final_upper[i] = upper if (upper < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
            final_lower[i] = lower if (lower > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]) else final_lower[i - 1]
            if direction[i - 1] == 1.0:
                direction[i] = -1.0 if c[i] < final_lower[i] else 1.0
            else:
                direction[i] = 1.0 if c[i] > final_upper[i] else -1.0
        line[i] = final_lower[i] if direction[i] == 1.0 else final_upper[i]
    return pd.DataFrame({"line": line, "direction": direction}, index=close.index)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku lines as they are known *at* each bar (cloud = spans computed `displacement` bars ago).

    The lagging span (chikou) is deliberately omitted: it is the close shifted back in time,
    which would leak future data into past rows.
    """

    def _mid(n: int) -> pd.Series:
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    t, k = _mid(tenkan), _mid(kijun)
    span_a = ((t + k) / 2).shift(displacement)
    span_b = _mid(senkou_b).shift(displacement)
    return pd.DataFrame(
        {
            "tenkan": t,
            "kijun": k,
            "span_a": span_a,
            "span_b": span_b,
            "cloud_top": pd.concat([span_a, span_b], axis=1).max(axis=1, skipna=False),
            "cloud_bottom": pd.concat([span_a, span_b], axis=1).min(axis=1, skipna=False),
        }
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    middle = ema(close, ema_window)
    width = multiplier * average_true_range(high, low, close, atr_window)
    return pd.DataFrame({"upper": middle + width, "middle": middle, "lower": middle - width})


def squeeze_ratio(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Bollinger width / Keltner width. Below 1 the bands sit inside the channel: volatility is compressed."""
    bb = bollinger_bands(close, window, 2.0)
    kc = keltner_channels(high, low, close, window, window, 1.5)
    return (bb["upper"] - bb["lower"]) / (kc["upper"] - kc["lower"]).replace(0, np.nan)


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    tr_sum = true_range(high, low, close).rolling(window).sum().replace(0, np.nan)
    plus = (high - low.shift(1)).abs().rolling(window).sum() / tr_sum
    minus = (low - high.shift(1)).abs().rolling(window).sum() / tr_sum
    return pd.DataFrame({"plus": plus, "minus": minus, "diff": plus - minus})


def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 13) -> pd.DataFrame:
    """Bull/bear power as a fraction of the EMA, so it is comparable across stocks."""
    base = ema(close, window)
    return pd.DataFrame({"bull_power": (high - base) / base, "bear_power": (low - base) / base})


def force_index(close: pd.Series, volume: pd.Series, window: int = 13) -> pd.Series:
    """EMA of price change x volume, divided by average rupee volume so it is comparable across stocks."""
    raw = ema(close.diff() * volume, window)
    return raw / (sma(close * volume, 50).replace(0, np.nan))


def rolling_vwap_distance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    typical = (high + low + close) / 3
    vwap = (typical * volume).rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)
    return close / vwap - 1


def chaikin_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/distribution line (cumulative level; use the oscillator for features)."""
    span = (high - low).replace(0, np.nan)
    return ((((close - low) - (high - close)) / span).fillna(0.0) * volume).cumsum()


def chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    ad = chaikin_ad_line(high, low, close, volume)
    return ema(ad, fast) - ema(ad, slow)


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """100 = pure chop, 0 = straight-line trend (typically >61.8 range-bound, <38.2 trending)."""
    span = (high.rolling(window).max() - low.rolling(window).min()).replace(0, np.nan)
    return 100 * np.log10(true_range(high, low, close).rolling(window).sum() / span) / np.log10(window)


def ulcer_index(close: pd.Series, window: int = 14) -> pd.Series:
    """RMS of percentage drawdown from the rolling high: downside-only risk."""
    drawdown_pct = 100 * (close / close.rolling(window).max() - 1)
    return np.sqrt((drawdown_pct**2).rolling(window).mean())


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    sq = np.log(high / low) ** 2
    return np.sqrt(sq.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(periods_per_year)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    term = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    return np.sqrt(term.rolling(window).mean().clip(lower=0)) * np.sqrt(periods_per_year)


def return_skewness(close: pd.Series, window: int = 60) -> pd.Series:
    return close.pct_change().rolling(window).skew()


def return_kurtosis(close: pd.Series, window: int = 60) -> pd.Series:
    return close.pct_change().rolling(window).kurt()


def drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    """Current percentage drawdown from the trailing `window`-bar closing high (0 = at the high)."""
    return close / close.rolling(window).max() - 1


def up_day_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    up = (close.diff() > 0).astype(float).where(close.diff().notna())
    return up.rolling(window).mean()


def streak(close: pd.Series) -> pd.Series:
    """Signed run length of consecutive up (+n) or down (-n) closes; flat days reset to 0."""
    sign = np.sign(close.diff()).fillna(0.0)
    group = (sign != sign.shift()).cumsum()
    return sign * (sign.groupby(group).cumcount() + 1) * (sign != 0)


def rolling_beta(close: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    bench = benchmark.reindex(close.index).ffill()
    r_s, r_b = close.pct_change(), bench.pct_change()
    return r_s.rolling(window).cov(r_b) / r_b.rolling(window).var().replace(0, np.nan)


def rolling_correlation(close: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    bench = benchmark.reindex(close.index).ffill()
    return close.pct_change().rolling(window).corr(bench.pct_change())


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    std = volume.rolling(window).std().replace(0, np.nan)
    return (volume - volume.rolling(window).mean()) / std


def average_dollar_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Average traded value (price x volume): a liquidity measure, not a directional signal."""
    return sma(close * volume, window)
