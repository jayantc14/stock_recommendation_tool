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
