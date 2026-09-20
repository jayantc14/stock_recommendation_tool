from __future__ import annotations

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


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def volume_trend(volume: pd.Series, short_window: int = 10, long_window: int = 50) -> pd.Series:
    """Ratio of recent average volume to longer-term average volume.

    >1 means volume is picking up relative to its baseline (a price move is
    better confirmed); <1 means it's fading (a move is more likely noise).
    """
    return sma(volume, short_window) / sma(volume, long_window)


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
