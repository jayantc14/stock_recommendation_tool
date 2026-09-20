from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

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

_WEIGHTS = {
    "trend": 0.35,
    "rsi": 0.20,
    "macd": 0.25,
    "bollinger": 0.10,
    "volume": 0.10,
}


@dataclass
class TechnicalSignal:
    momentum_score: float
    label: str
    entry_range: tuple[float, float] | None
    current_price: float
    sub_scores: dict[str, float] = field(default_factory=dict)
    raw_indicators: dict[str, float] = field(default_factory=dict)


def _trend_score(price: float, sma50: float, sma200: float) -> float:
    if pd.isna(sma50) or pd.isna(sma200):
        return 0.0
    if price > sma50 > sma200:
        return 1.0
    if price > sma50 and sma50 <= sma200:
        return 0.5
    if price < sma50 < sma200:
        return -1.0
    if price < sma50 and sma50 >= sma200:
        return -0.5
    return 0.0


def _rsi_score(rsi_value: float) -> float:
    if pd.isna(rsi_value):
        return 0.0
    if rsi_value >= 70:
        return -0.5
    if rsi_value >= 50:
        return 1.0
    if rsi_value >= 30:
        return -0.3
    return 0.5


def _macd_score(histogram: float, prev_histogram: float) -> float:
    if pd.isna(histogram) or pd.isna(prev_histogram):
        return 0.0
    rising = histogram > prev_histogram
    if histogram > 0:
        return 1.0 if rising else 0.3
    if histogram < 0:
        return -1.0 if not rising else -0.3
    return 0.0


def _bollinger_score(price: float, upper: float, lower: float) -> float:
    if pd.isna(upper) or pd.isna(lower) or upper == lower:
        return 0.0
    percent_b = (price - lower) / (upper - lower)
    if percent_b >= 0.8:
        return -0.3
    if percent_b <= 0.2:
        return 0.3
    return 0.0


def _volume_score(volume_ratio: float, trend_score: float) -> float:
    if pd.isna(volume_ratio):
        return 0.0
    direction = 1.0 if trend_score > 0 else (-1.0 if trend_score < 0 else 0.0)
    magnitude = min(max(volume_ratio - 1.0, -0.5), 0.5)
    return direction * magnitude


def compute_technical_signal(df: pd.DataFrame) -> TechnicalSignal:
    """Combine deterministic indicators into a momentum score and entry range.

    `df` must have Open/High/Low/Close/Volume columns and enough history for
    a 200-day SMA (at least ~200 rows) to be meaningful.
    """
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    rsi14 = rsi(close, 14)
    macd_df = macd(close)
    bands = bollinger_bands(close)
    vol_trend = volume_trend(volume)
    atr14 = average_true_range(high, low, close)
    levels = support_resistance(high, low)

    price = close.iloc[-1]

    trend = _trend_score(price, sma50.iloc[-1], sma200.iloc[-1])
    rsi_s = _rsi_score(rsi14.iloc[-1])
    macd_s = _macd_score(macd_df["histogram"].iloc[-1], macd_df["histogram"].iloc[-2])
    boll_s = _bollinger_score(price, bands["upper"].iloc[-1], bands["lower"].iloc[-1])
    vol_s = _volume_score(vol_trend.iloc[-1], trend)

    sub_scores = {
        "trend": trend,
        "rsi": rsi_s,
        "macd": macd_s,
        "bollinger": boll_s,
        "volume": vol_s,
    }
    weighted = sum(_WEIGHTS[k] * v for k, v in sub_scores.items())
    momentum_score = round(weighted * 100, 2)

    if momentum_score >= 20:
        label = "Bullish"
    elif momentum_score <= -20:
        label = "Bearish"
    else:
        label = "Neutral"

    support = levels["support"].iloc[-1]
    atr = atr14.iloc[-1]
    entry_range = None
    if not pd.isna(support) and not pd.isna(atr):
        entry_range = (round(float(support), 2), round(float(support + 0.5 * atr), 2))

    return TechnicalSignal(
        momentum_score=momentum_score,
        label=label,
        entry_range=entry_range,
        current_price=round(float(price), 2),
        sub_scores={k: round(v, 3) for k, v in sub_scores.items()},
        raw_indicators={
            "sma50": round(float(sma50.iloc[-1]), 2) if not pd.isna(sma50.iloc[-1]) else None,
            "sma200": round(float(sma200.iloc[-1]), 2) if not pd.isna(sma200.iloc[-1]) else None,
            "rsi14": round(float(rsi14.iloc[-1]), 2) if not pd.isna(rsi14.iloc[-1]) else None,
            "macd_histogram": round(float(macd_df["histogram"].iloc[-1]), 3),
            "bollinger_upper": round(float(bands["upper"].iloc[-1]), 2) if not pd.isna(bands["upper"].iloc[-1]) else None,
            "bollinger_lower": round(float(bands["lower"].iloc[-1]), 2) if not pd.isna(bands["lower"].iloc[-1]) else None,
            "volume_trend": round(float(vol_trend.iloc[-1]), 3) if not pd.isna(vol_trend.iloc[-1]) else None,
            "atr14": round(float(atr), 2) if not pd.isna(atr) else None,
            "support": round(float(support), 2) if not pd.isna(support) else None,
            "resistance": round(float(levels["resistance"].iloc[-1]), 2) if not pd.isna(levels["resistance"].iloc[-1]) else None,
        },
    )
