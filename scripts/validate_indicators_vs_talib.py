"""Cross-check our indicators against TA-Lib on cached real price data.

Needs the optional dependency:  pip install -e ".[validate]"
and the price cache:            python scripts/build_price_dataset.py

Usage:
    python scripts/validate_indicators_vs_talib.py
    python scripts/validate_indicators_vs_talib.py RELIANCE.NS TCS.NS
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import talib

from stock_advisor.data.history_store import load_prices
from stock_advisor.technical import indicators as ind

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "PATANJALI.NS", "YESBANK.NS"]
TAIL = 1000  # compare after the smoothing warm-up


def _diff(mine: pd.Series, ref: np.ndarray) -> tuple[float, int]:
    m, r = mine.iloc[-TAIL:].to_numpy(float), np.asarray(ref, dtype=float)[-TAIL:]
    ok = ~np.isnan(m) & ~np.isnan(r)
    return float(np.abs(m[ok] - r[ok]).max()), int((np.isnan(m) != np.isnan(r)).sum())


def main() -> None:
    tickers = sys.argv[1:] or DEFAULT_TICKERS
    rows = []
    for t in tickers:
        df = load_prices(t)
        h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
        H, L, C, V = (df[k].to_numpy(float) for k in ["High", "Low", "Close", "Volume"])
        a, st, ar = ind.adx(h, l, c), ind.stochastic(h, l, c), ind.aroon(h, l)
        fastk, fastd = talib.STOCHF(H, L, C, fastk_period=14, fastd_period=3, fastd_matype=0)
        aroon_down, aroon_up = talib.AROON(H, L, 25)
        checks = {
            "rsi": (ind.rsi(c), talib.RSI(C, 14)),
            "atr": (ind.average_true_range(h, l, c), talib.ATR(H, L, C, 14)),
            "adx": (a["adx"], talib.ADX(H, L, C, 14)),
            "plus_di": (a["plus_di"], talib.PLUS_DI(H, L, C, 14)),
            "minus_di": (a["minus_di"], talib.MINUS_DI(H, L, C, 14)),
            "stoch_k": (st["k"], fastk),
            "stoch_d": (st["d"], fastd),
            "williams_r": (ind.williams_r(h, l, c), talib.WILLR(H, L, C, 14)),
            "cci": (ind.cci(h, l, c), talib.CCI(H, L, C, 20)),
            "mfi": (ind.mfi(h, l, c, v), talib.MFI(H, L, C, V, 14)),
            "aroon_up": (ar["up"], aroon_up),
            "aroon_down": (ar["down"], aroon_down),
            "roc_10_pct": (ind.roc(c, 10) * 100, talib.ROC(C, 10)),
            "obv_daily_change": (ind.obv(c, v).diff(), pd.Series(talib.OBV(C, V)).diff().to_numpy()),
        }
        for name, (mine, ref) in checks.items():
            d, nan_mismatch = _diff(mine, ref)
            rows.append((t, name, d, nan_mismatch))
    res = pd.DataFrame(rows, columns=["ticker", "indicator", "max_abs_diff", "nan_mismatch"])
    print(res.groupby("indicator").agg(max_abs_diff=("max_abs_diff", "max"), nan_mismatch=("nan_mismatch", "max")).round(6))
    print("\nNot checkable against TA-Lib (no equivalent): CMF (unit-tested vs a plain loop), %B, bandwidth, ATR%, "
          "realised vol, range position, MA distance, relative strength.")


if __name__ == "__main__":
    main()
