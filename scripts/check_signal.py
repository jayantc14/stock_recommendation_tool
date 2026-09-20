"""Manual sanity-check CLI: fetch a real ticker and print its technical signal.

Usage:
    python scripts/check_signal.py RELIANCE.NS
    python scripts/check_signal.py AAPL --period 6mo
"""

from __future__ import annotations

import argparse
import json

from stock_advisor.data.price_fetcher import fetch_price_history
from stock_advisor.technical.scoring import compute_technical_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="e.g. RELIANCE.NS, TCS.NS, AAPL")
    parser.add_argument("--period", default="1y", help="yfinance period, default 1y")
    args = parser.parse_args()

    df = fetch_price_history(args.ticker, period=args.period)
    signal = compute_technical_signal(df)

    print(f"\n{args.ticker}  (as of {df.index[-1].date()}, {len(df)} bars)")
    print(f"  Current price     : {signal.current_price}")
    print(f"  Momentum score    : {signal.momentum_score}  ({signal.label})")
    print(f"  Suggested entry   : {signal.entry_range}")
    print("  Sub-scores        :")
    print(json.dumps(signal.sub_scores, indent=4))
    print("  Raw indicators    :")
    print(json.dumps(signal.raw_indicators, indent=4))
    print()


if __name__ == "__main__":
    main()
