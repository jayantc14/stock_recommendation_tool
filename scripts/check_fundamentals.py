"""Manual sanity-check CLI: fetch a real ticker's statements and print its
structured fundamental signal.

Usage:
    python scripts/check_fundamentals.py RELIANCE.NS
"""

from __future__ import annotations

import argparse
import json

from stock_advisor.data.statements_fetcher import fetch_financial_statements
from stock_advisor.fundamental.scoring import compute_fundamental_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="e.g. RELIANCE.NS, TCS.NS, AAPL")
    args = parser.parse_args()

    statements = fetch_financial_statements(args.ticker)
    signal = compute_fundamental_signal(statements.income_statement, statements.balance_sheet)

    print(f"\n{args.ticker}")
    print(f"  Fundamental score : {signal.fundamental_score}  ({signal.label})")
    print("  Latest ratios     :")
    print(json.dumps(signal.latest_ratios, indent=4))
    print("  Growth            :")
    print(json.dumps(signal.growth, indent=4))
    print("  Sub-scores        :")
    print(json.dumps(signal.sub_scores, indent=4))
    print()


if __name__ == "__main__":
    main()
