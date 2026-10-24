"""Download and cache daily OHLCV for the Nifty 200 universe plus the Nifty 50 benchmark.

Usage:
    python scripts/build_price_dataset.py
    python scripts/build_price_dataset.py --start 2012-01-01 --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_advisor.data.history_store import DEFAULT_START, build_dataset, load_prices, suspect_moves
from stock_advisor.data.universe import BENCHMARK_TICKER, load_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--limit", type=int, default=None, help="only first N tickers (smoke test)")
    parser.add_argument("--out", default="data/prices")
    args = parser.parse_args()

    tickers = load_universe()["ticker"].tolist()
    if args.limit:
        tickers = tickers[: args.limit]
    tickers = [BENCHMARK_TICKER] + tickers

    table, errors = build_dataset(tickers, out_dir=args.out, start=args.start)

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "price_quality.csv", index=False)

    suspects = pd.concat(
        [suspect_moves(t, load_prices(t, args.out)) for t in table["ticker"]], ignore_index=True
    ).sort_values(["ticker", "date"])
    suspects.to_csv(report_dir / "suspect_moves.csv", index=False)

    print(f"\nDownloaded {len(table)}/{len(tickers)} tickers -> {args.out}")
    if errors:
        print(f"Failed ({len(errors)}): {errors}")
    print(f"Date range     : {table['first_date'].min()} .. {table['last_date'].max()}")
    print(f"Rows per ticker: min {table['rows'].min()}, median {int(table['rows'].median())}, max {table['rows'].max()}")
    print(f"Tickers with < 3y history     : {(table['rows'] < 750).sum()}")
    print(f"Tickers stale (> 7 days)      : {(table['days_stale'] > 7).sum()}")
    print(f"Tickers with |ret| > 25% days : {(table['abs_ret_gt_25pct'] > 0).sum()}  (possible unadjusted splits)")
    print(f"Suspect-move days listed      : {len(suspects)}")
    print("Reports: data/reports/price_quality.csv, data/reports/suspect_moves.csv\n")


if __name__ == "__main__":
    main()
