from __future__ import annotations

from pathlib import Path

import pandas as pd

NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
DEFAULT_UNIVERSE_PATH = Path("data/universe/nifty200.csv")
BENCHMARK_TICKER = "^NSEI"


def load_universe(path: Path | str = DEFAULT_UNIVERSE_PATH) -> pd.DataFrame:
    """Load an NSE index-constituent CSV and add a `ticker` column with the Yahoo ".NS" suffix.

    Note: this is the *current* constituent list, so a backtest on it carries
    survivorship bias (stocks delisted or dropped from the index are missing).
    """
    df = pd.read_csv(path)
    df["ticker"] = df["Symbol"].str.strip() + ".NS"
    return df[["ticker", "Symbol", "Company Name", "Industry"]].drop_duplicates("ticker").reset_index(drop=True)
