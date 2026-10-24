from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_START = "2010-01-01"
DEFAULT_DIR = Path("data/prices")


def clean_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw yfinance frame: tz-naive daily index, OHLCV only, invalid bars dropped."""
    df = raw[OHLCV].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    valid = (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (df["High"] >= df["Low"])
    return df[valid]


def download_history(ticker: str, start: str = DEFAULT_START, retries: int = 3) -> pd.DataFrame:
    """Split/dividend-adjusted daily OHLCV from `start` to today, with simple retry."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
            if not raw.empty:
                return clean_history(raw)
            last_err = ValueError(f"empty result for {ticker}")
        except Exception as exc:  # network / rate-limit; retried below
            last_err = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {ticker}: {last_err}")


def quality_stats(ticker: str, df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> dict:
    """Per-ticker data-quality summary used to spot gaps, stale feeds and unadjusted splits."""
    as_of = as_of or pd.Timestamp.today().normalize()
    ret = df["Close"].pct_change()
    gaps = df.index.to_series().diff().dt.days
    return {
        "ticker": ticker,
        "rows": len(df),
        "first_date": df.index[0].date(),
        "last_date": df.index[-1].date(),
        "days_stale": (as_of - df.index[-1]).days,
        "max_gap_days": int(gaps.max()) if len(df) > 1 else 0,
        "zero_volume_days": int((df["Volume"] == 0).sum()),
        "abs_ret_gt_25pct": int((ret.abs() > 0.25).sum()),
        "max_abs_ret": round(float(ret.abs().max()), 3) if len(df) > 1 else 0.0,
    }


def suspect_moves(ticker: str, df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
    """Days with an |close-to-close return| above `threshold`.

    These are either genuine shocks (e.g. the Mar-2020 crash/rebound) or corporate-action
    artifacts yfinance did not back-adjust (splits, demergers); they need review before
    they are used to build forward-return labels.
    """
    ret = df["Close"].pct_change()
    hit = ret.abs() > threshold
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": df.index[hit],
            "ret": ret[hit].round(3).to_numpy(),
            "prev_close": df["Close"].shift(1)[hit].round(2).to_numpy(),
            "close": df["Close"][hit].round(2).to_numpy(),
            "volume": df["Volume"][hit].to_numpy(),
        }
    )


def build_dataset(
    tickers: list[str],
    out_dir: Path | str = DEFAULT_DIR,
    start: str = DEFAULT_START,
    workers: int = 4,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download and cache one parquet per ticker. Returns (quality table, {ticker: error})."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    errors: dict[str, str] = {}

    def work(t: str) -> None:
        try:
            df = download_history(t, start=start)
            df.to_parquet(out_dir / f"{t}.parquet")
            stats.append(quality_stats(t, df))
        except Exception as exc:
            errors[t] = str(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, tickers))
    table = pd.DataFrame(stats).sort_values("ticker").reset_index(drop=True) if stats else pd.DataFrame()
    return table, errors


def load_prices(ticker: str, out_dir: Path | str = DEFAULT_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / f"{ticker}.parquet")
