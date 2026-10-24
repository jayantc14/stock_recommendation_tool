import pandas as pd

from stock_advisor.data.history_store import clean_history, quality_stats


def _raw():
    idx = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"], tz="Asia/Kolkata"
    )
    return pd.DataFrame(
        {
            "Open": [10, 11, 12, 0, 13],
            "High": [11, 12, 13, 5, 12],
            "Low": [9, 10, 11, 0, 14],
            "Close": [10, 11, 12, 3, 13],
            "Volume": [100, 100, 100, 100, 100],
            "Dividends": [0] * 5,
        },
        index=idx,
    )


def test_clean_history_dedupes_and_drops_invalid_bars():
    out = clean_history(_raw())
    assert out.index.tz is None
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.is_unique
    # zero-price bar and High<Low bar are dropped; duplicate date keeps the last row
    assert list(out["Close"]) == [10, 12]


def test_quality_stats_flags_big_moves_and_staleness():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"])
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [100.0, 50.0, 51.0], "Volume": [1, 0, 1]}, index=idx
    )
    s = quality_stats("X.NS", df, as_of=pd.Timestamp("2024-01-20"))
    assert s["rows"] == 3 and s["max_gap_days"] == 8
    assert s["abs_ret_gt_25pct"] == 1 and s["zero_volume_days"] == 1
    assert s["days_stale"] == 10


def test_suspect_moves_lists_only_large_jumps():
    from stock_advisor.data.history_store import suspect_moves

    idx = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    df = pd.DataFrame({"Close": [100.0, 24.0, 25.0], "Volume": [1, 2, 3]}, index=idx)
    out = suspect_moves("X.NS", df)
    assert len(out) == 1 and out.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    assert out.iloc[0]["ret"] == -0.76
