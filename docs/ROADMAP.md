# Technical-model roadmap

Goal: replace the hand-picked weights and thresholds in `technical/scoring.py` with weights
learned from historical data, then use them at inference time. Work is done one step at a time;
each step ends with something to validate before the next begins.

## Decisions so far

- Universe: Nifty 200 (current constituents, from NSE), benchmark `^NSEI`. ~16 years of daily data from 2010.
- Targets: forward return over 20 trading days first, then 5 and 60. Raw and excess over the Nifty 50.
- Features: each indicator as both a continuous value and a categorical rule-based bucket (standard
  thresholds, e.g. RSI <30 / 30-70 / >70). We keep whichever carries more signal on history.
- Add many indicators; the model and feature-importance analysis decide which survive.
- Models: Ridge/ElasticNet/logistic (interpretable weights) vs LightGBM (non-linear), compared with
  purged walk-forward CV. Last ~2 years are a locked holdout that nothing is tuned on.
- Inference stays auditable: per-feature contribution breakdown remains part of the output.

## Known problems in the original `technical/scoring.py` (to be fixed)

1. Overlapping signals (trend, RSI, MACD ~80% of weight all measure momentum) and conflicting ones (Bollinger is contrarian).
2. Hard-threshold cliffs (RSI 69.9 -> +1.0, 70.0 -> -0.5).
3. Sub-scores on different scales (Bollinger +/-0.3, volume +/-0.5, others +/-1), so stated weights are not real influence.
4. Weights (35/20/25/10/10) and label cut-offs (+/-20) are undocumented, not derived from data.
5. Fewer than 200 bars silently zeroes the trend and volume scores.
6. Entry range = 60-day low to low + 0.5 ATR: unrelated to current price, produced even for Bearish, no stop/target.
7. Volume ratio divides by zero on illiquid names.

## Steps

| # | Step | Status |
|---|---|---|
| 0 | Data access (Yahoo reachable) | done |
| 1 | Price data pipeline (Nifty 200 + benchmark, parquet cache, quality reports) | done, validate |
| 2 | Expanded indicator library (ADX/DI, Stochastic, Williams %R, CCI, MFI, OBV, CMF, multi-horizon ROC, MA distances/crossovers, 52w high/low position, %B + bandwidth, ATR%, realised vol, Donchian, Aroon, relative strength vs index) plus history-length guard and div-by-zero fix | next |
| 3 | Feature builder: continuous + categorical per indicator, scale-free, no look-ahead, leakage test | |
| 4 | Labels and dataset: forward returns (5/20/60, raw + excess + cross-sectional rank), locked holdout, corporate-action exclusion | |
| 5 | Univariate analysis: IC, quantile-bucket returns, continuous vs categorical | |
| 6 | Modelling with purged walk-forward CV (Ridge/ElasticNet/logistic vs LightGBM) | |
| 7 | Feature selection and importance (prune correlated, permutation importance / SHAP) | |
| 8 | Replace `scoring.py`: learned weights in a versioned artifact, calibrated Bullish/Bearish cut-offs | |
| 9 | One-shot holdout test and backtest with transaction costs | |
| 10 | Entry range redesign: swing support/resistance, stop-loss, target, risk/reward | |

## Step 1 findings

- 200 stocks + benchmark downloaded, 2010-01-04 to 2026-10-23; median 4,149 rows; no gaps >10 days; none stale.
- 15 tickers have <3 years of history (recent IPOs: Swiggy, Hyundai, Groww, Meesho, Lenskart, ...). Kept.
- 61 days across 35 tickers have an absolute daily move >25% (`data/reports/suspect_moves.csv`).
  Many are real (Yes Bank Mar-2020, IndusInd Mar-2020, PNB Oct-2017). At least these are data artifacts
  that would create fake -65% to -95% "returns": Nestle 2010-01-08 (split), Motilal Oswal 2024-01-01 (split),
  Patanjali 2020-01-27, Vedanta 2026-04-30 (demerger), Tata Motors/TMPV 2025-10-14 (demerger).
- Proposed handling (awaiting confirmation): in step 4, drop training rows whose forward window overlaps a
  move >40%; review the 25-40% moves by eye.
- Survivorship bias: the universe is today's Nifty 200, so delisted/dropped stocks are missing and backtests
  will look slightly optimistic. State this in the final report.

## Expectations

Technical indicators usually have weak predictive power; a rank IC of 0.02-0.05 would be a decent result.
The holdout test in step 9 decides whether the score has any edge.

## Working setup

Source of truth is GitHub. Heavy logic lives in `src/stock_advisor/` with tests; notebooks in `notebooks/` stay
thin and import the package. Price data is not committed: rebuild with `python scripts/build_price_dataset.py`.
