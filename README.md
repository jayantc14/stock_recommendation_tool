# Stock Recommendation Tool

A compound AI system for stock analysis: deterministic technical, fundamental,
and valuation engines plus a news/sentiment engine feed into a transparent
weighted scoring aggregator, and an LLM synthesis layer turns the structured
result into a natural-language recommendation with citations.

Core design principle: **rules/quant compute the numbers, the LLM only
reasons over them.** No engine lets an LLM invent a ratio, an indicator
value, or a sentiment score — every number handed to the LLM synthesis step
is pre-computed and auditable.

## Status

- [x] Technical Analysis Engine (`src/stock_advisor/technical/`) — SMA/EMA,
      RSI, MACD, Bollinger Bands, ATR, volume trend, support/resistance,
      rolled up into a weighted momentum score and a suggested entry range.
- [ ] Fundamental Analysis Engine (structured ratios + RAG over filings)
- [ ] Valuation Engine (DCF + relative valuation)
- [ ] News/Sentiment Engine
- [ ] Weighted Scoring Aggregator
- [ ] LLM Synthesis Agent

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Usage

```python
from stock_advisor.data.price_fetcher import fetch_price_history
from stock_advisor.technical.scoring import compute_technical_signal

df = fetch_price_history("RELIANCE.NS", period="1y")
signal = compute_technical_signal(df)

print(signal.label, signal.momentum_score)   # e.g. "Bullish" 47.5
print(signal.entry_range)                    # e.g. (1240.0, 1260.0)
print(signal.sub_scores)                     # per-indicator breakdown, for auditability
```

Note: `fetch_price_history` calls Yahoo Finance via `yfinance` and needs
outbound internet access to `finance.yahoo.com`. The technical engine itself
(`stock_advisor.technical`) is pure pandas/numpy and has no network
dependency, which is why it's unit-tested against synthetic price data.
