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
- [x] Fundamental Analysis Engine — structured half
      (`src/stock_advisor/fundamental/`) — ROE, ROCE, D/E, margins,
      revenue/net-income growth and CAGR extracted from balance sheet/P&L,
      rolled up into a weighted fundamental score, with optional peer
      comparison.
- [ ] Fundamental Analysis Engine — RAG half (embed + retrieve annual
      report text for qualitative risk flags; the one place an LLM reasons
      over prose rather than numbers)
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

```python
from stock_advisor.data.statements_fetcher import fetch_financial_statements
from stock_advisor.fundamental.scoring import compute_fundamental_signal

statements = fetch_financial_statements("RELIANCE.NS")
signal = compute_fundamental_signal(statements.income_statement, statements.balance_sheet)

print(signal.label, signal.fundamental_score)  # e.g. "Strong" 62.5
print(signal.latest_ratios)                    # ROE, ROCE, D/E, margins
print(signal.growth)                           # revenue/net income growth, CAGR
```

Or from the command line:

```bash
python scripts/check_signal.py RELIANCE.NS
python scripts/check_fundamentals.py RELIANCE.NS
```

Note: both fetchers call Yahoo Finance via `yfinance` and need outbound
internet access to `finance.yahoo.com`. The engines themselves
(`stock_advisor.technical`, `stock_advisor.fundamental`) are pure
pandas/numpy with no network dependency, which is why they're unit-tested
against synthetic data.
