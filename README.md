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
- [x] Fundamental Analysis Engine — RAG half (`src/stock_advisor/rag/`) —
      chunk annual report text, retrieve the passages most relevant to a
      qualitative-risk query (TF-IDF, no model download needed), and ask a
      free/open-source LLM (via [Ollama](https://ollama.com), e.g.
      `llama3.2`) to summarize risk flags grounded only in those retrieved
      passages. The one place in the whole system an LLM reasons over prose
      instead of computing a number. `LLMClient` is a small interface, so
      swapping in a paid model later (e.g. Claude) is a one-file addition.
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

```python
from stock_advisor.rag.llm_client import OllamaClient
from stock_advisor.rag.risk_extraction import extract_risk_flags

with open("annual_report.txt") as f:
    document_text = f.read()

llm_client = OllamaClient(model="llama3.2")  # needs Ollama running locally
flags = extract_risk_flags(document_text, llm_client)

for flag in flags:
    print(flag.risk_type, flag.severity, flag.summary)
```

Or from the command line:

```bash
python scripts/check_signal.py RELIANCE.NS
python scripts/check_fundamentals.py RELIANCE.NS
python scripts/check_qualitative_risks.py annual_report.txt
```

Note: `check_signal.py`/`check_fundamentals.py` call Yahoo Finance via
`yfinance` and need outbound internet access to `finance.yahoo.com`.
`check_qualitative_risks.py` needs [Ollama](https://ollama.com) installed
and running locally with a model pulled (`ollama pull llama3.2`). The
engines themselves (`stock_advisor.technical`, `stock_advisor.fundamental`,
and the retrieval/chunking/scoring parts of `stock_advisor.rag`) are pure
Python with no network dependency, which is why they're unit-tested against
synthetic data and a mocked LLM client.
