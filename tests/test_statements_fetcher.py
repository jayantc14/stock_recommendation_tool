import pandas as pd
import pytest

from stock_advisor.data.statements_fetcher import FinancialsNotFoundError, fetch_financial_statements


class _FakeYFTicker:
    def __init__(self, income_statement, balance_sheet, cash_flow):
        self.financials = income_statement
        self.balance_sheet = balance_sheet
        self.cashflow = cash_flow


def test_fetch_financial_statements_returns_raw_frames(monkeypatch):
    income_statement = pd.DataFrame({pd.Timestamp("2024-03-31"): [1000.0]}, index=["Total Revenue"])
    balance_sheet = pd.DataFrame({pd.Timestamp("2024-03-31"): [1000.0]}, index=["Total Assets"])
    cash_flow = pd.DataFrame({pd.Timestamp("2024-03-31"): [500.0]}, index=["Operating Cash Flow"])

    monkeypatch.setattr(
        "stock_advisor.data.statements_fetcher.yf.Ticker",
        lambda ticker: _FakeYFTicker(income_statement, balance_sheet, cash_flow),
    )

    result = fetch_financial_statements("RELIANCE.NS")

    assert result.income_statement.equals(income_statement)
    assert result.balance_sheet.equals(balance_sheet)
    assert result.cash_flow.equals(cash_flow)


def test_fetch_financial_statements_raises_when_empty(monkeypatch):
    monkeypatch.setattr(
        "stock_advisor.data.statements_fetcher.yf.Ticker",
        lambda ticker: _FakeYFTicker(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )

    with pytest.raises(FinancialsNotFoundError):
        fetch_financial_statements("NOT_A_TICKER")
