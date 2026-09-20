from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


class FinancialsNotFoundError(ValueError):
    pass


@dataclass
class RawStatements:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame


def fetch_financial_statements(ticker: str) -> RawStatements:
    """Fetch annual income statement, balance sheet, and cash flow via yfinance.

    Each returned DataFrame is indexed by line item (e.g. "Total Revenue")
    with one column per fiscal period end date, exactly as yfinance provides
    it -- no cleaning or aliasing happens here, see `fundamental/metrics.py`
    for that.
    """
    handle = yf.Ticker(ticker)
    income_statement = handle.financials
    balance_sheet = handle.balance_sheet
    cash_flow = handle.cashflow

    if income_statement.empty or balance_sheet.empty:
        raise FinancialsNotFoundError(f"No financial statements returned for ticker '{ticker}'")

    return RawStatements(
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
    )
