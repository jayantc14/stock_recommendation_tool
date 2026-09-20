from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_ALIASES: dict[str, list[str]] = {
    "total_revenue": ["Total Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "ebit": ["EBIT", "Operating Income"],
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"],
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
}


@dataclass
class FinancialMetrics:
    period: object
    total_revenue: float | None
    gross_profit: float | None
    operating_income: float | None
    ebit: float | None
    net_income: float | None
    total_assets: float | None
    current_assets: float | None
    current_liabilities: float | None
    total_equity: float | None
    total_debt: float | None


def _lookup(statement: pd.DataFrame, column, aliases: list[str]) -> float | None:
    for alias in aliases:
        if alias in statement.index:
            value = statement.loc[alias, column]
            if pd.notna(value):
                return float(value)
    return None


def _total_debt(balance_sheet: pd.DataFrame, column) -> float | None:
    direct = _lookup(balance_sheet, column, ["Total Debt"])
    if direct is not None:
        return direct
    long_term = _lookup(balance_sheet, column, ["Long Term Debt"])
    current = _lookup(
        balance_sheet, column, ["Current Debt", "Current Debt And Capital Lease Obligation"]
    )
    if long_term is None and current is None:
        return None
    return (long_term or 0.0) + (current or 0.0)


def extract_metrics(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    period,
) -> FinancialMetrics:
    return FinancialMetrics(
        period=period,
        total_revenue=_lookup(income_statement, period, _ALIASES["total_revenue"]),
        gross_profit=_lookup(income_statement, period, _ALIASES["gross_profit"]),
        operating_income=_lookup(income_statement, period, _ALIASES["operating_income"]),
        ebit=_lookup(income_statement, period, _ALIASES["ebit"]),
        net_income=_lookup(income_statement, period, _ALIASES["net_income"]),
        total_assets=_lookup(balance_sheet, period, _ALIASES["total_assets"]),
        current_assets=_lookup(balance_sheet, period, _ALIASES["current_assets"]),
        current_liabilities=_lookup(balance_sheet, period, _ALIASES["current_liabilities"]),
        total_equity=_lookup(balance_sheet, period, _ALIASES["total_equity"]),
        total_debt=_total_debt(balance_sheet, period),
    )


def extract_all_periods(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
) -> list[FinancialMetrics]:
    """Extract metrics for every fiscal period present in both statements,
    oldest first (so callers can index [-1] for latest, [-2] for prior)."""
    periods = sorted(set(income_statement.columns) & set(balance_sheet.columns))
    return [extract_metrics(income_statement, balance_sheet, period) for period in periods]
