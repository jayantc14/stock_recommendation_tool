from __future__ import annotations

from stock_advisor.fundamental.metrics import FinancialMetrics


def return_on_equity(
    metrics: FinancialMetrics,
    prior: FinancialMetrics | None = None,
) -> float | None:
    if metrics.net_income is None or metrics.total_equity is None:
        return None
    equity = metrics.total_equity
    if prior is not None and prior.total_equity is not None:
        equity = (metrics.total_equity + prior.total_equity) / 2
    if equity == 0:
        return None
    return metrics.net_income / equity


def return_on_capital_employed(metrics: FinancialMetrics) -> float | None:
    ebit = metrics.ebit if metrics.ebit is not None else metrics.operating_income
    if ebit is None or metrics.total_assets is None or metrics.current_liabilities is None:
        return None
    capital_employed = metrics.total_assets - metrics.current_liabilities
    if capital_employed == 0:
        return None
    return ebit / capital_employed


def debt_to_equity(metrics: FinancialMetrics) -> float | None:
    if metrics.total_debt is None or not metrics.total_equity:
        return None
    return metrics.total_debt / metrics.total_equity


def gross_margin(metrics: FinancialMetrics) -> float | None:
    if metrics.gross_profit is None or not metrics.total_revenue:
        return None
    return metrics.gross_profit / metrics.total_revenue


def operating_margin(metrics: FinancialMetrics) -> float | None:
    if metrics.operating_income is None or not metrics.total_revenue:
        return None
    return metrics.operating_income / metrics.total_revenue


def net_margin(metrics: FinancialMetrics) -> float | None:
    if metrics.net_income is None or not metrics.total_revenue:
        return None
    return metrics.net_income / metrics.total_revenue


def growth_rate(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior)


def cagr(start_value: float | None, end_value: float | None, years: int) -> float | None:
    if start_value is None or end_value is None or start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1
