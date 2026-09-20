from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stock_advisor.fundamental import ratios as r
from stock_advisor.fundamental.metrics import extract_all_periods
from stock_advisor.fundamental.peer import compare_to_peers

_WEIGHTS = {
    "profitability": 0.35,
    "leverage": 0.20,
    "margins": 0.20,
    "growth": 0.25,
}


@dataclass
class FundamentalSignal:
    fundamental_score: float
    label: str
    latest_ratios: dict[str, float | None]
    growth: dict[str, float | None]
    sub_scores: dict[str, float] = field(default_factory=dict)
    peer_comparison: dict[str, dict] | None = None


def _profitability_score(roe: float | None, roce: float | None) -> float:
    values = [v for v in (roe, roce) if v is not None]
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    if avg >= 0.20:
        return 1.0
    if avg >= 0.12:
        return 0.5
    if avg >= 0.0:
        return 0.0
    return -1.0


def _leverage_score(debt_to_equity: float | None) -> float:
    if debt_to_equity is None:
        return 0.0
    if debt_to_equity <= 0.3:
        return 1.0
    if debt_to_equity <= 1.0:
        return 0.3
    if debt_to_equity <= 2.0:
        return -0.3
    return -1.0


def _margin_score(net_margin_now: float | None, net_margin_prior: float | None) -> float:
    if net_margin_now is None:
        return 0.0
    level_score = 0.5 if net_margin_now >= 0.10 else (0.0 if net_margin_now >= 0 else -0.5)
    trend_score = 0.0
    if net_margin_prior is not None:
        if net_margin_now > net_margin_prior:
            trend_score = 0.3
        elif net_margin_now < net_margin_prior:
            trend_score = -0.3
    return max(min(level_score + trend_score, 1.0), -1.0)


def _growth_score(revenue_growth: float | None, net_income_growth: float | None) -> float:
    values = [v for v in (revenue_growth, net_income_growth) if v is not None]
    if not values:
        return 0.0
    avg = sum(values) / len(values)
    if avg >= 0.15:
        return 1.0
    if avg >= 0.05:
        return 0.5
    if avg >= 0.0:
        return 0.0
    return -1.0


def compute_fundamental_signal(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    peer_ratios: list[dict[str, float | None]] | None = None,
) -> FundamentalSignal:
    """Combine deterministic ratios from the latest available fiscal period
    into a weighted fundamental score. Needs at least one fiscal period;
    growth/trend figures need at least two.
    """
    periods = extract_all_periods(income_statement, balance_sheet)
    if not periods:
        raise ValueError("No overlapping fiscal periods between income statement and balance sheet")

    latest = periods[-1]
    prior = periods[-2] if len(periods) >= 2 else None

    roe = r.return_on_equity(latest, prior)
    roce = r.return_on_capital_employed(latest)
    debt_to_equity = r.debt_to_equity(latest)
    gross_margin = r.gross_margin(latest)
    operating_margin = r.operating_margin(latest)
    net_margin = r.net_margin(latest)
    net_margin_prior = r.net_margin(prior) if prior is not None else None

    revenue_growth_yoy = r.growth_rate(latest.total_revenue, prior.total_revenue) if prior else None
    net_income_growth_yoy = r.growth_rate(latest.net_income, prior.net_income) if prior else None
    revenue_cagr = (
        r.cagr(periods[0].total_revenue, latest.total_revenue, len(periods) - 1)
        if len(periods) > 1
        else None
    )

    sub_scores = {
        "profitability": _profitability_score(roe, roce),
        "leverage": _leverage_score(debt_to_equity),
        "margins": _margin_score(net_margin, net_margin_prior),
        "growth": _growth_score(revenue_growth_yoy, net_income_growth_yoy),
    }
    weighted = sum(_WEIGHTS[k] * v for k, v in sub_scores.items())
    fundamental_score = round(weighted * 100, 2)

    if fundamental_score >= 20:
        label = "Strong"
    elif fundamental_score <= -20:
        label = "Weak"
    else:
        label = "Average"

    latest_ratios = {
        "roe": round(roe, 4) if roe is not None else None,
        "roce": round(roce, 4) if roce is not None else None,
        "debt_to_equity": round(debt_to_equity, 4) if debt_to_equity is not None else None,
        "gross_margin": round(gross_margin, 4) if gross_margin is not None else None,
        "operating_margin": round(operating_margin, 4) if operating_margin is not None else None,
        "net_margin": round(net_margin, 4) if net_margin is not None else None,
    }

    growth = {
        "revenue_growth_yoy": round(revenue_growth_yoy, 4) if revenue_growth_yoy is not None else None,
        "net_income_growth_yoy": round(net_income_growth_yoy, 4) if net_income_growth_yoy is not None else None,
        "revenue_cagr": round(revenue_cagr, 4) if revenue_cagr is not None else None,
    }

    peer_comparison = compare_to_peers(latest_ratios, peer_ratios) if peer_ratios else None

    return FundamentalSignal(
        fundamental_score=fundamental_score,
        label=label,
        latest_ratios=latest_ratios,
        growth=growth,
        sub_scores={k: round(v, 3) for k, v in sub_scores.items()},
        peer_comparison=peer_comparison,
    )
