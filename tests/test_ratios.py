import pytest

from stock_advisor.fundamental.metrics import FinancialMetrics
from stock_advisor.fundamental.ratios import (
    cagr,
    debt_to_equity,
    gross_margin,
    growth_rate,
    net_margin,
    operating_margin,
    return_on_capital_employed,
    return_on_equity,
)


def _metrics(**overrides) -> FinancialMetrics:
    defaults = dict(
        period="2024",
        total_revenue=1000.0,
        gross_profit=400.0,
        operating_income=200.0,
        ebit=210.0,
        net_income=140.0,
        total_assets=1000.0,
        current_assets=300.0,
        current_liabilities=200.0,
        total_equity=700.0,
        total_debt=150.0,
    )
    defaults.update(overrides)
    return FinancialMetrics(**defaults)


def test_roe_uses_average_equity_when_prior_given():
    latest = _metrics(net_income=140.0, total_equity=700.0)
    prior = _metrics(total_equity=600.0)
    # average equity = (700 + 600) / 2 = 650
    assert return_on_equity(latest, prior) == pytest.approx(140.0 / 650.0)


def test_roe_uses_point_in_time_equity_without_prior():
    latest = _metrics(net_income=140.0, total_equity=700.0)
    assert return_on_equity(latest) == pytest.approx(140.0 / 700.0)


def test_roe_is_none_when_equity_missing():
    latest = _metrics(total_equity=None)
    assert return_on_equity(latest) is None


def test_roce_uses_ebit_over_capital_employed():
    latest = _metrics(ebit=210.0, total_assets=1000.0, current_liabilities=200.0)
    assert return_on_capital_employed(latest) == pytest.approx(210.0 / 800.0)


def test_roce_falls_back_to_operating_income_when_ebit_missing():
    latest = _metrics(ebit=None, operating_income=200.0, total_assets=1000.0, current_liabilities=200.0)
    assert return_on_capital_employed(latest) == pytest.approx(200.0 / 800.0)


def test_debt_to_equity_none_when_equity_is_zero():
    latest = _metrics(total_debt=150.0, total_equity=0.0)
    assert debt_to_equity(latest) is None


def test_margins():
    latest = _metrics(total_revenue=1000.0, gross_profit=400.0, operating_income=200.0, net_income=140.0)
    assert gross_margin(latest) == pytest.approx(0.4)
    assert operating_margin(latest) == pytest.approx(0.2)
    assert net_margin(latest) == pytest.approx(0.14)


def test_growth_rate_positive_and_negative():
    assert growth_rate(120.0, 100.0) == pytest.approx(0.2)
    assert growth_rate(80.0, 100.0) == pytest.approx(-0.2)


def test_growth_rate_none_when_prior_zero():
    assert growth_rate(100.0, 0.0) is None


def test_cagr_over_multiple_years():
    # 100 -> 144 over 2 years is a 20% CAGR
    assert cagr(100.0, 144.0, 2) == pytest.approx(0.2, abs=1e-6)


def test_cagr_none_for_non_positive_inputs():
    assert cagr(0.0, 144.0, 2) is None
    assert cagr(100.0, -50.0, 2) is None
