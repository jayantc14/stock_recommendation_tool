import pandas as pd
import pytest

from stock_advisor.fundamental.scoring import compute_fundamental_signal

Y1 = pd.Timestamp("2022-03-31")
Y2 = pd.Timestamp("2023-03-31")
Y3 = pd.Timestamp("2024-03-31")


def _statements(income_rows: dict, balance_rows: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    income_statement = pd.DataFrame(income_rows, index=[Y1, Y2, Y3]).T
    balance_sheet = pd.DataFrame(balance_rows, index=[Y1, Y2, Y3]).T
    return income_statement, balance_sheet


def test_strong_growing_low_debt_company_scores_strong():
    income_statement, balance_sheet = _statements(
        income_rows={
            "Total Revenue": [1000.0, 1200.0, 1450.0],
            "Gross Profit": [400.0, 500.0, 620.0],
            "Operating Income": [200.0, 260.0, 330.0],
            "EBIT": [200.0, 260.0, 330.0],
            "Net Income": [140.0, 180.0, 230.0],
        },
        balance_rows={
            "Total Assets": [1000.0, 1150.0, 1300.0],
            "Current Liabilities": [200.0, 220.0, 240.0],
            "Stockholders Equity": [600.0, 700.0, 830.0],
            "Total Debt": [150.0, 140.0, 130.0],
        },
    )

    signal = compute_fundamental_signal(income_statement, balance_sheet)

    assert signal.label == "Strong"
    assert signal.fundamental_score > 20
    assert signal.latest_ratios["roe"] > 0.2
    assert signal.growth["revenue_growth_yoy"] == pytest.approx((1450 - 1200) / 1200, abs=1e-4)


def test_declining_indebted_company_scores_weak():
    income_statement, balance_sheet = _statements(
        income_rows={
            "Total Revenue": [1000.0, 900.0, 800.0],
            "Gross Profit": [300.0, 250.0, 200.0],
            "Operating Income": [50.0, 20.0, -10.0],
            "EBIT": [50.0, 20.0, -10.0],
            "Net Income": [20.0, -30.0, -60.0],
        },
        balance_rows={
            "Total Assets": [1000.0, 950.0, 900.0],
            "Current Liabilities": [300.0, 320.0, 340.0],
            "Stockholders Equity": [400.0, 350.0, 280.0],
            "Total Debt": [500.0, 550.0, 600.0],
        },
    )

    signal = compute_fundamental_signal(income_statement, balance_sheet)

    assert signal.label == "Weak"
    assert signal.fundamental_score < -20
    assert signal.latest_ratios["debt_to_equity"] > 2.0


def test_peer_comparison_is_included_when_peer_ratios_given():
    income_statement, balance_sheet = _statements(
        income_rows={
            "Total Revenue": [1000.0, 1200.0, 1450.0],
            "Gross Profit": [400.0, 500.0, 620.0],
            "Operating Income": [200.0, 260.0, 330.0],
            "EBIT": [200.0, 260.0, 330.0],
            "Net Income": [140.0, 180.0, 230.0],
        },
        balance_rows={
            "Total Assets": [1000.0, 1150.0, 1300.0],
            "Current Liabilities": [200.0, 220.0, 240.0],
            "Stockholders Equity": [600.0, 700.0, 830.0],
            "Total Debt": [150.0, 140.0, 130.0],
        },
    )
    peer_ratios = [{"roe": 0.15, "roce": 0.18, "debt_to_equity": 0.6,
                    "gross_margin": 0.35, "operating_margin": 0.15, "net_margin": 0.10}]

    signal = compute_fundamental_signal(income_statement, balance_sheet, peer_ratios=peer_ratios)

    assert signal.peer_comparison is not None
    assert signal.peer_comparison["roe"]["delta_vs_peers"] > 0
