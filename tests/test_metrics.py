import pandas as pd

from stock_advisor.fundamental.metrics import extract_all_periods, extract_metrics

Y1 = pd.Timestamp("2022-03-31")
Y2 = pd.Timestamp("2023-03-31")


def test_extract_metrics_reads_direct_fields():
    income_statement = pd.DataFrame(
        {Y1: [1000.0, 400.0, 200.0, 210.0, 140.0]},
        index=["Total Revenue", "Gross Profit", "Operating Income", "EBIT", "Net Income"],
    )
    balance_sheet = pd.DataFrame(
        {Y1: [1000.0, 300.0, 200.0, 600.0, 150.0]},
        index=[
            "Total Assets",
            "Current Assets",
            "Current Liabilities",
            "Stockholders Equity",
            "Total Debt",
        ],
    )

    metrics = extract_metrics(income_statement, balance_sheet, Y1)

    assert metrics.total_revenue == 1000.0
    assert metrics.ebit == 210.0
    assert metrics.total_debt == 150.0
    assert metrics.total_equity == 600.0


def test_extract_metrics_falls_back_to_operating_income_when_ebit_missing():
    income_statement = pd.DataFrame(
        {Y1: [1000.0, 400.0, 200.0, 140.0]},
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )
    balance_sheet = pd.DataFrame({Y1: [1000.0]}, index=["Total Assets"])

    metrics = extract_metrics(income_statement, balance_sheet, Y1)

    assert metrics.ebit == 200.0  # no "EBIT" row, so it falls back to Operating Income


def test_extract_metrics_derives_total_debt_from_long_and_current_debt():
    income_statement = pd.DataFrame({Y1: [1000.0]}, index=["Total Revenue"])
    balance_sheet = pd.DataFrame(
        {Y1: [300.0, 50.0]},
        index=["Long Term Debt", "Current Debt"],
    )

    metrics = extract_metrics(income_statement, balance_sheet, Y1)

    assert metrics.total_debt == 350.0


def test_extract_metrics_returns_none_for_fully_absent_field():
    income_statement = pd.DataFrame({Y1: [1000.0]}, index=["Total Revenue"])
    balance_sheet = pd.DataFrame({Y1: [1000.0]}, index=["Total Assets"])

    metrics = extract_metrics(income_statement, balance_sheet, Y1)

    assert metrics.net_income is None
    assert metrics.total_equity is None


def test_extract_all_periods_only_keeps_overlapping_columns_sorted_oldest_first():
    income_statement = pd.DataFrame(
        {Y2: [1200.0], Y1: [1000.0], pd.Timestamp("2021-03-31"): [900.0]},
        index=["Total Revenue"],
    )
    balance_sheet = pd.DataFrame(
        {Y2: [1100.0], Y1: [1000.0]},
        index=["Total Assets"],
    )

    periods = extract_all_periods(income_statement, balance_sheet)

    assert [p.period for p in periods] == [Y1, Y2]
