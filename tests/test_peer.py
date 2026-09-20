import pytest

from stock_advisor.fundamental.peer import compare_to_peers


def test_compare_to_peers_computes_average_and_delta():
    company = {"roe": 0.25, "debt_to_equity": 0.5}
    peers = [
        {"roe": 0.15, "debt_to_equity": 0.8},
        {"roe": 0.20, "debt_to_equity": 1.0},
    ]

    result = compare_to_peers(company, peers)

    assert result["roe"]["peer_average"] == pytest.approx(0.175)
    assert result["roe"]["delta_vs_peers"] == pytest.approx(0.075)
    assert result["debt_to_equity"]["peer_average"] == pytest.approx(0.9)


def test_compare_to_peers_handles_missing_company_value():
    result = compare_to_peers({"roe": None}, [{"roe": 0.2}])
    assert result["roe"] == {"company": None, "peer_average": None, "delta_vs_peers": None}


def test_compare_to_peers_handles_no_peer_data():
    result = compare_to_peers({"roe": 0.2}, [{"roe": None}, {}])
    assert result["roe"] == {"company": 0.2, "peer_average": None, "delta_vs_peers": None}
