from __future__ import annotations


def compare_to_peers(
    company_ratios: dict[str, float | None],
    peer_ratios: list[dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    """Compare each of the company's ratios to the simple average across peers.

    `peer_ratios` is a list of per-peer ratio dicts with the same keys as
    `company_ratios` (typically produced by running the same ratio functions
    on each peer's own statements). A ratio missing on the company, or on
    every peer, comes back with `peer_average`/`delta_vs_peers` as None.
    """
    comparison: dict[str, dict[str, float | None]] = {}
    for key, company_value in company_ratios.items():
        peer_values = [p[key] for p in peer_ratios if p.get(key) is not None]
        if company_value is None or not peer_values:
            comparison[key] = {
                "company": company_value,
                "peer_average": None,
                "delta_vs_peers": None,
            }
            continue
        peer_average = sum(peer_values) / len(peer_values)
        comparison[key] = {
            "company": company_value,
            "peer_average": round(peer_average, 4),
            "delta_vs_peers": round(company_value - peer_average, 4),
        }
    return comparison
