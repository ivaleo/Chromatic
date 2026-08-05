import numpy as np

from chromatic_research.campaigns.prime_threshold_search import maximize_discrete_threshold


def test_maximize_discrete_threshold_uses_monotone_bracket():
    levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    calls = []

    def decide(threshold):
        calls.append(threshold)
        if threshold <= 0.3:
            return np.asarray([1, 2]), {
                "threshold": threshold,
                "status": "FEASIBLE",
            }, False
        return None, {
            "threshold": threshold,
            "status": "INFEASIBLE",
        }, True

    threshold, row, records = maximize_discrete_threshold(
        levels,
        {"threshold": 1.0, "status": "INFEASIBLE"},
        decide,
    )

    assert threshold == 0.3
    assert np.array_equal(row, [1, 2])
    assert calls[0] == 0.1
    assert all(record["status"] in {"FEASIBLE", "INFEASIBLE"} for record in records)


def test_maximize_discrete_threshold_deduplicates_levels():
    def decide(threshold):
        return np.asarray([1]), {"threshold": threshold, "status": "FEASIBLE"}, False

    threshold, _, _ = maximize_discrete_threshold(
        [0.25, 0.25, 0.5],
        {"threshold": 1.0, "status": "INFEASIBLE"},
        decide,
    )

    assert threshold == 0.5
