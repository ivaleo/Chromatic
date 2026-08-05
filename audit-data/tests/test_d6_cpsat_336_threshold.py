import numpy as np

from chromatic_research.campaigns.d6_cpsat_336_threshold import (
    best_campaign_hint,
    canonical_free_rows_mod4,
    integer_conflict_weights,
    normalized_prime_row,
    rref_mod2,
    solve_pattern,
)
from chromatic_research.campaigns.threshold_multiblock_search import row_module_key


def test_rref_mod2_and_canonical_mod4_preserve_row_module() -> None:
    rows = np.asarray(
        [[1, 2, 3, 0], [2, 1, 1, 2]], dtype=np.int64
    )
    reduced, pivots = rref_mod2(rows)
    assert pivots == (0, 1)
    assert np.array_equal(reduced[:, pivots], np.eye(2, dtype=np.int64))
    canonical, canonical_pivots = canonical_free_rows_mod4(rows)
    assert canonical_pivots == pivots
    assert np.array_equal(
        canonical[:, pivots], np.eye(2, dtype=np.int64)
    )
    assert row_module_key(rows, 4) == row_module_key(canonical, 4)


def test_prime_row_normalization_is_projectively_equal() -> None:
    normalized, pivot = normalized_prime_row([0, 3, 6, 1], 7)
    assert pivot == 1
    assert normalized.tolist() == [0, 1, 2, 5]


def test_soft_cpsat_pattern_can_reach_zero_conflicts() -> None:
    status, rows, metadata = solve_pattern(
        np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        (0, 1),
        hints={},
        time_limit=2.0,
        workers=1,
        seed=11,
        minimize_conflicts=True,
    )
    assert status == "OPTIMAL"
    assert rows is not None
    assert metadata["objective_value"] == 0.0


def test_integer_conflict_weights_penalize_deep_violations() -> None:
    weights = integer_conflict_weights(
        [0.0, 0.5, 0.99, 0.999999],
        threshold=1.0,
        power=4.0,
        scale=1_000_000,
    )
    assert weights.tolist() == [1_000_000, 62_500, 1, 1]


def test_weighted_cpsat_objective_still_reaches_zero() -> None:
    status, rows, metadata = solve_pattern(
        np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        (0, 1),
        hints={},
        time_limit=2.0,
        workers=1,
        seed=12,
        minimize_conflicts=True,
        violation_weights=[17, 3],
    )
    assert status == "OPTIMAL"
    assert rows is not None
    assert metadata["objective_value"] == 0.0
    assert metadata["weighted_conflicts"] is True


def test_best_campaign_hint_prefers_fewer_conflicts() -> None:
    payload = {
        "results": [
            {
                "moduli": [7, 4, 4, 3],
                "rows": [[1, 0]] * 4,
                "killed": 5,
                "minimum_conflict_ratio": 0.9,
            },
            {
                "moduli": [7, 4, 4, 3],
                "rows": [[0, 1]] * 4,
                "killed": 2,
                "minimum_conflict_ratio": 0.1,
            },
        ]
    }
    hints, record = best_campaign_hint(payload, 2)
    assert record is not None
    assert record["killed"] == 2
    assert [row.tolist() for row in hints[7]] == [[0, 1]]
