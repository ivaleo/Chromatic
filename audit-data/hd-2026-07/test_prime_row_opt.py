import numpy as np

from prime_row_opt import (
    _best_coordinate_values,
    _cached_forbidden_with_weights,
    _canonical_prime_row,
    _update_pareto_archive,
)


def test_best_coordinate_values_keeps_float_costs_for_fixed_constraints():
    forbidden = np.array([[0, 1], [0, 2]], dtype=np.int64)
    weights = np.array([0.25, 0.5], dtype=np.float64)
    row = np.array([1, 0], dtype=np.int64)
    dots = np.array([0, 0], dtype=np.int64)

    candidates, counts, costs = _best_coordinate_values(
        forbidden,
        weights,
        row,
        dots,
        coordinate=0,
        prime=5,
        objective="lexicographic",
    )

    np.testing.assert_array_equal(candidates, np.arange(5))
    np.testing.assert_array_equal(counts, np.full(5, 2))
    np.testing.assert_allclose(costs, np.full(5, 0.75))
    assert costs.dtype == np.float64


def test_forbidden_cache_recomputes_weights_for_new_power(tmp_path):
    basis = np.eye(2)
    diameter = 1.25
    forbidden = np.asarray([[1, 0], [1, 1]], dtype=np.int64)
    ratios = np.asarray([0.5, 0.8], dtype=np.float64)
    cache = tmp_path / "forbidden.npz"
    np.savez_compressed(
        cache,
        basis=basis,
        diameter=np.asarray(diameter),
        forbidden=forbidden,
        ratios=ratios,
    )

    loaded_forbidden, loaded_ratios, weights = (
        _cached_forbidden_with_weights(
            basis,
            diameter,
            cache,
            weight_power=4.0,
        )
    )

    np.testing.assert_array_equal(loaded_forbidden, forbidden)
    np.testing.assert_allclose(loaded_ratios, ratios)
    np.testing.assert_allclose(weights, np.asarray([0.5**4, 0.2**4]))


def test_pareto_archive_keeps_distinct_count_depth_tradeoffs():
    archive = []
    assert _update_pareto_archive(
        archive,
        np.array([2, 4, 6], dtype=np.int64),
        7,
        (14, 1e-12),
        0.987,
        8,
    )
    # Projective scaling of the same kernel is deduplicated.
    assert not _update_pareto_archive(
        archive,
        np.array([4, 1, 5], dtype=np.int64),
        7,
        (14, 1e-12),
        0.987,
        8,
    )
    assert _update_pareto_archive(
        archive,
        np.array([1, 0, 3], dtype=np.int64),
        7,
        (1, 0.2),
        0.45,
        8,
    )
    assert len(archive) == 2
    # Worse in all objectives than the first entry, hence rejected.
    assert not _update_pareto_archive(
        archive,
        np.array([1, 2, 2], dtype=np.int64),
        7,
        (15, 2e-12),
        0.98,
        8,
    )
    np.testing.assert_array_equal(
        _canonical_prime_row(np.array([2, 4, 6]), 7),
        np.array([1, 2, 3]),
    )
