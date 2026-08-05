import numpy as np

from d6_highs_psd_outer import (
    covering_cut_coefficients,
    pack_gram,
    parse_indices,
    solve_psd_outer,
    symmetric_coefficients,
    unpack_gram,
    wall_branches,
)
from d6_sdp_hybrid import covering_lmi_numeric


def test_symmetric_coordinate_round_trip_and_linear_form() -> None:
    gram = np.asarray([[2.0, -0.3], [-0.3, 1.5]])
    matrix = np.asarray([[1.0, 0.4], [0.4, -2.0]])
    packed = pack_gram(gram)
    assert np.allclose(unpack_gram(packed, 2), gram)
    assert np.isclose(
        symmetric_coefficients(matrix) @ packed,
        np.sum(matrix * gram),
        atol=1e-12,
    )


def test_covering_eigenvector_cut_matches_full_lmi() -> None:
    rows = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    gram = np.asarray([[1.7, 0.2], [0.2, 1.2]])
    rho = 0.8
    vector = np.asarray([0.3, -0.7, 1.1])
    matrix, rho_coefficient = covering_cut_coefficients(rows, vector)
    direct = float(
        vector @ covering_lmi_numeric(gram, rows, rho) @ vector
    )
    linear = float(np.sum(matrix * gram) + rho_coefficient * rho)
    assert np.isclose(linear, direct, atol=1e-12)


def test_highs_outer_approximation_solves_one_dimensional_sdp() -> None:
    result = solve_psd_outer(
        1,
        [np.asarray([[1]], dtype=np.int64)],
        [np.asarray([[1.0]])],
        warm_gram=np.asarray([[1.0]]),
        warm_rho=0.25,
        positive_floor=1e-8,
        max_rounds=30,
        cuts_per_round=8,
        violation_tolerance=1e-9,
        gram_bound_factor=4.0,
    )
    assert result["success"]
    assert result["converged"]
    assert np.isclose(result["gram"][0, 0], 1.0, atol=2e-7)
    assert np.isclose(result["rho"], 0.25, atol=2e-7)


def test_wall_index_parser_and_chamber_generation() -> None:
    assert parse_indices("0,2-4", 6) == [0, 2, 3, 4]
    records = [{"wall_matrix": [[index + 1]]} for index in range(3)]
    branches = wall_branches(
        records,
        [0, 1, 2],
        intersection_walls=3,
        max_combination_width=2,
    )
    sign_patterns = {tuple(branch["signed_walls"]) for branch in branches}
    assert ((0, -1),) in sign_patterns
    assert ((0, -1), (1, 1), (2, 1)) in sign_patterns
    assert ((0, -1), (1, -1), (2, 1)) in sign_patterns
