import numpy as np

from chromatic_research.campaigns.d6_ltype_wall_cross import (
    circuit_matrix,
    oriented_wall_key,
    wall_slack_from_gram,
)


def direct_circumcenter_slack(
    gram: np.ndarray, active: np.ndarray, candidate: np.ndarray
) -> float:
    system = active @ gram
    right = 0.5 * np.einsum(
        "ij,ij->i", active @ gram, active
    )
    center = np.linalg.solve(system, right)
    return float(
        0.5 * candidate @ gram @ candidate
        - center @ gram @ candidate
    )


def test_circuit_wall_matches_direct_power_slack() -> None:
    active = np.eye(3, dtype=np.int64)
    candidate = np.asarray([1, 1, 0], dtype=np.int64)
    gram = np.asarray(
        [
            [2.0, -0.2, 0.1],
            [-0.2, 1.7, 0.05],
            [0.1, 0.05, 1.3],
        ]
    )
    wall, coefficients = circuit_matrix(active, candidate)
    assert coefficients.tolist() == [1, 1, 0]
    assert np.isclose(
        wall_slack_from_gram(gram, wall),
        direct_circumcenter_slack(gram, active, candidate),
        atol=1e-12,
    )


def test_oriented_wall_key_is_primitive_and_positive() -> None:
    wall = np.asarray([[0, 2], [2, 0]], dtype=np.int64)
    gram = np.asarray([[1.0, -0.25], [-0.25, 1.0]])
    key, oriented, slack = oriented_wall_key(wall, gram)
    assert key == (0, -2, 0)
    assert oriented.tolist() == [[0, -2], [-2, 0]]
    assert slack == 0.5


def test_circuit_rejects_nonunimodular_simplex() -> None:
    active = np.asarray([[2, 0], [0, 1]], dtype=np.int64)
    try:
        circuit_matrix(active, [1, 1])
    except ValueError as error:
        assert "unimodular" in str(error)
    else:
        raise AssertionError("nonunimodular active rows were accepted")
