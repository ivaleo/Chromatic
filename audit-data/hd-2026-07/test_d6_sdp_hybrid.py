import numpy as np

from d6_sdp_hybrid import (
    canonical_simplex,
    circumradius_squared,
    covering_lmi_numeric,
    separation_dual_matrix,
    simplex_rows,
)


def test_simplex_canonicalization_is_translation_invariant() -> None:
    original = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    translated = np.asarray([[-1, 0], [-1, 1]], dtype=np.int64)
    assert canonical_simplex(original) == canonical_simplex(translated)
    assert simplex_rows(canonical_simplex(original)).shape == (2, 2)


def test_covering_lmi_matches_circumradius_schur_complement() -> None:
    gram = np.eye(2)
    rows = np.eye(2, dtype=np.int64)
    radius_squared = circumradius_squared(gram, rows)
    assert np.isclose(radius_squared, 0.5, atol=1e-12)
    on_boundary = covering_lmi_numeric(gram, rows, radius_squared)
    below_boundary = covering_lmi_numeric(
        gram, rows, radius_squared - 1e-3
    )
    assert np.linalg.eigvalsh(on_boundary).min() >= -1e-12
    assert np.linalg.eigvalsh(below_boundary).min() < -1e-5


def test_fixed_dual_multiplier_gives_exact_one_dimensional_distance() -> None:
    coordinate = np.asarray([3])
    rows = np.asarray([[-1], [1]])
    multipliers = np.asarray([0.0, 4.0])
    certificate = separation_dual_matrix(
        coordinate, rows, multipliers
    )
    assert certificate.shape == (1, 1)
    assert np.isclose(certificate[0, 0], 4.0, atol=1e-12)
