"""Контроли циклотомического семейства ℤ[ζ₇] для индекса 337."""

import math

import numpy as np
import pytest

from chromatic_research.campaigns.dim6_cyclotomic7_337 import (
    DIM,
    INDEX,
    P,
    UNIT_LOGS,
    gram_from_alpha,
    prime_ideals_of_norm,
)


def _zeta_multiplication() -> np.ndarray:
    """Матрица умножения на ζ в базисе 1, ζ, …, ζ⁵ (ζ⁶ = −(1+ζ+…+ζ⁵))."""
    matrix = np.zeros((DIM, DIM), dtype=int)
    for i in range(DIM - 1):
        matrix[i, i + 1] = 1
    matrix[DIM - 1, :] = -1
    return matrix


def test_337_splits_completely():
    """337 ≡ 1 (mod 7): ровно шесть простых идеалов нормы 337."""
    assert INDEX % P == 1
    ideals = prime_ideals_of_norm()
    assert len(ideals) == P - 1


def test_ideals_have_index_337():
    for ideal in prime_ideals_of_norm():
        assert round(abs(np.linalg.det(ideal.astype(float)))) == INDEX


def test_ideals_are_zeta_modules():
    """Идеал обязан быть ℤ[ζ]-подмодулем: умножение на ζ не выводит из него."""
    mult = _zeta_multiplication()
    for ideal in prime_ideals_of_norm():
        image = ideal @ mult
        coordinates = np.linalg.solve(ideal.T.astype(float), image.T.astype(float)).T
        assert np.allclose(coordinates, np.rint(coordinates), atol=1e-7)


def test_ideals_are_distinct():
    ideals = prime_ideals_of_norm()
    keys = {tuple(map(tuple, ideal.tolist())) for ideal in ideals}
    assert len(keys) == len(ideals)


def test_trace_form_at_alpha_one():
    """α = 1 даёт стандартную следовую форму: Tr(ζ^i ζ^{-j}) = 6 или −1."""
    gram = gram_from_alpha(np.ones(3))
    expected = np.full((DIM, DIM), -1.0)
    np.fill_diagonal(expected, 6.0)
    assert np.allclose(gram, expected)
    # спектр {1, 7×5} и det = 7^5 = disc Q(ζ₇)
    eigenvalues = np.sort(np.linalg.eigvalsh(gram))
    assert np.allclose(eigenvalues, [1, 7, 7, 7, 7, 7])
    assert round(np.linalg.det(gram)) == P ** 5


def test_gram_is_circulant_folded():
    """G_ij зависит только от |i−j| mod 7, свёрнутого в 0..3."""
    rng = np.random.default_rng(0)
    alpha = np.exp(rng.normal(0, 0.3, 3))
    gram = gram_from_alpha(alpha)
    for i in range(DIM):
        for j in range(DIM):
            for a in range(DIM):
                for b in range(DIM):
                    ka = (i - j) % P
                    kb = (a - b) % P
                    if min(ka, P - ka) == min(kb, P - kb):
                        assert gram[i, j] == pytest.approx(gram[a, b])


def test_trace_relation():
    """Из Σ_{k=1}^{6} ζ^k = −1 следует m₁ + m₂ + m₃ = −m₀/2."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        alpha = np.exp(rng.normal(0, 0.5, 3))
        gram = gram_from_alpha(alpha)
        m0 = gram[0, 0]
        m1, m2, m3 = gram[0, 1], gram[0, 2], gram[0, 3]
        assert m1 + m2 + m3 == pytest.approx(-m0 / 2)


def test_positive_definite_iff_totally_positive():
    gram = gram_from_alpha(np.array([1.0, 1.0, 1.0]))
    assert np.linalg.eigvalsh(gram)[0] > 0
    # не вполне положительное α даёт неопределённую форму
    gram = gram_from_alpha(np.array([1.0, -1.0, 1.0]))
    assert np.linalg.eigvalsh(gram)[0] < 0


def test_cyclotomic_units_are_units():
    """ξ_a = sin(2πa/7)/sin(2π/7) — единицы: произведение сопряжений равно ±1."""
    for a in (2, 3):
        conjugates = [math.sin(2 * math.pi * a * j / P) / math.sin(2 * math.pi * j / P)
                      for j in (1, 2, 3)]
        assert abs(abs(np.prod(conjugates)) - 1.0) < 1e-12


def test_unit_logs_are_traceless_and_independent():
    """Логарифмы лежат в плоскости Σ = 0 и порождают решётку ранга 2."""
    assert UNIT_LOGS.shape == (2, 3)
    assert np.allclose(UNIT_LOGS.sum(axis=1), 0.0, atol=1e-12)
    assert np.linalg.matrix_rank(UNIT_LOGS) == 2


def test_unit_action_is_a_change_of_basis():
    """α ↦ α·θ² даёт ту же решётку в другом базисе: G' = M G Mᵀ.

    Здесь θ = ζ + ζ⁻¹ — единица (N(1+ζ²) = Φ₇(−1) = 1), а
    Q_{αθθ̄}(x) = Q_α(θx).  Поэтому изометричны именно РЕШЁТКИ, а спектры
    матриц Грама в стандартном базисе, разумеется, разные — сравнивать надо
    целочисленную эквивалентность, что и делает этот контроль.
    """
    mult = _zeta_multiplication()
    inverse = np.rint(np.linalg.inv(mult.astype(float))).astype(int)
    assert np.allclose(mult @ inverse, np.eye(DIM))
    theta = mult + inverse                       # умножение на ζ + ζ⁻¹
    assert round(abs(np.linalg.det(theta.astype(float)))) == 1   # единица

    rng = np.random.default_rng(2)
    logs = rng.normal(0, 0.3, 3)
    logs -= logs.mean()
    gram = gram_from_alpha(np.exp(logs))
    shifted = gram_from_alpha(np.exp(logs + UNIT_LOGS[0]))
    assert np.allclose(theta @ gram @ theta.T, shifted)
