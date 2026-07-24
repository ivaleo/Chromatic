"""Тесты разложения определителя на множители (factorization.py).

С версии 1.1.0 перечисляются ВСЕ упорядоченные диагонали (включая перестановки):
суммарное число HNF-матриц по всем диагоналям обязано совпадать с классическим
числом подрешёток данного индекса Σ d2 * d3**2 * d4**3.
"""

from voronoi4d import compute_factorizations, ordered_factorizations, pad_lists_with_ones


def _sublattice_count(n):
    return sum(d[1] * d[2] ** 2 * d[3] ** 3 for d in compute_factorizations(n))


def test_compute_factorizations_are_ordered_and_complete():
    facts = compute_factorizations(12)
    # все разложения длины 4 с произведением 12, включая перестановки
    assert all(len(f) == 4 for f in facts)
    assert all(f[0] * f[1] * f[2] * f[3] == 12 for f in facts)
    assert [1, 1, 1, 12] in facts
    assert [12, 1, 1, 1] in facts          # убывающая диагональ (раньше терялась)
    assert [2, 2, 3, 1] in facts
    # число упорядоченных разложений — мультипликативная функция d_4(n):
    # d_4(12) = d_4(2^2) * d_4(3) = C(5,3) * C(4,3) = 10 * 4 = 40
    assert len(facts) == 40
    assert len(facts) == len({tuple(f) for f in facts})  # без дубликатов


def test_compute_factorizations_prime_power():
    facts = compute_factorizations(25)
    assert len(facts) == 10  # d_4(5^2) = C(5,3) = 10
    assert [5, 5, 1, 1] in facts and [1, 5, 1, 5] in facts


def test_sublattice_counts_match_classical_formula():
    # число подрешёток индекса n в Z^4: n=2 -> 15, n=12 -> 6200, n=49 -> 140050
    assert _sublattice_count(2) == 15
    assert _sublattice_count(12) == 6200
    assert _sublattice_count(49) == 140050


def test_ordered_factorizations_small():
    assert ordered_factorizations(4, 2) == [[1, 4], [2, 2], [4, 1]]


def test_pad_lists_with_ones():
    padded = pad_lists_with_ones([[12], [2, 6], [2, 2, 3]])
    assert padded == [
        [1, 1, 1, 12],
        [1, 1, 2, 6],
        [1, 2, 2, 3],
    ]


def test_pad_lists_with_ones_does_not_mutate_input():
    original = [[12]]
    pad_lists_with_ones(original)
    assert original == [[12]]
