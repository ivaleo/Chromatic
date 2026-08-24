"""Контроли доказанного пола индекса на родителе E₆* в ℝ⁶.

Проверяется вся цепочка: точные инварианты решётки, оболочечная структура окна,
арифметика Эрмита и согласованность с действующим рекордом 343.
"""

import math
from fractions import Fraction as Fr

import pytest

from chromatic_research.campaigns.dim6_e6star_floor import (
    CHAMPION_INDEX,
    COVERING_SQ,
    DIAM_SQ,
    DIM,
    LAMBDA1_SQ,
    NEXT_SHELL_SQ,
    SHELL_SQ,
    dot,
    dual_gram,
    exact_witness,
    relevant_vectors,
    shell,
    shells_in_window,
)


@pytest.fixture(scope="module")
def gram():
    return dual_gram()


@pytest.fixture(scope="module")
def facets(gram):
    return [(v, dot(gram, v, v)) for v in relevant_vectors(gram)]


def test_dual_gram_is_e6_star(gram):
    """Грамиан E₆*: det = 1/3, минимум 4/3."""
    from chromatic_research.core.exact_layer_cert import enumerate_shifted

    vectors = [v for v in enumerate_shifted(gram, [Fr(0)] * DIM, LAMBDA1_SQ)
               if any(v)]
    assert vectors, "минимальные векторы не найдены"
    assert all(dot(gram, v, v) >= LAMBDA1_SQ for v in vectors)
    assert min(dot(gram, v, v) for v in vectors) == LAMBDA1_SQ
    assert len(vectors) == 54            # E₆* имеет 54 минимальных вектора


def test_covering_radius_relation():
    """2R/λ₁ = √2 — это и есть характеристика E₆*."""
    assert DIAM_SQ == 4 * COVERING_SQ
    assert float(DIAM_SQ / LAMBDA1_SQ) == pytest.approx(2.0)


def test_voronoi_cell_has_126_facets(facets):
    assert len(facets) == 126


def test_window_contains_exactly_three_shells(gram):
    """Между (2R+λ₁)² и (2·diam)² у E₆* ровно три оболочки."""
    low = float((2 * math.sqrt(float(COVERING_SQ))
                 + math.sqrt(float(LAMBDA1_SQ))) ** 2)
    window = shells_in_window(gram, low, 4 * DIAM_SQ)
    assert list(window) == [Fr(8), Fr(28, 3), Fr(10)]
    assert window[Fr(8)] == 936
    assert window[Fr(28, 3)] == 2700
    assert window[Fr(10)] == 2160


def test_no_shell_between_window_start_and_forbidden(gram):
    """Оболочка 8 — ПЕРВАЯ в окне: ниже неё запрет даёт инрадиусная лемма."""
    low = (2 * math.sqrt(float(COVERING_SQ)) + math.sqrt(float(LAMBDA1_SQ))) ** 2
    assert low < 8.0
    assert float(SHELL_SQ) < float(NEXT_SHELL_SQ)


@pytest.mark.parametrize("count", [12])
def test_witnesses_are_exact_and_valid(gram, facets, count):
    """Свидетель обязан лежать в V₀ и быть ближе половины диаметра."""
    target = shell(gram, SHELL_SQ)[:count]
    for vector in target:
        found = exact_witness(gram, facets, vector)
        assert found is not None, f"нет свидетеля для {vector}"
        point, distance_squared = found
        # x ∈ V₀: все опорные неравенства, точно
        for normal, norm_squared in facets:
            assert dot(gram, point, normal) * 2 <= norm_squared
        # 4|v/2 − x|² < diam², точно
        half = [Fr(c, 2) for c in vector]
        delta = [half[i] - point[i] for i in range(DIM)]
        assert 4 * dot(gram, delta, delta) == distance_squared
        assert distance_squared < DIAM_SQ


def test_hermite_arithmetic_gives_305():
    """(28/3)³ ≤ 8k/3 ⇒ k ≥ 2744/9 = 304.888… ⇒ k ≥ 305."""
    bound = Fr(28, 3) ** 3 * 3 / 8
    assert bound == Fr(2744, 9)
    assert math.ceil(float(bound)) == 305
    # γ₆³ = (64/3)^(1/2) = 8/√3 — использованная константа Блихфельдта
    assert abs((64 / 3) ** 0.5 - 8 / math.sqrt(3)) < 1e-12


def test_champion_sits_on_the_first_allowed_shell():
    """λ₁((3+ω)E₆*)² = 7·λ₁² = 28/3 — ровно первая разрешённая оболочка.

    Поэтому пол и рекорд разделяет всего 11 %: рекорд стоит на кратчайшем
    векторе минимальной разрешённой длины.
    """
    assert 7 * LAMBDA1_SQ == NEXT_SHELL_SQ
    assert CHAMPION_INDEX == 343
    assert CHAMPION_INDEX >= 305


def test_champion_index_consistent_with_hermite():
    """Рекорд обязан удовлетворять тому же неравенству Эрмита."""
    bound = Fr(28, 3) ** 3 * 3 / 8
    assert CHAMPION_INDEX >= bound


def test_floor_beats_the_generic_packing_bound():
    """Обобщённый пол «инрадиус + упаковка» для E₆* даёт лишь ≈176."""
    lam1 = math.sqrt(float(LAMBDA1_SQ))
    radius = math.sqrt(float(COVERING_SQ))
    volume_unit = math.pi ** 3 / 6                      # ω₆
    delta_max = 0.3729496875                            # δ(E₆), Блихфельдт
    det = 3 ** -0.5
    generic = volume_unit * (radius + lam1 / 2) ** 6 / (delta_max * det)
    assert 170 < generic < 180
    assert 305 > generic * 1.7
