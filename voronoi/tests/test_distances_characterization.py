"""Опорные значения dist_to_s: страховка для оптимизаций горячего пути.

Значения получены на текущей реализации 2026-08-05 и не должны меняться
ни при каких оптимизациях — только при изменении самого алгоритма.
"""

import numpy as np
import pytest

from voronoi4d import VoronoiPolyhedra, dist_to_s

D4 = np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float)

# 12 точек из np.random.default_rng(20260805).normal(size=4) * 1.2
EXPECTED = [
    1.877943432871477, 0.15194681369983248, 2.8776962779871913,
    1.8344149411138804, 0.3182553077904282, 1.1951097888868285,
    2.419226762485762, 0.921579230714697, 0.9523475752796856,
    1.3212828842435729, 1.5721696031153471, 2.5254306489745346,
]

# точки с известным геометрическим смыслом
FIXED = [
    ((1.0, 0.0, 0.0, 0.0), 0.0),                  # внутри ячейки
    ((0.5, 0.5, 0.0, 0.0), 0.0),                  # внутри ячейки
    ((1.0, 1.0, 0.0, 0.0), 0.7071067811865475),   # на грани
    ((0.75, 0.25, 0.5, 0.0), 0.17677669529663687),
    ((2.0, 0.0, 0.0, 0.0), 1.0),                  # ровно порог d = 1
]


@pytest.fixture(scope="module")
def cell():
    vor = VoronoiPolyhedra(D4)
    vor.build(verbose=False)
    return vor


def test_random_points_match_reference(cell):
    rng = np.random.default_rng(20260805)
    got = [float(dist_to_s(cell, rng.normal(size=4) * 1.2, cell.max_len, early_stop=0.0))
           for _ in range(12)]
    assert got == pytest.approx(EXPECTED, abs=1e-12)


@pytest.mark.parametrize("point,expected", FIXED)
def test_fixed_points_match_reference(cell, point, expected):
    got = dist_to_s(cell, np.array(point), cell.max_len, early_stop=0.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_result_is_plain_float(cell):
    got = dist_to_s(cell, np.array([1.0, 1.0, 0.0, 0.0]), cell.max_len, early_stop=0.0)
    assert type(got) is float, "функция должна возвращать float, а не np.float64"
