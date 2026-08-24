"""Проверка точного перечисления вершин: известные многогранники + сверка с qhull."""

from fractions import Fraction as Fr
import itertools

import numpy as np
import pytest

from chromatic_research.core.exact_dd import dd_vertices


def _box(dim, half=1):
    rows, rhs = [], []
    for i in range(dim):
        for sign in (1, -1):
            row = [Fr(0)] * dim
            row[i] = Fr(sign)
            rows.append(row)
            rhs.append(Fr(half))
    return rows, rhs


def test_cube_3d():
    rows, rhs = _box(3)
    verts = dd_vertices(rows, rhs, 3, Fr(2))
    assert len(verts) == 8
    assert set(verts) == {tuple(Fr(s) for s in c)
                          for c in itertools.product([-1, 1], repeat=3)}


def test_cube_5d():
    rows, rhs = _box(5)
    verts = dd_vertices(rows, rhs, 5, Fr(2))
    assert len(verts) == 32


def test_cross_polytope_4d():
    """|x_1| + ... + |x_4| <= 1: 16 фасет, 8 вершин (сильно вырожденный случай)."""
    dim = 4
    rows, rhs = [], []
    for signs in itertools.product([1, -1], repeat=dim):
        rows.append([Fr(s) for s in signs])
        rhs.append(Fr(1))
    verts = dd_vertices(rows, rhs, dim, Fr(2))
    assert len(verts) == 2 * dim
    for v in verts:
        assert sum(abs(x) for x in v) == 1


def test_simplex():
    dim = 4
    rows, rhs = [], []
    for i in range(dim):
        row = [Fr(0)] * dim
        row[i] = Fr(-1)
        rows.append(row)
        rhs.append(Fr(0))
    rows.append([Fr(1)] * dim)
    rhs.append(Fr(1))
    verts = dd_vertices(rows, rhs, dim, Fr(2))
    assert len(verts) == dim + 1


def test_empty():
    rows = [[Fr(1), Fr(0)], [Fr(-1), Fr(0)]]
    rhs = [Fr(-1), Fr(-1)]          # x <= -1 и -x <= -1
    assert dd_vertices(rows, rhs, 2, Fr(10)) == []


def test_redundant_constraints_ignored():
    """Дублирующие и заведомо избыточные неравенства не портят ответ."""
    rows, rhs = _box(3)
    rows = rows + rows + [[Fr(1), Fr(1), Fr(1)]]
    rhs = rhs + rhs + [Fr(100)]
    verts = dd_vertices(rows, rhs, 3, Fr(2))
    assert len(verts) == 8


@pytest.mark.parametrize("seed", range(6))
def test_matches_qhull_random(seed):
    """Случайные политопы: тот же набор вершин, что у независимого qhull."""
    scipy_spatial = pytest.importorskip("scipy.spatial")
    dim = 4
    rng = np.random.default_rng(seed)
    normals = rng.standard_normal((14, dim))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # осевые неравенства делают многогранник ограниченным (иначе сравнивать
    # нечего: qhull и DD по-разному обходятся с неограниченностью)
    axes = np.vstack([np.eye(dim), -np.eye(dim)])
    normals = np.vstack([normals, axes])
    # рационализуем, чтобы сравнивать одну и ту же систему
    rows = [[Fr(int(round(x * 1000)), 1000) for x in row] for row in normals]
    rhs = [Fr(1)] * len(rows)
    exact = dd_vertices(rows, rhs, dim, Fr(50))

    A = np.array([[float(v) for v in row] for row in rows])
    b = np.ones(len(rows))
    hs = scipy_spatial.HalfspaceIntersection(np.hstack([A, -b[:, None]]),
                                             np.zeros(dim))
    approx = hs.intersections
    got = np.array([[float(v) for v in p] for p in exact])
    assert len(got) == len(approx)
    for point in approx:
        assert np.min(np.linalg.norm(got - point, axis=1)) < 1e-7
