"""Тесты LLL-приведения (lll.py): оба бэкенда — fpylll и чистый python."""

import numpy as np
import pytest

from voronoi4d import HAS_FPYLLL, lll_reduce, lll_reduce_python

TEST_MATRIX = np.array([
    [2, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
], dtype=float)


def _check_reduction(reduced, original):
    # определитель решётки сохраняется (с точностью до знака)
    assert np.isclose(
        abs(np.linalg.det(reduced)), abs(np.linalg.det(original)), atol=1e-6
    ), "LLL-приведение должно сохранять определитель"

    # приведённый базис не длиннее исходного
    assert np.linalg.norm(reduced, axis=1).max() <= np.linalg.norm(original, axis=1).max() + 1e-9


def test_lll_reduce_python():
    _check_reduction(lll_reduce_python(TEST_MATRIX.copy()), TEST_MATRIX)


@pytest.mark.skipif(not HAS_FPYLLL, reason="fpylll не установлена")
def test_lll_reduce_fpylll():
    _check_reduction(lll_reduce(TEST_MATRIX.copy(), precision=3), TEST_MATRIX)


def test_backends_agree_on_det():
    """Оба бэкенда дают базис одной и той же решётки."""
    py = lll_reduce_python(TEST_MATRIX.copy())
    assert np.isclose(abs(np.linalg.det(py)), 2.0, atol=1e-6)

    if HAS_FPYLLL:
        fp = lll_reduce(TEST_MATRIX.copy(), precision=3)
        assert np.isclose(abs(np.linalg.det(fp)), 2.0, atol=1e-6)


def test_lll_identity():
    """Единичная матрица уже приведена — не должна меняться по длинам."""
    reduced = lll_reduce_python(np.eye(4))
    assert np.allclose(np.abs(np.linalg.det(reduced)), 1.0)
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0)
