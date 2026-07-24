"""Тесты генерации базисов решёток (grids.py)."""

import numpy as np

from voronoi4d import (
    canonical_form_by_rows,
    check_grid,
    generate_integer_grids,
    generate_random_matrix,
    normalize_rows,
)


def test_normalize_rows():
    matrix = np.array([
        [-1.0, 2.0, 0.0],
        [0.0, -3.0, 1.0],
        [2.0, -1.0, 0.0],
    ])
    result = normalize_rows(matrix)

    # первый ненулевой элемент каждой строки неотрицателен
    for row in result:
        first_non_zero = next((x for x in row if x != 0), None)
        assert first_non_zero is None or first_non_zero > 0

    # исходная матрица не изменилась
    assert matrix[0][0] == -1.0


def test_canonical_form_by_rows_is_deterministic():
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    shuffled = matrix[::-1].copy()
    assert np.array_equal(canonical_form_by_rows(matrix), canonical_form_by_rows(shuffled))


def test_generate_random_matrix_nonsingular():
    mat = generate_random_matrix()
    assert mat.shape == (4, 4)
    assert not np.isclose(np.linalg.det(mat), 0, atol=1e-8)
    assert np.array_equal(mat[0], [1.0, 0.0, 0.0, 0.0])


def test_check_grid_identity():
    # единичная решётка: ортогональные векторы единичной длины — проходит проверку
    assert check_grid(np.eye(4))


def test_check_grid_rejects_short_vectors():
    grid = np.eye(4) * 0.5  # все векторы короче 1
    assert not check_grid(grid)


def test_generate_integer_grids_smoke():
    """Регрессия: модуль раньше падал с NameError при импорте."""
    grids = generate_integer_grids(coeff_range=(0, 2), show_progress=False)

    assert len(grids) > 0
    for grid in grids:
        assert grid.shape == (4, 4)
        assert not np.isclose(np.linalg.det(grid), 0)

    # уникальность
    for i in range(len(grids)):
        for j in range(i + 1, len(grids)):
            assert not np.allclose(grids[i], grids[j])
