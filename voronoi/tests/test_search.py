"""Интеграционные тесты поиска подрешёток (search.py)."""

import numpy as np
import pytest

from voronoi4d import (
    VoronoiPolyhedra,
    dist_to_s,
    find_optimal,
    lattice_points_within,
    lll_reduce,
)

# решётка D4 (шахматная): её 24-ячейка компактна относительно подрешёток
D4_GRID = np.array([
    [1, 1, 0, 0],
    [1, -1, 0, 0],
    [0, 1, -1, 0],
    [0, 0, 1, -1],
], dtype=float)


@pytest.fixture(scope="module")
def vor_d4():
    vor = VoronoiPolyhedra(D4_GRID)
    vor.build(verbose=False)
    return vor


def test_find_optimal_threshold_semantics(vor, tmp_path):
    """threshold управляет отбраковкой честно (до 1.1.0 порог 1 был зашит).

    Для Z^4 и det=4 все подрешётки дают d < 1: с threshold=1.0 результат пуст,
    с threshold=0.0 лучшая подрешётка возвращается с точным 0 < d < 1.
    """
    grid = np.eye(4, dtype=float)

    det_dist, det_center, det_mat = find_optimal(
        range(4, 5), 1, grid, vor, vor.max_len,
        threshold=1.0, output_file=str(tmp_path / "r1.txt"), verbose=False,
    )
    assert det_dist == {} and det_center == {} and det_mat == {}

    det_dist, det_center, det_mat = find_optimal(
        range(4, 5), 1, grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r2.txt"), verbose=False,
    )
    # каждая подрешётка индекса 4 в Z^4 содержит единичный вектор — одноцветные
    # кубы касаются, точный оптимум d = 0 (раньше фильтр это скрывал)
    assert np.isclose(det_dist[4], 0.0, atol=1e-9)


def test_find_optimal_smoke_d4(vor_d4, tmp_path):
    """Малый end-to-end прогон: det=16 для решётки D4.

    Заодно регрессия: сохранённая матрица должна воспроизводить сохранённое
    расстояние. Раньше в list_mats сохранялась ссылка на мутируемую матрицу,
    и все сохранённые матрицы перетирались её последним состоянием.
    """
    output_file = str(tmp_path / "results.txt")

    det_dist, det_center, det_mat = find_optimal(
        range(16, 17), 1, D4_GRID, vor_d4, vor_d4.max_len,
        threshold=0.0, output_file=output_file, verbose=False,
    )

    assert 16 in det_dist and 16 in det_center and 16 in det_mat

    # матрица перехода верхнетреугольная с определителем 16
    mat = det_mat[16]
    assert np.allclose(mat, np.triu(mat))
    assert np.isclose(np.linalg.det(mat), 16.0)

    # результат записан в файл
    content = open(output_file).read()
    assert "Determinant: 16" in content

    # 2*D4 — оптимальная подрешётка индекса 16: точное значение d = sqrt(1/2)
    assert np.isclose(det_dist[16], np.sqrt(0.5), atol=1e-9)

    # регрессия mat.copy(): пересчитываем расстояние по сохранённой матрице
    # точным перебором в границе достаточности |v| < (d+1)*diam
    sub_grid = np.dot(mat, D4_GRID)
    sub_grid_lll = lll_reduce(sub_grid)
    bound = (det_dist[16] + 1.0) * vor_d4.max_len
    centers = lattice_points_within(sub_grid_lll, bound)

    recomputed = min(
        dist_to_s(vor_d4, 0.5 * center, vor_d4.max_len, early_stop=0.0)
        for center in centers
    )

    assert np.isclose(recomputed, det_dist[16], atol=1e-9), \
        "Сохранённая матрица не соответствует сохранённому расстоянию"
