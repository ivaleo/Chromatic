"""Тесты расчёта расстояний (distances.py)."""

import numpy as np

from voronoi4d import (
    check_dist,
    dist_to_s,
    lattice_points_no_central_symmetry,
)


def test_lattice_points_no_central_symmetry(vor):
    points = lattice_points_no_central_symmetry(np.eye(4), 1, vor.max_len)

    assert isinstance(points, list)
    assert any(np.array_equal(p, np.zeros(4)) for p in points), \
        "Нулевая точка должна присутствовать в списке"

    # из пары центрально-симметричных точек остаётся только одна
    points_list = [list(p) for p in points]
    if [1.0, 0.0, 0.0, 0.0] in points_list:
        assert [-1.0, 0.0, 0.0, 0.0] not in points_list


def test_check_dist_does_not_raise():
    check_dist(1.0, 1.0000000001)
    check_dist(1.0, 2.0)


def test_dist_orthogonal(vor):
    """Точка строго напротив центра грани.

    Точка (1.0, 0, 0, 0); ближайшая грань куба при x = 0.5.
    max_len для куба = 2.0, функция возвращает raw_dist * 2 / max_len,
    значит ожидаем 0.5. early_stop=0 — точное значение (по умолчанию значения
    ниже early_stop=1.0 — верхние оценки раннего выхода).
    """
    s = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.isclose(dist_to_s(vor, s, vor.max_len, early_stop=0.0), 0.5, atol=1e-5)


def test_dist_inside_and_boundary_is_zero(vor):
    """Точка внутри ячейки (и центр) дают 0.0 — раньше возвращался мусор/inf."""
    assert dist_to_s(vor, np.array([0.0, 0.0, 0.0, 0.0]), vor.max_len) == 0.0
    assert dist_to_s(vor, np.array([0.2, -0.1, 0.3, 0.0]), vor.max_len) == 0.0
    # верхняя оценка < early_stop остаётся верхней оценкой истинного расстояния
    s = np.array([1.0, 0.0, 0.0, 0.0])
    truncated = dist_to_s(vor, s, vor.max_len)          # early_stop = 1.0
    exact = dist_to_s(vor, s, vor.max_len, early_stop=0.0)
    assert truncated >= exact - 1e-12


def test_dist_diagonal_vertex(vor):
    """Точка на диагонали, за вершиной куба (0.5, 0.5, 0.5, 0.5).

    Расстояние считается от s до вершины: sqrt(0.5^2 * 4) = 1.0.
    """
    vertex = np.array([0.5, 0.5, 0.5, 0.5])
    s = np.array([1.0, 1.0, 1.0, 1.0])

    expected = np.linalg.norm(s - vertex)
    assert np.isclose(dist_to_s(vor, s, vor.max_len), expected, atol=1e-5)


def test_point_on_surface(vor):
    """Точка лежит прямо на грани — расстояние 0."""
    s = np.array([0.5, 0.0, 0.0, 0.0])
    assert np.isclose(dist_to_s(vor, s, vor.max_len), 0.0, atol=1e-5)


def test_point_slightly_offset(vor):
    """Проекция: точка (1.7, 0.1, 0.1, 0.1) проецируется на грань x=0.5.

    Расстояние зависит только от координаты X: 1.7 - 0.5 = 1.2.
    """
    s = np.array([1.7, 0.1, 0.1, 0.1])
    assert np.isclose(dist_to_s(vor, s, vor.max_len), 1.2, atol=1e-5)
