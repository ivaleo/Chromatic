"""Тесты единого фасада chromatic (для каждого доступного бэкенда отдельно)."""

import math

import pytest

import chromatic

D4 = [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
Z3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# --- общие проверки фасада, не требующие бэкендов -----------------------------


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        chromatic.get_backend("несуществующий")


def test_available_backends_is_list():
    assert isinstance(chromatic.available_backends(), list)


def test_optimal_result_feasible_flag():
    r = chromatic.OptimalResult(num_colors=49, diameter=2.0, min_distance=2.16, normalized=1.08)
    assert r.feasible
    r2 = chromatic.OptimalResult(num_colors=16, diameter=2.0, min_distance=1.41, normalized=0.707)
    assert not r2.feasible


# --- параметризация по доступным бэкендам -------------------------------------


def _backend(name):
    if name not in chromatic.available_backends():
        pytest.skip(f"бэкенд {name} не установлен")
    return chromatic.get_backend(name)


@pytest.mark.parametrize("name", ["voronoi4d", "combigeo"])
def test_build_cell_d4_is_24cell(name):
    b = _backend(name)
    cell = b.build_cell(D4)
    assert cell.dim == 4
    assert math.isclose(cell.diameter, 2.0, abs_tol=1e-9)
    assert len(cell.vertices) == 24  # 24-ячейка
    if cell.f_vector is not None:  # combigeo вычисляет f-вектор
        assert cell.f_vector == [24, 96, 96, 24]


def test_find_optimal_combigeo_d4_16():
    # combigeo быстрый — проверяем содержательный результат на нём;
    # voronoi4d на индексе 16 без fpylll медленный (см. examples/cross_validate.py)
    b = _backend("combigeo")
    r = b.find_optimal(D4, 16)
    assert r.num_colors == 16
    assert math.isclose(r.diameter, 2.0, abs_tol=1e-9)
    # лучшая подрешётка индекса 16 для D4 даёт D = sqrt(2)
    assert math.isclose(r.min_distance, math.sqrt(2), abs_tol=1e-4)
    assert math.isclose(r.normalized, r.min_distance / r.diameter, abs_tol=1e-9)
    assert not r.feasible  # d = 0.707 < 1


def test_find_optimal_voronoi4d_small():
    # voronoi4d на малом индексе (быстро даже без fpylll)
    b = _backend("voronoi4d")
    r = b.find_optimal(D4, 8)
    assert r.num_colors == 8
    assert math.isclose(r.diameter, 2.0, abs_tol=1e-9)


@pytest.mark.parametrize("name", ["voronoi4d", "combigeo"])
def test_lll_preserves_lattice(name):
    import numpy as np

    b = _backend(name)
    reduced = b.lll_reduce(D4)
    assert math.isclose(abs(np.linalg.det(reduced)), 2.0, abs_tol=1e-6)


def test_combigeo_only_dims():
    b = _backend("combigeo")
    cell = b.build_cell(Z3)  # размерность 3 — только combigeo
    assert cell.f_vector == [8, 12, 6]


def test_voronoi4d_rejects_non4d():
    b = _backend("voronoi4d")
    with pytest.raises(ValueError):
        b.build_cell(Z3)  # voronoi4d только 4D
