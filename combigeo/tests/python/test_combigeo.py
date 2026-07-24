"""Тесты python-модуля combigeo + сверка с эталонной python-реализацией voronoi4d.

Запуск: .venv-test/bin/python -m pytest tests/python/ -v
"""

import math

import pytest

import combigeo

D4 = [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]
Z3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


# --------------------------------------------------------------------------------
# модульные тесты API


def test_voronoi_cell_24cell():
    cell = combigeo.voronoi_cell(D4)
    assert cell.f_vector == [24, 96, 96, 24]  # 24-ячейка
    assert math.isclose(cell.diameter, 2.0, abs_tol=1e-9)
    assert cell.contains([0, 0, 0, 0])
    assert not cell.contains([2, 2, 2, 2])


def test_voronoi_cell_cube():
    cell = combigeo.voronoi_cell(Z3)
    assert cell.f_vector == [8, 12, 6]
    assert math.isclose(cell.diameter, math.sqrt(3), abs_tol=1e-9)


def test_lll_preserves_det():
    import numpy as np

    reduced = combigeo.lll_reduce(D4)
    assert math.isclose(abs(np.linalg.det(reduced)), 2.0, abs_tol=1e-9)


def test_sublattice_counts():
    assert combigeo.count_sublattices(4, 2) == 15
    assert combigeo.count_sublattices(4, 3) == 40
    assert len(combigeo.sublattices(3, 2)) == 7


def test_min_color_distance_known_values():
    # 2*Z^3: соседние одноцветные кубы на расстоянии 1
    sub = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert math.isclose(combigeo.min_color_distance(Z3, sub), 1.0, abs_tol=1e-9)

    # 2*D4: D = sqrt(2) (аналитика: проекция (1,1,0,0) -> (0.5,0.5,0,0))
    sub4 = [[4, 0, 0, 0], [2, 2, 0, 0], [2, 0, 2, 0], [2, 0, 0, 2]]
    assert math.isclose(combigeo.min_color_distance(D4, sub4), math.sqrt(2), abs_tol=1e-9)


def test_find_optimal_d4_16():
    r = combigeo.find_optimal(D4, 16)
    assert r.examined == combigeo.count_sublattices(4, 16)
    assert math.isclose(r.best.min_distance, math.sqrt(2), abs_tol=1e-9)
    assert math.isclose(r.normalized, math.sqrt(2) / 2, abs_tol=1e-9)


# --------------------------------------------------------------------------------
# валидация входных данных (защита от heap-OOB в release-сборке без assert)


def test_mismatched_point_dimension_raises():
    cell = combigeo.voronoi_cell(Z3)
    with pytest.raises(Exception):  # точка dim=5 против ячейки dim=3
        combigeo.distance_to_cell([0, 0, 0, 0, 0], cell)


def test_mismatched_basis_dimensions_raises():
    with pytest.raises(Exception):  # basis dim=3, sub_basis dim=2
        combigeo.min_color_distance(Z3, [[2, 0], [0, 2]])


def test_nonsquare_basis_raises():
    with pytest.raises(Exception):
        combigeo.voronoi_cell([[1, 0, 0], [0, 1, 0]])  # 2 строки по 3


def test_nan_basis_raises():
    with pytest.raises(Exception):
        combigeo.voronoi_cell([[1, 0], [float("nan"), 1]])


def test_invalid_index_raises():
    with pytest.raises(Exception):
        combigeo.find_optimal(Z3, 0)  # index < 1 — нет разложений


# --------------------------------------------------------------------------------
# сверка с эталонной python-реализацией voronoi4d (известно-рабочая для 4D)


@pytest.fixture(scope="module")
def vor4():
    # эталонный пакет из соседнего проекта воркспейса (../voronoi), установлен в venv;
    # на машине без него сверочные тесты пропускаются, а не падают
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    voronoi4d = pytest.importorskip("voronoi4d")

    grid = np.array(D4, dtype=float)
    vor = voronoi4d.VoronoiPolyhedra(grid)
    vor.build(verbose=False)
    return vor


def test_distance_agreement_with_voronoi4d(vor4):
    """Главная сверка: расстояние от точки до ячейки — GJK (C++) против
    каскада проекций (python, проверенный) на содержательном наборе точек."""
    import numpy as np

    from voronoi4d import dist_to_s

    cell = combigeo.voronoi_cell(D4)
    assert math.isclose(cell.diameter, vor4.max_len, abs_tol=1e-9)

    # точки: полувекторы решётки (как в реальном алгоритме) + детерминированные случайные
    rng = np.random.default_rng(42)
    points = [0.5 * np.array(v, dtype=float) for v in
              ([2, 0, 0, 0], [1, 1, 0, 0], [2, 2, 0, 0], [1, 1, 1, 1], [3, 1, 1, 1],
               [2, 1, 1, 0], [4, 0, 0, 0], [2, 2, 2, 0])]
    points += [rng.uniform(-2.5, 2.5, size=4) for _ in range(40)]

    for s in points:
        d_py = dist_to_s(vor4, s, vor4.max_len)          # нормированное: 2*dist/diam
        d_cpp = 2.0 * combigeo.distance_to_cell(list(s), cell) / cell.diameter

        if d_py >= 1.0:
            # python завершил полный перебор — значения должны совпасть точно
            assert math.isclose(d_py, d_cpp, abs_tol=1e-6), \
                f"расхождение в точке {s}: {d_py} vs {d_cpp}"
        else:
            # python делает ранний выход при d < 1 и возвращает верхнюю оценку
            # (для алгоритма достаточно факта d < 1); GJK даёт точное значение
            # (сверено с независимым QP-арбитром scipy до 1e-12)
            assert d_cpp <= d_py + 1e-6, f"GJK больше оценки python в {s}: {d_cpp} vs {d_py}"
            assert d_cpp < 1.0 + 1e-6, f"несогласованный реджект в {s}"


def test_min_distance_agreement_with_voronoi4d(vor4):
    """Минимум по подрешётке: C++ min_color_distance против точного перебора python.

    С 1.1.0 перебор кандидатов точный (граница достаточности |v| < D + diam),
    а dist_to_s с early_stop=0 возвращает точные значения при любом d.
    """
    import numpy as np

    from voronoi4d import dist_to_s, lattice_points_within, lll_reduce, shortest_vector

    sub = [[4, 0, 0, 0], [2, 2, 0, 0], [2, 0, 2, 0], [2, 0, 0, 2]]  # 2*D4

    sub_lll = lll_reduce(np.array(sub, dtype=float))
    d0 = dist_to_s(vor4, 0.5 * np.asarray(shortest_vector(sub_lll), dtype=float),
                   vor4.max_len, early_stop=0.0)
    bound = (d0 + 1.0) * vor4.max_len
    d_py = min(dist_to_s(vor4, 0.5 * np.asarray(c, dtype=float), vor4.max_len,
                         early_stop=0.0)
               for c in lattice_points_within(sub_lll, bound))

    # python: d = 2*dist(s)/diam = D/diam; C++ min_color_distance возвращает D
    d_cpp = combigeo.min_color_distance(D4, sub) / vor4.max_len
    assert math.isclose(d_py, d_cpp, abs_tol=1e-6)
    assert math.isclose(d_py, math.sqrt(0.5), abs_tol=1e-9)  # точное значение 2*D4


# --------------------------------------------------------------------------------
# доработки 1.1.0: валидация, версия, transition в базисе пользователя, масштаб


def test_version():
    assert combigeo.__version__ == "1.1.0"


def test_lll_reduce_rejects_jagged_input():
    # раньше рваный список читал память за границей буфера (ASan) — теперь исключение
    with pytest.raises(ValueError):
        combigeo.lll_reduce([[1, 0], [0, 1, 5]])
    with pytest.raises(ValueError):
        combigeo.lll_reduce([[1, 0], [0, float("nan")]])


def test_find_optimal_range_kwargs_and_shared_cache():
    res = combigeo.find_optimal_range(Z3, frm=2, to=3)
    assert sorted(res) == [2, 3]
    assert res[2].examined == combigeo.count_sublattices(3, 2)


def test_transition_in_user_basis():
    # transition * входной_базис порождает ту же подрешётку, что sub_basis
    # (для скошенного базиса это ловило старую ошибку кадра LLL)
    import numpy as np

    skew = [[3, 1], [2, 1]]  # унимодулярная перестройка Z^2
    r = combigeo.find_optimal(skew, index=5)
    t = np.array(r.best.transition, dtype=float)
    from_t = t @ np.array(skew, dtype=float)
    sb = np.array(r.best.sub_basis, dtype=float)
    coeffs = sb @ np.linalg.inv(from_t)
    assert np.allclose(coeffs, np.round(coeffs), atol=1e-9)
    assert math.isclose(abs(np.linalg.det(coeffs)), 1.0, abs_tol=1e-9)


def test_extreme_scale_invariance():
    # d — масштабный инвариант; ячейка и расстояния корректны при масштабе 1e8
    s = 1e8
    hex_big = [[2 * s, 0], [1 * s, math.sqrt(3) * s]]
    cell = combigeo.voronoi_cell(hex_big)
    assert cell.f_vector == [6, 6]
    r = combigeo.find_optimal(hex_big, index=7)
    assert math.isclose(r.normalized, math.sqrt(7) / 2, rel_tol=1e-9)
