"""Тест кросс-валидации бэкендов (требует ОБОИХ: voronoi4d и combigeo).

Рутинный набор использует малый индекс — voronoi4d без fpylll медленный на
больших индексах. Полная кросс-валидация (включая пригодные раскраски) —
в examples/cross_validate.py.
"""

import pytest

import chromatic

D4 = [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]


def _need_both():
    avail = chromatic.available_backends()
    if "voronoi4d" not in avail or "combigeo" not in avail:
        pytest.skip("нужны оба бэкенда (voronoi4d и combigeo)")


def test_compare_requires_dim4():
    _need_both()
    with pytest.raises(ValueError):
        chromatic.compare_backends([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [2])


def test_compare_nontrivial_index_agrees():
    """Индекс 16 (2·D4, d = sqrt(1/2) — невырожденный случай): оба движка
    обязаны совпасть по диаметру и ТОЧНОМУ d, а не сравнить 0.0 с 0.0.

    Независимые геометрические алгоритмы (GJK против каскада проекций) дают
    один результат — сильная взаимная проверка корректности.
    """
    import math

    _need_both()
    report = chromatic.compare_backends(D4, [16])
    assert report.agree, f"расхождения: {report.discrepancies}"
    rv = report.results_voronoi4d[16]
    rc = report.results_combigeo[16]
    assert abs(rv.diameter - rc.diameter) < 1e-9
    assert abs(rv.normalized - rc.normalized) < 1e-6
    assert abs(rv.normalized - math.sqrt(0.5)) < 1e-9  # точное значение, не сентинел

    # transition обоих бэкендов — в базисе пользователя: обе матрицы задают
    # подрешётки с равным det и целыми взаимными координатами
    import numpy as np

    for r in (rv, rc):
        t = np.array(r.transition, dtype=float)
        assert abs(abs(np.linalg.det(t)) - 16.0) < 1e-9
        sb = np.array(r.sub_basis, dtype=float)
        coeffs = sb @ np.linalg.inv(t @ np.array(D4, dtype=float))
        assert np.allclose(coeffs, np.rint(coeffs), atol=1e-8)
        assert abs(abs(np.linalg.det(coeffs)) - 1.0) < 1e-8


def test_compare_flagship_d4_49_combigeo_pin():
    """Флагманский пин в python: combigeo D4/49 -> d = sqrt(7/6) (под секунду)."""
    import math

    avail = chromatic.available_backends()
    if "combigeo" not in avail:
        pytest.skip("нужен combigeo")
    res = chromatic.get_backend("combigeo").find_optimal(D4, 49)
    assert res.feasible
    assert abs(res.normalized - math.sqrt(7.0 / 6.0)) < 1e-9
    assert res.examined == 140050


def test_compare_report_structure():
    _need_both()
    report = chromatic.compare_backends(D4, [8])
    assert set(report.results_combigeo) == {8}
    assert set(report.results_voronoi4d) == {8}
    assert isinstance(report.agree, bool)


def test_at_least_one_backend_available():
    """Набор не должен «зеленеть», когда тестировать нечего (аудит, M2)."""
    assert chromatic.available_backends(), (
        "не установлен ни один бэкенд (voronoi4d/combigeo) — тесты фасада "
        "не проверяют ничего; установите хотя бы один"
    )
