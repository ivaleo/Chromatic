"""Расчёт расстояний от точек до центрального многогранника Вороного.

Основная функция — dist_to_s(): расстояние от точки s до центрального
многогранника через каскад проекций (3D-грань → 2D-грань → ребро → вершина).
"""

import math
import warnings

import numpy as np

TOL_SIMPLEX = 1e-9  # допуск поиска симплекса в триангуляции

# --------------------------------------------------------------------------------


def _dist(a, b):
    """Евклидово расстояние между 4-мерными точками.

    scipy.spatial.distance.euclidean на векторах такой длины тратит на
    валидацию входа больше, чем на само вычисление: в профиле dist_to_s это
    58% времени при 263 тыс. вызовов на 200 точек.
    """
    delta = a - b
    return math.sqrt(delta @ delta)


def check_dist(dist1, dist2):
    """Сверяет два квадрата расстояний (контроль по теореме Пифагора).

    При расхождении выдаёт предупреждение через модуль warnings (не прерывает счёт).
    """
    if not math.isclose(dist1, dist2, abs_tol=1e-9):
        warnings.warn(
            f"расхождение квадратов расстояний (теорема Пифагора): {dist1 - dist2}",
            stacklevel=2,
        )


# --------------------------------------------------------------------------------


def dist_to_s_cascade(vor4, s, max_len, early_stop=1.0, check=True):
    """Каскадный эталон dist_to_s: полный скан граней на чистом python.

    Оставлен как независимая реализация для сверки с векторизованной версией
    (tests/test_distances_vectorized.py). В API пакета не экспортируется.

    s — середина отрезка между началом координат и центром соседней области
    Вороного подрешётки. По лемме Иванова минимальное расстояние D между
    соседними областями реализуется в точках, симметричных относительно s;
    D = 2*dist(s, V0). Возвращается нормированное запрещённое расстояние
    d = D / diam(V0) = dist * 2 / max_len.

    Для точки внутри V0 (включая границу) возвращается 0.0.

    Проекция ищется каскадом по ВСЕМ 3-мерным граням (полный скан — без
    эвристического отсечения по ближайшим вершинам): 3D-грань → 2D-грань →
    ребро → вершина.

    Ранний выход: как только текущий минимум в нормировке опускается ниже
    early_stop, функция сразу возвращает ТЕКУЩИЙ минимум (верхнюю оценку,
    не обязательно глобальный минимум). Для отбраковки подрешётки с порогом
    threshold этого достаточно; early_stop=0 отключает ранний выход и
    гарантирует точное значение.

    :param vor4: объект VoronoiPolyhedra после build().
    :param s: координаты точки (np.array).
    :param max_len: диаметр центрального многогранника diam(V0).
    :param early_stop: порог раннего выхода в нормировке d (по умолчанию 1.0).
    :param check: сверять расстояния по теореме Пифагора (диагностика; выключение
                  экономит около 1% времени).
    :return: нормированное расстояние d (точное, если >= early_stop).
    """
    polyhedrons = vor4.polyhedrons

    # точка внутри многогранника (все гиперграни: normal наружу от центра ячейки)
    if all(pol.normal @ (s - pol.center) <= TOL_SIMPLEX for pol in polyhedrons):
        return 0.0

    min_dist_to_pol = float("inf")  # минимальное расстояние до центрального многогранника

    # полный скан всех 3-мерных граней
    for i in range(len(polyhedrons)):
        # проекция на 3-мерную грань
        d0 = polyhedrons[i].normal @ (s - polyhedrons[i].center)
        coord0 = s - d0 * polyhedrons[i].normal
        simplex = vor4.delaunay.find_simplex(coord0, tol=TOL_SIMPLEX)

        d0_squared = d0 * d0

        if simplex != -1:  # проекция принадлежит центральному многограннику
            min_dist_to_pol = min(min_dist_to_pol, abs(d0))
            continue

        for face2d in polyhedrons[i].faces:
            # проекция на 2-мерную грань
            d1 = face2d.normal @ (coord0 - face2d.center)
            coord1 = coord0 - d1 * face2d.normal
            simplex = vor4.delaunay.find_simplex(coord1, tol=TOL_SIMPLEX)

            d1_squared = d1 * d1

            if simplex != -1:  # проекция принадлежит центральному многограннику
                dist = _dist(s, coord1)

                if check:
                    check_dist(dist * dist, d0_squared + d1_squared)

                min_dist_to_pol = min(min_dist_to_pol, dist)
                continue

            for edge in face2d.edges:
                # проекция на ребро
                d2 = edge.normal @ (coord1 - edge.center)
                coord2 = coord1 - d2 * edge.normal
                simplex = vor4.delaunay.find_simplex(coord2, tol=TOL_SIMPLEX)

                d2_squared = d2 * d2

                if simplex != -1:  # проекция принадлежит центральному многограннику
                    dist = _dist(s, coord2)

                    if check:
                        check_dist(dist * dist, d0_squared + d1_squared + d2_squared)

                    min_dist_to_pol = min(min_dist_to_pol, dist)
                else:
                    # проекция вне ребра — берём ближайшую вершину ребра
                    d3 = _dist(coord2, edge.vertex1)
                    d4 = _dist(coord2, edge.vertex2)

                    if d3 < d4:
                        dist = _dist(s, edge.vertex1)
                        d34_squared = d3 * d3
                    else:
                        dist = _dist(s, edge.vertex2)
                        d34_squared = d4 * d4

                    if check:
                        check_dist(dist * dist, d0_squared + d1_squared + d2_squared + d34_squared)

                    min_dist_to_pol = min(min_dist_to_pol, dist)

        # если нормированное расстояние уже ниже порога, дальше можно не считать
        if early_stop > 0.0 and min_dist_to_pol * 2 / max_len < early_stop:
            return float(min_dist_to_pol * 2 / max_len)

    return float(min_dist_to_pol * 2 / max_len)


# --------------------------------------------------------------------------------


def dist_to_s(vor4, s, max_len, early_stop=1.0, check=True):
    """Нормированное расстояние от точки s до центрального многогранника V0.

    Векторизованный вариант каскада проекций: все грани каждого уровня
    обрабатываются одним numpy-выражением, принадлежность проверяется одним
    вызовом Delaunay.find_simplex на массив точек. Эталон — dist_to_s_cascade,
    сверка в tests/test_distances_vectorized.py.

    Каскад обрывал спуск, когда проекция на грань попадала внутрь ячейки.
    Пропущенные при этом кандидаты — точки той же грани или её границы, то есть
    не ближе найденной, поэтому вычисление всех уровней даёт тот же минимум.

    Отличие от каскада: значение всегда точное. Каскад при early_stop возвращал
    верхнюю оценку, а точное значение её не хуже, поэтому решения find_optimal
    не меняются, и сам параметр здесь не нужен.

    Для точки внутри V0 (включая границу) возвращается 0.0.

    :param vor4: объект VoronoiPolyhedra после build().
    :param s: координаты точки (np.array).
    :param max_len: диаметр центрального многогранника diam(V0).
    :param early_stop: не используется (принимается ради совместимости вызовов).
    :param check: не используется: прямые расстояния сверять по Пифагору нечего.
    :return: нормированное расстояние d.
    """
    del early_stop, check
    s = np.asarray(s, dtype=float)

    # точка внутри многогранника (все гиперграни: normal наружу от центра ячейки)
    offset = np.einsum("ij,ij->i", vor4.face3_normal, vor4.face3_center)
    d0 = vor4.face3_normal @ s - offset
    if np.all(d0 <= TOL_SIMPLEX):
        return 0.0

    # уровень 3: проекция на каждую гипергрань
    coord0 = s - d0[:, None] * vor4.face3_normal
    hit0 = vor4.delaunay.find_simplex(coord0, tol=TOL_SIMPLEX) != -1
    best = float(np.abs(d0[hit0]).min()) if hit0.any() else float("inf")

    # уровень 2: проекция на каждую 2-мерную грань
    base1 = coord0[vor4.face2_parent]
    d1 = np.einsum("ij,ij->i", vor4.face2_normal, base1 - vor4.face2_center)
    coord1 = base1 - d1[:, None] * vor4.face2_normal
    hit1 = vor4.delaunay.find_simplex(coord1, tol=TOL_SIMPLEX) != -1
    if hit1.any():
        best = min(best, float(np.linalg.norm(coord1[hit1] - s, axis=1).min()))

    # уровень 1: проекция на каждое ребро, иначе — ближайшая его вершина
    base2 = coord1[vor4.edge_parent]
    d2 = np.einsum("ij,ij->i", vor4.edge_normal, base2 - vor4.edge_center)
    coord2 = base2 - d2[:, None] * vor4.edge_normal
    hit2 = vor4.delaunay.find_simplex(coord2, tol=TOL_SIMPLEX) != -1
    if hit2.any():
        best = min(best, float(np.linalg.norm(coord2[hit2] - s, axis=1).min()))

    outside = ~hit2
    if outside.any():
        v1, v2 = vor4.edge_vertex1[outside], vor4.edge_vertex2[outside]
        nearer_v1 = (np.linalg.norm(coord2[outside] - v1, axis=1)
                     < np.linalg.norm(coord2[outside] - v2, axis=1))
        nearest = np.where(nearer_v1[:, None], v1, v2)
        best = min(best, float(np.linalg.norm(nearest - s, axis=1).min()))

    return float(best * 2 / max_len)
