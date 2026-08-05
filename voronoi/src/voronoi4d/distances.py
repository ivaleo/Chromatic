"""Расчёт расстояний от точек до центрального многогранника Вороного.

Основная функция — dist_to_s(): расстояние от точки s до центрального
многогранника через каскад проекций (3D-грань → 2D-грань → ребро → вершина).
"""

import math
import warnings

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


def dist_to_s(vor4, s, max_len, early_stop=1.0, check=True):
    """Нормированное расстояние от точки s до центрального многогранника V0.

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
