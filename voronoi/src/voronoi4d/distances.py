"""Расчёт расстояний от точек до центрального многогранника Вороного.

Основная функция — dist_to_s(): расстояние от точки s до центрального
многогранника через каскад проекций (3D-грань → 2D-грань → ребро → вершина).
"""

import math
import warnings

from scipy.spatial import distance

CHECK_DIST = True  # проверка расстояний по теореме Пифагора (можно отключить: установить False)
TOL_SIMPLEX = 1e-9  # допуск поиска симплекса в триангуляции

# --------------------------------------------------------------------------------


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


def dist_to_s(vor4, s, max_len, early_stop=1.0):
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
    :return: нормированное расстояние d (точное, если >= early_stop).
    """
    polyhedrons = vor4.polyhedrons

    # точка внутри многогранника (все гиперграни: normal наружу от центра ячейки)
    if all(pol.normal @ (s - pol.center) <= TOL_SIMPLEX for pol in polyhedrons):
        return 0.0

    min_dist_to_pol = float("inf")  # минимальное расстояние до центрального многогранника

    def update_min_distance():
        nonlocal min_dist_to_pol

        if dist < min_dist_to_pol:
            min_dist_to_pol = dist

    # полный скан всех 3-мерных граней
    for i in range(len(polyhedrons)):
        # проекция на 3-мерную грань
        d0 = polyhedrons[i].normal @ (s - polyhedrons[i].center)
        coord0 = s - d0 * polyhedrons[i].normal
        simplex = vor4.delaunay.find_simplex(coord0, tol=TOL_SIMPLEX)

        d0_squared = d0 * d0

        if simplex != -1:  # проекция принадлежит центральному многограннику
            dist = abs(d0)
            update_min_distance()
            continue

        for face2d in polyhedrons[i].faces:
            # проекция на 2-мерную грань
            d1 = face2d.normal @ (coord0 - face2d.center)
            coord1 = coord0 - d1 * face2d.normal
            simplex = vor4.delaunay.find_simplex(coord1, tol=TOL_SIMPLEX)

            d1_squared = d1 * d1

            if simplex != -1:  # проекция принадлежит центральному многограннику
                dist = distance.euclidean(s, coord1)

                if CHECK_DIST:
                    check_dist(dist * dist, d0_squared + d1_squared)

                update_min_distance()
                continue

            for edge in face2d.edges:
                # проекция на ребро
                d2 = edge.normal @ (coord1 - edge.center)
                coord2 = coord1 - d2 * edge.normal
                simplex = vor4.delaunay.find_simplex(coord2, tol=TOL_SIMPLEX)

                d2_squared = d2 * d2

                if simplex != -1:  # проекция принадлежит центральному многограннику
                    dist = distance.euclidean(s, coord2)

                    if CHECK_DIST:
                        check_dist(dist * dist, d0_squared + d1_squared + d2_squared)

                    update_min_distance()
                else:
                    # проекция вне ребра — берём ближайшую вершину ребра
                    d3 = distance.euclidean(coord2, edge.vertex1)
                    d4 = distance.euclidean(coord2, edge.vertex2)

                    if d3 < d4:
                        dist = distance.euclidean(s, edge.vertex1)
                        d34_squared = d3 * d3
                    else:
                        dist = distance.euclidean(s, edge.vertex2)
                        d34_squared = d4 * d4

                    if CHECK_DIST:
                        check_dist(dist * dist, d0_squared + d1_squared + d2_squared + d34_squared)

                    update_min_distance()

        # если нормированное расстояние уже ниже порога, дальше можно не считать
        if early_stop > 0.0 and min_dist_to_pol * 2 / max_len < early_stop:
            return min_dist_to_pol * 2 / max_len

    return min_dist_to_pol * 2 / max_len
