"""Геометрия разбиения Вороного 4-мерной решётки.

Классы:
- Edge2D  — ребро 2-мерной грани;
- Face2D  — 2-мерная грань 3-мерной гиперграни;
- Face3D  — 3-мерная гипергрань центрального многогранника;
- VoronoiPolyhedra — разбиение Вороного решётки с построением
  центрального многогранника и всей его иерархии граней.
"""

import numpy as np
from itertools import product, combinations
from scipy.spatial import Voronoi, Delaunay, ConvexHull, distance

# допуски для численной стабильности
TOL_DEGENERATE = 1e-9  # порог вырожденности (нулевые векторы, SVD)
TOL_DIRECTION = 1e-7  # допуск при проверке направления нормали

# --------------------------------------------------------------------------------


class Edge2D:
    """Ребро 2-мерной грани."""

    def __init__(self, vertex1, vertex2, face_center, normal_2d, normal_3d):
        """
        :param vertex1: координаты первой вершины ребра.
        :param vertex2: координаты второй вершины ребра.
        :param face_center: координаты центра 2-мерной грани.
        :param normal_2d: нормаль к 2-мерной грани.
        :param normal_3d: нормаль к 3-мерной гиперграни.
        """
        self.vertex1 = np.array(vertex1)
        self.vertex2 = np.array(vertex2)
        self.face_center = np.array(face_center)
        self.normal_2d = normal_2d
        self.normal_3d = normal_3d

        # середина ребра
        self.center = (self.vertex1 + self.vertex2) / 2

        # нормаль к ребру: ортогональна ребру, нормали 2D-грани и нормали 3D-грани
        normal_candidate = self._find_edge_normal()
        norm_val = np.linalg.norm(normal_candidate)

        if norm_val < TOL_DEGENERATE:
            # вырожденный случай — нулевая нормаль
            self.normal = np.zeros_like(normal_candidate)
            self.bias = 0.0
        else:
            oriented_normal = normal_candidate / norm_val

            # нормаль должна быть направлена "наружу" от центра 2-мерной грани
            vector_to_face_center = self.face_center - self.center
            if np.dot(oriented_normal, vector_to_face_center) > TOL_DIRECTION:
                self.normal = -oriented_normal
            else:
                self.normal = oriented_normal

            self.bias = self.normal @ self.vertex1

    def _find_edge_normal(self):
        """Ищет вектор, ортогональный ребру и обеим нормалям (через SVD)."""
        edge = self.vertex2 - self.vertex1

        # проверка, не вырождено ли само ребро
        if np.linalg.norm(edge) < TOL_DEGENERATE:
            return np.zeros_like(self.vertex1)

        edge = edge / np.linalg.norm(edge)

        # матрица условий ортогональности
        A = np.column_stack((edge, self.normal_2d, self.normal_3d))

        try:
            U, S, Vt = np.linalg.svd(A)
        except np.linalg.LinAlgError:
            return np.zeros_like(self.vertex1)

        if S[-1] < TOL_DIRECTION:  # столбцы линейно зависимы
            return np.zeros_like(self.vertex1)

        # нормаль к ребру — последний столбец U
        return U[:, -1]

    def __repr__(self):
        return (
            f"Edge2D(vertex1={self.vertex1}, vertex2={self.vertex2}, "
            f"normal={self.normal}, midpoint={self.center})"
        )


# --------------------------------------------------------------------------------


class Face2D:
    """2-мерная грань 3-мерной гиперграни."""

    def __init__(self, vertices, polyhedron_center, vor4, parent_normal):
        """
        :param vertices: координаты вершин грани.
        :param polyhedron_center: центр родительской 3-мерной грани.
        :param vor4: объект VoronoiPolyhedra (для поиска рёбер).
        :param parent_normal: нормаль к родительской 3-мерной грани.
        """
        self.vertices = vertices
        self.vor4 = vor4
        self.parent_normal = parent_normal
        self.parent_center = np.array(polyhedron_center)

        self.center = np.mean(self.vertices, axis=0)
        self._calculate_normal()
        self._calculate_edges()
        self.bias = self.normal @ vertices[0]

    def _calculate_normal(self):
        """Нормаль к грани: ортогональна плоскости грани и нормали родителя (SVD).

        Пары вершин перебираются до невырожденной конфигурации: порядок вершин
        произволен (приходит из set-пересечений), и первая тройка может оказаться
        почти коллинеарной — контроль по младшему сингулярному числу
        (столбец ортогональной матрицы U всегда единичен, проверять его норму
        бессмысленно — прежний «guard» был мёртвым кодом).
        """
        n_vert = len(self.vertices)
        for j in range(1, n_vert):
            for k in range(j + 1, n_vert):
                v1 = self.vertices[0] - self.vertices[j]
                v2 = self.vertices[0] - self.vertices[k]
                A = np.column_stack((v1, v2, self.parent_normal))
                U, S, _ = np.linalg.svd(A)
                if S[-1] < TOL_DIRECTION:  # тройка (почти) вырождена — берём другую
                    continue

                oriented_normal = U[:, -1]  # единичный вектор, ортогональный столбцам A

                # нормаль должна указывать "наружу" относительно центра родителя
                vector_to_parent_center = self.parent_center - self.center
                if np.dot(oriented_normal, vector_to_parent_center) > 0:
                    self.normal = -oriented_normal
                else:
                    self.normal = oriented_normal
                return

        # все тройки вырождены — дефектная грань
        self.normal = np.zeros(len(self.vertices[0]))
        self.bias = 0.0

    def _find_edge_coords(self):
        """Находит рёбра грани (пары координат вершин) по списку рёбер vor4."""
        vertex_indices = []  # индексы вершин грани в vor4.vertices
        edge_indices = []  # рёбра грани в индексах

        for vert in self.vertices:
            ind = np.argwhere((self.vor4.vertices == vert).all(axis=1))[0][0]
            vertex_indices.append(ind)

        for edge in self.vor4.list_edges:
            if edge[0] in vertex_indices and edge[1] in vertex_indices:
                edge_indices.append(edge)

        return self.vor4.vertices[edge_indices]

    def _calculate_edges(self):
        """Создаёт объекты Edge2D для всех рёбер грани."""
        self.edges = [
            Edge2D(vertex1, vertex2, self.center, self.normal, self.parent_normal)
            for vertex1, vertex2 in self._find_edge_coords()
        ]

    def __repr__(self):
        return f"Face2D(vertices={self.vertices}, center={self.center}, normal={self.normal})"


# --------------------------------------------------------------------------------


class Face3D:
    """3-мерная гипергрань центрального многогранника в 4-мерном пространстве."""

    def __init__(self, vertices, faces_list, normal, vor4):
        """
        :param vertices: координаты вершин гиперграни.
        :param faces_list: списки вершин 2-мерных граней этой гиперграни.
        :param normal: вектор нормали к гиперграни.
        :param vor4: объект VoronoiPolyhedra.
        """
        self.vor4 = vor4
        self.vertices = vertices
        self.normal = normal
        self.faces_list = faces_list

        self.center = np.mean(self.vertices, axis=0)
        self.faces = [
            Face2D(face_vertices, self.center, self.vor4, self.normal)
            for face_vertices in self.faces_list
        ]
        self.bias = normal @ vertices[0]

    def __repr__(self):
        return (
            f"Face3D(vertices={self.vertices}, normal={self.normal}, "
            f"center={self.center}, faces={self.faces})"
        )


# --------------------------------------------------------------------------------


class VoronoiPolyhedra(Voronoi):
    """Разбиение Вороного 4-мерной решётки.

    После вызова build() содержит центральный многогранник (central),
    его 3-мерные грани (polyhedrons), триангуляцию Делоне (delaunay)
    и вспомогательные структуры для расчёта расстояний.
    """

    # диапазон коэффициентов для генерации центров: СИММЕТРИЧНЫЙ (до 1.1.0 был
    # несимметричный range(-3, 3) — облако центров не было центрально-симметричным);
    # меньше +-3 нельзя — не хватает точек для корректного центрального региона
    COEFF_RANGE = range(-3, 4)

    def __init__(self, grid):
        """
        :param grid: базис 4-мерной решётки — матрица 4x4 (numpy.ndarray),
                     строки которой задают векторы базиса. Центры многогранников
                     генерируются как целочисленные комбинации строк базиса
                     с коэффициентами из COEFF_RANGE.
        """
        self.grid = grid

        # координаты центров многогранников
        self.coords4 = [(self.grid.T).dot(var).tolist() for var in product(self.COEFF_RANGE, repeat=4)]

        # строим диаграмму Вороного
        super().__init__(self.coords4)

    def find_central(self, verbose=True):
        """Находит центральный многогранник.

        Для каждого конечного региона считается суммарное расстояние от его вершин
        до начала координат; центральный — регион с минимальной суммой.
        """
        sum_dist_min = float("inf")
        central_index = None

        for index, region in enumerate(self.regions):
            if -1 in region or len(region) == 0:
                continue  # регион неограничен или пуст

            # суммарное расстояние от вершин региона до начала координат
            length = sum(
                distance.euclidean(self.vertices[vert], [0.0, 0.0, 0.0, 0.0]) for vert in region
            )

            if length < sum_dist_min:
                sum_dist_min = length
                central_index = index

        if central_index is None:
            raise ValueError(
                "центральный ограниченный регион не найден — расширьте COEFF_RANGE"
            )

        if verbose:
            print("суммарное расстояние =", sum_dist_min, "индекс центрального региона =", central_index)

        self.central_region_index = self.regions[central_index]  # индексы вершин центрального региона
        self.central = self.vertices[self.central_region_index]  # координаты вершин

    def find_faces_3d(self):
        """Находит 3-мерные грани центрального региона."""
        self.faces_3d = []  # 3-мерные грани в индексах вершин

        for ridge in self.ridge_vertices:
            if np.all(np.isin(ridge, self.central_region_index)):
                self.faces_3d.append(ridge)

        # в список могли попасть и грани меньшей размерности:
        # если грань полностью входит в другую, она не 3-мерная — удаляем
        to_remove = []
        for face_1 in self.faces_3d:
            for face_2 in self.faces_3d:
                if np.all(np.isin(face_1, face_2)) and len(face_1) < len(face_2):
                    to_remove.append(face_1)
                    break

        for face in to_remove:
            self.faces_3d.remove(face)

        # переводим индексы вершин 3-мерных граней в координаты
        self.edge_central_coords = [
            [self.vertices[vert] for vert in face] for face in self.faces_3d
        ]

    def find_faces_2d(self):
        """Формирует список 2-мерных граней.

        Если у двух 3-мерных граней более двух общих вершин — это 2-мерная грань.
        """
        self.faces_2d = [[] for _ in range(len(self.faces_3d))]

        for i, j in combinations(range(len(self.faces_3d)), 2):
            common_coords = list(set(self.faces_3d[i]) & set(self.faces_3d[j]))

            if len(common_coords) > 2:
                self.faces_2d[i].append(common_coords)
                self.faces_2d[j].append(common_coords)

    def find_edges(self):
        """Находит рёбра: пары общих вершин 2-мерных граней одной 3-мерной грани."""
        self.edges = [[] for _ in range(len(self.faces_2d))]

        for index, faces in enumerate(self.faces_2d):
            for i, j in combinations(range(len(faces)), 2):
                common_coords = list(set(faces[i]) & set(faces[j]))

                if len(common_coords) == 2:
                    common_coords.sort()
                    self.edges[index].append(common_coords)

        # общий список рёбер без дубликатов
        self.list_edges = []
        for face in self.edges:
            for edge in face:
                if edge not in self.list_edges:
                    self.list_edges.append(edge)

    def find_neighbors(self):
        """Составляет список центров многогранников, граничащих с центральным.

        Вектор из начала координат в соседний центр — нормаль к 3-мерной грани
        между центральным и соседним многогранниками.
        """
        self.central_point_index = self.coords4.index([0.0, 0.0, 0.0, 0.0])

        self.list_neigh_points = [[] for _ in range(len(self.faces_3d))]  # координаты соседних центров
        self.list_neigh_points_ind = [[] for _ in range(len(self.faces_3d))]  # индексы в coords4

        for pair in self.ridge_points:
            if self.central_point_index not in pair:
                continue

            # индекс соседней (не центральной) точки
            neighbor = pair[1] if pair[0] == self.central_point_index else pair[0]
            region = self.regions[self.point_region[neighbor]]

            # если регион соседа содержит все вершины какой-то 3-мерной грани центрального
            # многогранника, то сосед граничит с центральным именно по этой грани
            for i in range(len(self.faces_3d)):
                if all(item in region for item in self.faces_3d[i]):
                    self.list_neigh_points_ind[i] = neighbor
                    self.list_neigh_points[i] = self.coords4[neighbor]

    def find_2d_subfaces(self):
        """Переводит 2-мерные грани каждой 3-мерной грани из индексов в координаты."""
        self.list_faces = [
            [[self.vertices[vert] for vert in face2d] for face2d in face3d]
            for face3d in self.faces_2d
        ]

    def normalize_normals(self):
        """Нормирует векторы нормалей (направления к соседним центрам)."""
        # каждой 3-мерной грани должен соответствовать соседний центр;
        # пустой элемент означает, что сосед по грани не найден (find_neighbors)
        for i, vec in enumerate(self.list_neigh_points):
            if len(vec) == 0:
                raise ValueError(
                    f"для 3-мерной грани {i} не найден соседний центр — "
                    "построение нормали невозможно (проверьте COEFF_RANGE и разбиение)"
                )

        self.v_norm = [np.array(vec / np.linalg.norm(vec)) for vec in self.list_neigh_points]

    def map_vertices_to_faces(self):
        """Для каждой вершины центрального региона составляет список содержащих её 3-мерных граней."""
        self.vertex_to_faces = []

        for vertex in self.central_region_index:
            faces_with_vertex = [
                face for face in range(len(self.faces_3d)) if vertex in self.faces_3d[face]
            ]
            self.vertex_to_faces.append(faces_with_vertex)

    def create_polyhedrons(self):
        """Создаёт объекты Face3D для всех 3-мерных граней центрального многогранника."""
        self.polyhedrons = [
            Face3D(self.edge_central_coords[i], self.list_faces[i], self.v_norm[i], self)
            for i in range(len(self.edge_central_coords))
        ]

    def create_triangulation(self, verbose=True):
        """Строит триангуляцию Делоне центрального многогранника и проверяет её корректность."""
        self.delaunay = Delaunay(self.central)

        if not verbose:
            return

        # проверяем, что все точки из central используются
        unique_vertices = np.unique(self.delaunay.simplices)
        if len(unique_vertices) == len(self.central):
            print("Все точки используются в триангуляции.")
        else:
            print("Не все точки используются в триангуляции.")

        # сравниваем объёмы выпуклых оболочек исходных точек и точек триангуляции
        original_hull = ConvexHull(self.central)
        triangulated_points = self.central[self.delaunay.simplices.flatten()]
        triangulated_hull = ConvexHull(triangulated_points)

        is_convex_hull_correct = np.isclose(original_hull.volume, triangulated_hull.volume)
        print("Выпуклость триангуляции сохранена:", is_convex_hull_correct)

    def _find_max_len(self):
        """Диаметр многогранника: удвоенное максимальное расстояние от центра до вершины."""
        max_len = max(
            distance.euclidean(vert, np.array([0, 0, 0, 0])) for vert in self.central
        )
        return max_len * 2

    def _validate_cell(self):
        """Решающие проверки построенной ячейки (см. AUDIT-2026-07-21, C3).

        (a) Объём центрального региона обязан равняться |det(grid)| — иначе
        облако центров COEFF_RANGE не покрыло все релевантные векторы решётки
        и «ячейка» раздута (это происходит для скошенных, не LLL-приведённых
        базисов). (b) Множество вершин обязано быть центрально-симметричным.
        """
        vol = ConvexHull(self.central).volume
        det = abs(float(np.linalg.det(np.asarray(self.grid, dtype=float))))
        if not np.isclose(vol, det, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"объём построенной ячейки {vol:.12g} != |det(grid)| = {det:.12g}: "
                "облако центров не покрывает релевантные векторы решётки — "
                "примените lll_reduce к базису перед построением"
            )
        verts = np.asarray(self.central, dtype=float)
        scale = max(1.0, float(np.abs(verts).max()))
        for v in verts:
            if not np.any(np.all(np.isclose(verts, -v, atol=1e-7 * scale), axis=1)):
                raise ValueError(
                    "множество вершин ячейки не центрально-симметрично — "
                    "численный сбой построения (примените lll_reduce к базису)"
                )

    def build(self, verbose=True):
        """Полное построение всех структур разбиения.

        Бросает ValueError, если построенная ячейка не проходит решающие
        проверки (объём = |det(grid)|, центральная симметрия) — вместо тихо
        неверных результатов для неприведённых базисов.
        """
        self.find_central(verbose=verbose)
        self.find_faces_3d()
        self.find_faces_2d()
        self.find_edges()
        self.find_neighbors()

        self.max_len = self._find_max_len()  # диаметр центрального многогранника

        self.find_2d_subfaces()
        self.normalize_normals()
        self.create_polyhedrons()
        self.map_vertices_to_faces()
        self._validate_cell()
        self.create_triangulation(verbose=verbose)
