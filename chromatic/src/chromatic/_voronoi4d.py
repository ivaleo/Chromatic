"""Бэкенд поверх эталонного python-пакета voronoi4d (только размерность 4).

voronoi4d считает расстояния уже НОРМИРОВАННЫМИ (dist_to_s возвращает
d = 2·dist/max_len = D/diam), поэтому здесь восстанавливается «сырое» D = d·diam
для единой модели фасада. С версии 1.1.0 кандидаты перечисляются точно
(граница достаточности |v| < D + diam), расстояния фасада считаются без
раннего выхода (early_stop=0 — точные значения), а `transition` выдаётся
в координатах базиса ПОЛЬЗОВАТЕЛЯ; появился и `sub_basis`.
"""

from __future__ import annotations

import importlib.util
import tempfile
from typing import List, Sequence

from ._hnf import transition_in_user_basis
from .backend import Backend, register_backend
from .model import Cell, Facet, OptimalResult, as_matrix


@register_backend
class Voronoi4dBackend(Backend):
    name = "voronoi4d"
    supported_dims = (4,)

    @staticmethod
    def available() -> bool:
        return (
            importlib.util.find_spec("voronoi4d") is not None
            and importlib.util.find_spec("numpy") is not None
        )

    # --- вспомогательное -------------------------------------------------------

    def _reduced(self, basis: Sequence[Sequence[float]]):
        import numpy as np
        from voronoi4d import lll_reduce

        self.check_dim(len(basis))
        return np.asarray(lll_reduce(np.asarray(basis, dtype=float)), dtype=float)

    def _build(self, basis: Sequence[Sequence[float]]):
        from voronoi4d import VoronoiPolyhedra

        reduced = self._reduced(basis)
        vor = VoronoiPolyhedra(reduced)
        vor.build(verbose=False)
        return vor, reduced

    # --- интерфейс Backend -----------------------------------------------------

    def build_cell(self, basis: Sequence[Sequence[float]]) -> Cell:
        vor, _ = self._build(basis)
        facets = [Facet(list(map(float, p.normal)), float(p.bias)) for p in vor.polyhedrons]
        return Cell(
            dim=4,
            vertices=as_matrix(vor.central),
            diameter=float(vor.max_len),
            facets=facets,
            f_vector=None,  # voronoi4d не вычисляет f-вектор
            backend=self.name,
            handle=vor,
        )

    def cell_distance(self, point: Sequence[float], cell: Cell) -> float:
        import numpy as np
        from voronoi4d import dist_to_s

        if cell.backend != self.name or cell.handle is None:
            raise ValueError("cell_distance: ячейка построена другим бэкендом")
        # dist_to_s возвращает d = 2·dist/max_len; сырое расстояние = d·diam/2;
        # early_stop=0 — точное значение (внутри ячейки — ровно 0.0)
        normalized = dist_to_s(cell.handle, np.asarray(point, dtype=float), cell.diameter,
                               early_stop=0.0)
        return normalized * cell.diameter / 2.0

    def min_color_distance(self, basis: Sequence[Sequence[float]],
                           sub_basis: Sequence[Sequence[float]]) -> float:
        import numpy as np
        from voronoi4d import dist_to_s, lattice_points_within, lll_reduce, shortest_vector

        vor, _ = self._build(basis)
        sub_lll = np.asarray(lll_reduce(np.asarray(sub_basis, dtype=float)), dtype=float)

        # точный минимум: старт с кратчайшего вектора, затем полный перебор
        # в границе достаточности |v| < D_текущ + diam (D(v) >= |v| - diam)
        v_min = shortest_vector(sub_lll)
        best = dist_to_s(vor, 0.5 * np.asarray(v_min, dtype=float), vor.max_len,
                         early_stop=0.0)
        for c in sorted(lattice_points_within(sub_lll, (best + 1.0) * vor.max_len),
                        key=lambda v: float(v @ v)):
            if float(np.linalg.norm(c)) - vor.max_len >= best * vor.max_len:
                break
            best = min(best, dist_to_s(vor, 0.5 * np.asarray(c, dtype=float), vor.max_len,
                                       early_stop=0.0))
        return best * vor.max_len  # сырое D = d·diam

    def find_optimal(self, basis: Sequence[Sequence[float]], index: int) -> OptimalResult:
        import numpy as np
        from voronoi4d import find_optimal as _find_optimal

        if index < 1:
            raise ValueError("find_optimal: index (число цветов) должен быть >= 1")

        vor, reduced = self._build(basis)
        # threshold=0 — берём лучшую подрешётку независимо от пригодности;
        # вывод в файл подавляем во временный путь, прогресс отключаем
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            det_dist, det_center, det_mat = _find_optimal(
                range(index, index + 1), None, reduced, vor, vor.max_len,
                threshold=0.0, output_file=tmp.name, verbose=False,
            )

        diam = float(vor.max_len)
        if index not in det_dist:
            # ни одной подходящей подрешётки (все слишком плотные)
            return OptimalResult(
                num_colors=index, diameter=diam, min_distance=0.0, normalized=0.0,
                backend=self.name,
            )
        normalized = float(det_dist[index])
        # H — HNF во внутреннем LLL-кадре; наружу выдаём базис подрешётки в
        # объемлющих координатах и transition в базисе ПОЛЬЗОВАТЕЛЯ
        sub_ambient = np.asarray(det_mat[index], dtype=float) @ reduced
        return OptimalResult(
            num_colors=index,
            diameter=diam,
            min_distance=normalized * diam,
            normalized=normalized,
            transition=transition_in_user_basis(sub_ambient, basis),
            sub_basis=as_matrix(sub_ambient),
            witness=[float(x) for x in np.asarray(det_center[index])],
            examined=None,  # voronoi4d не сообщает число обсчитанных
            backend=self.name,
        )

    def lll_reduce(self, basis: Sequence[Sequence[float]],
                   delta: float = 0.75) -> List[List[float]]:
        import numpy as np
        from voronoi4d import lll_reduce as _lll

        return as_matrix(np.asarray(_lll(np.asarray(basis, dtype=float), delta), dtype=float))
