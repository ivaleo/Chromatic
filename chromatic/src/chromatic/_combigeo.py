"""Бэкенд поверх C++ модуля combigeo (pybind11), размерности 2…6.

combigeo возвращает «сырые» расстояния D напрямую, нормировку d = D/diam
фасад вычисляет единообразно. Размерности 5…6 поддерживаются кодом, но
надёжно проверены эталонами лишь 2…4 (см. README combigeo).
"""

from __future__ import annotations

import importlib.util
from typing import Iterable, List, Sequence

from .backend import Backend, register_backend
from .model import Cell, Facet, OptimalResult, as_matrix


@register_backend
class CombigeoBackend(Backend):
    name = "combigeo"
    supported_dims = (2, 3, 4, 5, 6)

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("combigeo") is not None

    def build_cell(self, basis: Sequence[Sequence[float]]) -> Cell:
        import combigeo

        self.check_dim(len(basis))
        raw = combigeo.voronoi_cell(_to_lists(basis))
        facets = [Facet(list(map(float, h.normal)), float(h.offset)) for h in raw.facets]
        return Cell(
            dim=int(raw.dim),
            vertices=as_matrix(raw.vertices),
            diameter=float(raw.diameter),
            facets=facets,
            f_vector=[int(x) for x in raw.f_vector],
            backend=self.name,
            handle=raw,
        )

    def cell_distance(self, point: Sequence[float], cell: Cell) -> float:
        import combigeo

        if cell.backend != self.name or cell.handle is None:
            raise ValueError("cell_distance: ячейка построена другим бэкендом")
        return float(combigeo.distance_to_cell([float(x) for x in point], cell.handle))

    def min_color_distance(self, basis: Sequence[Sequence[float]],
                           sub_basis: Sequence[Sequence[float]]) -> float:
        import combigeo

        return float(combigeo.min_color_distance(_to_lists(basis), _to_lists(sub_basis)))

    def find_optimal(self, basis: Sequence[Sequence[float]], index: int) -> OptimalResult:
        import combigeo

        self.check_dim(len(basis))
        r = combigeo.find_optimal(_to_lists(basis), int(index))
        return _to_result(r, self.name)

    def find_optimal_range(self, basis: Sequence[Sequence[float]],
                           indices: Iterable[int]) -> "dict[int, OptimalResult]":
        import combigeo

        idx = sorted({int(k) for k in indices})
        if not idx:
            return {}
        self.check_dim(len(basis))
        # для непрерывного диапазона используем нативную функцию
        if idx == list(range(idx[0], idx[-1] + 1)):
            native = combigeo.find_optimal_range(_to_lists(basis), idx[0], idx[-1])
            return {int(k): _to_result(v, self.name) for k, v in native.items()}
        return {k: self.find_optimal(basis, k) for k in idx}

    def lll_reduce(self, basis: Sequence[Sequence[float]],
                   delta: float = 0.75) -> List[List[float]]:
        import combigeo

        return as_matrix(combigeo.lll_reduce(_to_lists(basis), delta))

    def shortest_vector(self, basis: Sequence[Sequence[float]]) -> List[float]:
        import combigeo

        return [float(x) for x in combigeo.shortest_vector(_to_lists(basis))]

    # --- дополнительно: перебор подрешёток (есть только у combigeo) ------------

    def count_sublattices(self, dim: int, index: int) -> int:
        import combigeo

        return int(combigeo.count_sublattices(dim, index))


def _to_lists(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(x) for x in row] for row in matrix]


def _to_result(r, backend: str) -> OptimalResult:
    return OptimalResult(
        num_colors=int(r.index),
        diameter=float(r.diameter),
        min_distance=float(r.best.min_distance),
        normalized=float(r.normalized),
        transition=[[int(x) for x in row] for row in r.best.transition] if r.best.transition else None,
        sub_basis=as_matrix(r.best.sub_basis) if r.best.sub_basis else None,
        witness=[float(x) for x in r.best.witness] if r.best.witness else None,
        examined=int(r.examined),
        backend=backend,
    )
