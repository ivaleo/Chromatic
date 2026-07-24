"""Единая модель данных фасада chromatic.

Оба бэкенда (voronoi4d, combigeo) приводятся к этим структурам, чтобы
вызывающий код не зависел от деталей конкретной реализации.

Соглашение о расстояниях (единое для фасада):
- ``min_distance`` — «сырое» расстояние D между одноцветными ячейками
  (D = 2·dist(v/2, V0));
- ``normalized`` — нормированное запрещённое расстояние d = D / diam(V0);
  раскраска пригодна при d ≥ 1 (``feasible``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


@dataclass
class Facet:
    """Полупространство ячейки Вороного: x·normal ≤ offset (normal единичный)."""

    normal: List[float]
    offset: float


@dataclass
class Cell:
    """Ячейка Вороного центральной точки решётки (единое представление).

    :param dim: размерность пространства.
    :param vertices: вершины ячейки.
    :param diameter: диаметр diam(V0) = 2·max|вершина|.
    :param facets: релевантные фасеты (могут быть пусты, если бэкенд их не даёт).
    :param f_vector: f-вектор [f0, f1, …] (None, если бэкенд не вычисляет).
    :param backend: имя бэкенда, построившего ячейку.
    :param handle: «сырой» объект бэкенда (VoronoiPolyhedra или combigeo.VoronoiCell)
                   для внутреннего использования; не часть стабильного API.
    """

    dim: int
    vertices: List[List[float]]
    diameter: float
    facets: List[Facet] = field(default_factory=list)
    f_vector: Optional[List[int]] = None
    backend: str = ""
    handle: Any = None

    def __repr__(self) -> str:
        f = self.f_vector if self.f_vector is not None else "—"
        return (f"Cell(backend={self.backend!r}, dim={self.dim}, "
                f"vertices={len(self.vertices)}, f={f}, diam={self.diameter:.6g})")


@dataclass
class OptimalResult:
    """Результат поиска оптимальной подрешётки для заданного числа цветов.

    :param num_colors: число цветов k = |det(M)| = индекс подрешётки.
    :param diameter: диаметр центральной ячейки diam(V0).
    :param min_distance: сырое расстояние D между одноцветными ячейками.
    :param normalized: нормированное запрещённое расстояние d = D / diam(V0).
    :param transition: матрица перехода M (HNF) — строки целых коэффициентов.
    :param sub_basis: базис найденной подрешётки (если бэкенд его даёт).
    :param witness: вектор v, на котором достигается минимум (если есть).
    :param examined: сколько подрешёток обсчитано (None, если бэкенд не сообщает).
    :param backend: имя бэкенда.
    """

    num_colors: int
    diameter: float
    min_distance: float
    normalized: float
    transition: Optional[List[List[int]]] = None
    sub_basis: Optional[List[List[float]]] = None
    witness: Optional[List[float]] = None
    examined: Optional[int] = None
    backend: str = ""

    @property
    def feasible(self) -> bool:
        """Пригодна ли раскраска: нормированное расстояние d ≥ 1."""
        return self.normalized >= 1.0

    def __repr__(self) -> str:
        mark = " (пригодна)" if self.feasible else ""
        return (f"OptimalResult(backend={self.backend!r}, k={self.num_colors}, "
                f"d={self.normalized:.6g}, D={self.min_distance:.6g}{mark})")


def as_matrix(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    """Приводит вложенную последовательность (в т.ч. numpy) к спискам float."""
    return [[float(x) for x in row] for row in rows]
