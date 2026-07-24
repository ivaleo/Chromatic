"""chromatic — единый фасад над двумя бэкендами поиска периодических раскрасок R^n.

Над-проект объединяет два независимых движка:
- ``voronoi4d`` — эталонная python-реализация (размерность 4);
- ``combigeo`` — быстрое C++ ядро через pybind11 (размерности 2…6).

Бэкенд выбирается ЯВНО по имени — без автоопределения:

    >>> import chromatic
    >>> cg = chromatic.get_backend("combigeo")
    >>> cell = cg.build_cell([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]])
    >>> cell.f_vector            # 24-ячейка решётки D4
    [24, 96, 96, 24]
    >>> res = cg.find_optimal([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], 49)
    >>> res.feasible             # d = D/diam >= 1 — раскраска пригодна
    True

Кросс-валидация двух бэкендов (только размерность 4):

    >>> report = chromatic.compare_backends(basis, range(2, 17))
    >>> report.agree
    True

Терминология едина: index == det(M) == число цветов k; diameter == diam(V0);
min_distance == сырое D; normalized == d = D/diam (пригодно при d >= 1);
transition == матрица перехода M (HNF); witness == вектор, дающий минимум.
"""

from __future__ import annotations

from .backend import Backend, available_backends, get_backend, register_backend
from .compare import ComparisonReport, Discrepancy, compare_backends
from .model import Cell, Facet, OptimalResult, as_matrix

__version__ = "1.1.0"

__all__ = [
    "Backend",
    "Cell",
    "ComparisonReport",
    "Discrepancy",
    "Facet",
    "OptimalResult",
    "as_matrix",
    "available_backends",
    "compare_backends",
    "get_backend",
    "register_backend",
]
