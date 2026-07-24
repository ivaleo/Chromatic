"""Сравнение бэкендов (кросс-валидация).

Оба бэкенда пересекаются в размерности 4, при этом используют НЕЗАВИСИМЫЕ
геометрические алгоритмы (combigeo — GJK по вершинам, voronoi4d — каскад
проекций по граням). Совпадение результатов — сильная взаимная проверка
корректности.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .backend import get_backend
from .model import OptimalResult


@dataclass
class Discrepancy:
    """Расхождение результатов двух бэкендов для одного числа цветов."""

    num_colors: int
    field: str
    voronoi4d: float
    combigeo: float
    diff: float


@dataclass
class ComparisonReport:
    """Итог сравнения по диапазону индексов."""

    results_voronoi4d: "dict[int, OptimalResult]"
    results_combigeo: "dict[int, OptimalResult]"
    discrepancies: List[Discrepancy]

    @property
    def agree(self) -> bool:
        """True, если расхождений не найдено."""
        return not self.discrepancies

    def __repr__(self) -> str:
        verdict = "совпадают" if self.agree else f"расхождений: {len(self.discrepancies)}"
        return f"ComparisonReport({len(self.results_combigeo)} индексов, {verdict})"


def compare_backends(basis: Sequence[Sequence[float]], indices: Iterable[int],
                     tol: float = 1e-6) -> ComparisonReport:
    """Прогоняет find_optimal на обоих бэкендах и сверяет результаты.

    Допустимо только для размерности 4 (единственное пересечение бэкендов).

    Сверяются диаметр и нормированное расстояние d: с версии 1.1.0 оба бэкенда
    возвращают ТОЧНОЕ d при любом его значении (фасад voronoi4d считает без
    раннего выхода), поэтому значения сравниваются в пределах tol всегда;
    отдельно фиксируется расхождение вердикта пригодности с tol-зазором на
    границе d = 1 (чтобы честный граничный случай не давал ложного конфликта).

    :param basis: базис 4×4.
    :param indices: набор чисел цветов (индексов подрешёток).
    :param tol: допуск сравнения.
    :return: отчёт со словарями результатов и списком расхождений.
    """
    basis = [list(map(float, row)) for row in basis]  # материализуем (вход мог быть генератором)
    if len(basis) != 4 or any(len(row) != 4 for row in basis):
        raise ValueError("compare_backends применим только к базисам 4×4")

    vb = get_backend("voronoi4d")
    cb = get_backend("combigeo")
    idx = sorted({int(k) for k in indices})

    res_v = {k: vb.find_optimal(basis, k) for k in idx}
    res_c = {k: cb.find_optimal(basis, k) for k in idx}

    discrepancies: List[Discrepancy] = []
    for k in idx:
        rv, rc = res_v[k], res_c[k]

        if abs(rv.diameter - rc.diameter) > tol:
            discrepancies.append(Discrepancy(k, "diameter", rv.diameter, rc.diameter,
                                             abs(rv.diameter - rc.diameter)))

        dv, dc = rv.normalized, rc.normalized
        if abs(dv - dc) > tol:
            discrepancies.append(Discrepancy(k, "normalized", dv, dc, abs(dv - dc)))
        elif (dv >= 1.0) != (dc >= 1.0) and abs(dv - 1.0) > tol and abs(dc - 1.0) > tol:
            # разный вердикт пригодности вне tol-окрестности границы d = 1
            discrepancies.append(Discrepancy(k, "feasible", dv, dc, abs(dv - dc)))

    return ComparisonReport(res_v, res_c, discrepancies)
