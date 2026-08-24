"""Точный рациональный сертификат радиуса покрытия слоёной решётки.

Постановка.  Пусть ``Lambda = { (x + i c, i t) : x in L, i in Z }`` — решётка,
ламинированная слоями базовой решётки ``L`` (размерность ``n``) с высотой ``t``
и сдвигом ``c``.  Тогда

    dist^2((x, z), Lambda) = min_i [ f_i(x) + (z - i t)^2 ],
    f_i(x) = dist^2(x - i c, L),

и ограничение минимума ЛЮБЫМ подмножеством слоёв минимум только увеличивает.
Берём слои ``{0, 1}``; ``f_0`` и ``f_1`` периодичны по ``L``, поэтому достаточно
``x in V_0`` (ячейка Вороного ``L``) и ``z in [0, t]``:

    R^2 <= max_{x in V_0} U(x),
    U(x) = max_{z in [0,t]} min( A_0 + z^2, A_1 + (z - t)^2 ),
    A_0 = |x|^2,  A_1 = dist^2(x, L + c).

Далее — три точных шага.

1. **Замена U на гладкую мажоранту.**  Пересечение двух парабол находится в
   ``z* = (A_1 - A_0 + t^2) / (2t)``; если ``z*`` вне ``[0, t]``, максимум
   достигается в конце отрезка и он МЕНЬШЕ значения в ``z*``.  Поэтому всюду

       U(x) <= phi(x) = A_0 + (A_1 - A_0 + t^2)^2 / (4 t^2),

   и в мажоранту входит только ``t^2`` — иррациональная высота исчезает.

2. **Разбиение на куски.**  Ячейки Вороного слоя 1 покрывают пространство,
   поэтому ``V_0`` покрыто кусками ``P_s = V_0 ∩ (s + V_0)``, где ``s = c + m``
   пробегает смежный класс ``c + L``.  На куске ближайшая точка слоя 1
   постоянна, значит ``A_1 = |x - s|^2``, разность ``A_1 - A_0`` ЛИНЕЙНА, а
   ``phi`` — ВЫПУКЛЫЙ квадратичный многочлен.  Полнота списка кусков доказуема:
   ``|s| <= |x| + |x - s| <= 2 R_cov(L)``.

3. **Максимум выпуклой функции — в вершине.**  Он берётся точным перечислением
   вершин (:mod:`chromatic_research.core.exact_dd`).

Ослабления безопасны.  Если для куска взято лишь ПОДМНОЖЕСТВО его неравенств,
получается многогранник ``P' ⊇ P_s``, и максимум по ``P'`` — по-прежнему
верхняя оценка.  На этом стоит схема CEGAR: неравенства добавляются по одному,
пока максимум не опустится ниже цели.  Пустой ``P'`` — доказательство пустоты
``P_s``.  Поэтому float используется ровно как ОРАКУЛ выбора неравенства, а
любая его ошибка стоит лишней итерации, но не корректности.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import Iterable, Sequence

from chromatic_research.core.exact_dd import DDLimit, dd_vertices


# ============================================================================
# точная линейная алгебра
# ============================================================================


def solve_exact(matrix: Sequence[Sequence[Fr]], rhs: Sequence[Fr]) -> list[Fr]:
    """Решение квадратной системы Гауссом в дробях."""
    size = len(matrix)
    work = [[Fr(matrix[i][j]) for j in range(size)] + [Fr(rhs[i])]
            for i in range(size)]
    for col in range(size):
        pivot = next((r for r in range(col, size) if work[r][col] != 0), None)
        if pivot is None:
            raise ValueError("вырожденная система")
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [v / scale for v in work[col]]
        for row in range(size):
            if row != col and work[row][col] != 0:
                factor = work[row][col]
                work[row] = [work[row][k] - factor * work[col][k]
                             for k in range(size + 1)]
    return [work[i][size] for i in range(size)]


def inverse_exact(matrix: Sequence[Sequence[Fr]]) -> list[list[Fr]]:
    size = len(matrix)
    columns = []
    for j in range(size):
        unit = [Fr(1) if i == j else Fr(0) for i in range(size)]
        columns.append(solve_exact(matrix, unit))
    return [[columns[j][i] for j in range(size)] for i in range(size)]


def ldl_exact(matrix: Sequence[Sequence[Fr]]):
    """``d_i`` и ``mu_ij`` (j>i) разложения ``L D L^T`` в дробях."""
    size = len(matrix)
    work = [[Fr(matrix[i][j]) for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            work[j][i] = work[i][j]
            work[i][j] = work[i][j] / work[i][i]
        for k in range(i + 1, size):
            for l in range(k, size):
                work[k][l] = work[k][l] - work[k][i] * work[i][l]
    diag = [work[i][i] for i in range(size)]
    mu = [[work[i][j] if j > i else Fr(0) for j in range(size)]
          for i in range(size)]
    return diag, mu


def enumerate_shifted(
    gram: Sequence[Sequence[Fr]],
    shift: Sequence[Fr],
    bound: Fr,
) -> list[tuple[int, ...]]:
    """Все ``m in Z^n`` с ``(m + shift)^T G (m + shift) <= bound``.  Точно.

    Границы перебора берутся с запасом в одну единицу с каждой стороны и
    каждый кандидат затем проверяется точным неравенством, поэтому
    вычисление квадратного корня в плавающей точке на полноту не влияет.
    """
    size = len(gram)
    diag, mu = ldl_exact(gram)
    shift = [Fr(s) for s in shift]
    bound = Fr(bound)
    current = [0] * size
    out: list[tuple[int, ...]] = []

    def recurse(level: int, remaining: Fr) -> None:
        tail = sum(mu[level][j] * (Fr(current[j]) + shift[j])
                   for j in range(level + 1, size))
        limit = remaining / diag[level]
        span = math.sqrt(float(limit)) if limit > 0 else 0.0
        centre = float(-shift[level] - tail)
        for value in range(math.floor(centre - span) - 1,
                           math.ceil(centre + span) + 2):
            coordinate = Fr(value) + shift[level] + tail
            used = diag[level] * coordinate * coordinate
            if used > remaining:
                continue
            current[level] = value
            if level == 0:
                out.append(tuple(current))
            else:
                recurse(level - 1, remaining - used)
        current[level] = 0

    recurse(size - 1, bound)
    return out


# ============================================================================
# геометрия слоёной решётки
# ============================================================================


@dataclass(frozen=True)
class LayeredGeometry:
    """Точные данные базовой решётки и слоя (всё в координатах базиса базы)."""

    gram: tuple[tuple[Fr, ...], ...]
    offset: tuple[Fr, ...]
    height2: Fr
    base_r2: Fr
    relevant: tuple[tuple[int, ...], ...] = field(default=())

    @property
    def dim(self) -> int:
        return len(self.gram)

    def dot(self, u: Sequence[Fr], v: Sequence[Fr]) -> Fr:
        total = Fr(0)
        for i in range(self.dim):
            ui = u[i]
            if ui == 0:
                continue
            row = self.gram[i]
            total += ui * sum(row[j] * v[j] for j in range(self.dim))
        return total

    def norm2(self, u: Sequence[Fr]) -> Fr:
        return self.dot(u, u)

    def with_relevant(self) -> "LayeredGeometry":
        """Векторы ``v != 0`` с ``|v|^2 <= 4 R_cov^2`` — надмножество релевантных.

        Надмножество задаёт ту же ячейку Вороного (лишние неравенства
        избыточны), поэтому описание ``V_0`` точное.
        """
        if self.relevant:
            return self
        bound = 4 * self.base_r2
        vectors = [v for v in enumerate_shifted(self.gram, [Fr(0)] * self.dim, bound)
                   if any(v)]
        return LayeredGeometry(self.gram, self.offset, self.height2,
                               self.base_r2, tuple(vectors))

    def coordinate_bound(self) -> Fr:
        """Рациональное ``M`` с гарантией ``|x_i| <= M`` для ``x in V_0``.

        ``|x_i| = |<x, e*_i>| <= |x| * |e*_i| <= R_cov * sqrt((G^-1)_ii)``.
        """
        inverse = inverse_exact(self.gram)
        best = Fr(0)
        for i in range(self.dim):
            value = self.base_r2 * inverse[i][i]
            approximation = Fr(math.ceil(math.sqrt(float(value)) * 1000) + 1, 1000)
            if approximation > best:
                best = approximation
        return best

    def piece_shifts(self) -> list[tuple[Fr, ...]]:
        """Все ``s = c + m`` с ``|s|^2 <= 4 R_cov^2``: полный список кусков."""
        shifts = enumerate_shifted(self.gram, self.offset, 4 * self.base_r2)
        return [tuple(Fr(value) + self.offset[i] for i, value in enumerate(m))
                for m in shifts]


def geometry_from_gram(
    integer_gram: Sequence[Sequence[int]],
    denominator: int,
    base_r2: Fr,
) -> LayeredGeometry:
    """Разбирает грамиан ``(n+1) x (n+1)`` на базу, сдвиг слоя и высоту."""
    full = [[Fr(v, denominator) for v in row] for row in integer_gram]
    size = len(full) - 1
    base = [row[:size] for row in full[:size]]
    column = [full[size][i] for i in range(size)]
    offset = solve_exact(base, column)
    height2 = full[size][size] - sum(column[i] * offset[i] for i in range(size))
    geometry = LayeredGeometry(tuple(tuple(r) for r in base), tuple(offset),
                               height2, Fr(base_r2))
    return geometry.with_relevant()


# ============================================================================
# сертификат
# ============================================================================


def _phi(geometry: LayeredGeometry, point: Sequence[Fr],
         shift: Sequence[Fr]) -> Fr:
    a0 = geometry.norm2(point)
    delta = [point[i] - shift[i] for i in range(geometry.dim)]
    a1 = geometry.norm2(delta)
    numerator = a1 - a0 + geometry.height2
    return a0 + numerator * numerator / (4 * geometry.height2)


def _halfspace_rows(geometry: LayeredGeometry, shift: Sequence[Fr]):
    """Неравенства обоих слоёв в виде ``(row, rhs, float_row, float_rhs)``.

    Слой 0: ``<x, v> <= |v|^2 / 2``.  Слой 1: ``<x - s, v> <= |v|^2 / 2``.
    """
    dim = geometry.dim
    rows, rhs = [], []
    for vector in geometry.relevant:
        row = tuple(sum(geometry.gram[i][j] * vector[j] for j in range(dim))
                    for i in range(dim))
        norm = sum(row[i] * vector[i] for i in range(dim))
        rows.append(row)
        rhs.append(norm / 2)
    shifted_rhs = [rhs[i] + sum(rows[i][k] * shift[k] for k in range(dim))
                   for i in range(len(rows))]
    return rows, rhs, shifted_rhs


@dataclass
class PieceReport:
    shift: tuple[Fr, ...]
    bound: Fr | None                    # None => кусок доказано пуст
    vertices: int
    constraints: int
    iterations: int


def certify_piece(
    geometry: LayeredGeometry,
    shift: Sequence[Fr],
    target: Fr | None,
    rows, rhs, shifted_rhs,
    *,
    coordinate_bound: Fr,
    max_iterations: int = 400,
    batch: int = 6,
) -> PieceReport:
    """Точная верхняя оценка ``max phi`` на куске ``V_0 ∩ (s + V_0)``.

    Схема CEGAR: начинаем с пустого набора неравенств (то есть с объемлющего
    симплекса), на каждом шаге берём вершину с наибольшим ``phi`` и добавляем
    неравенства куска, которые она нарушает сильнее всего.

    ``target = None`` — считать ТОЧНЫЙ максимум (итерации до тех пор, пока
    вершина-аргмаксимум не окажется допустимой для всего куска); используется
    в контрольных прогонах и для отчёта о реальном значении.
    """
    dim = geometry.dim
    count = len(rows)
    # float-оракул выбора неравенства (на корректность не влияет)
    float_rows = [[float(v) for v in row] for row in rows]
    float_rhs0 = [float(v) for v in rhs]
    float_rhs1 = [float(v) for v in shifted_rhs]

    active_rows: list[tuple[Fr, ...]] = []
    active_rhs: list[Fr] = []
    used: set[tuple[int, int]] = set()

    for iteration in range(max_iterations):
        try:
            vertices = dd_vertices(active_rows, active_rhs, dim,
                                   coordinate_bound)
        except DDLimit as error:                      # pragma: no cover
            raise RuntimeError(f"кусок {shift}: {error}") from error
        if not vertices:
            return PieceReport(tuple(shift), None, 0, len(active_rows),
                               iteration)
        values = [_phi(geometry, vertex, shift) for vertex in vertices]
        best = max(values)
        if target is not None and best < target:
            return PieceReport(tuple(shift), best, len(vertices),
                               len(active_rows), iteration)
        worst = vertices[values.index(best)]

        # какие неравенства куска нарушает худшая вершина (float-скрининг)
        point = [float(v) for v in worst]
        violations: list[tuple[float, int, int]] = []
        for index in range(count):
            row = float_rows[index]
            value = sum(row[k] * point[k] for k in range(dim))
            gap0 = value - float_rhs0[index]
            if gap0 > 1e-12 and (0, index) not in used:
                violations.append((gap0, 0, index))
            gap1 = value - float_rhs1[index]
            if gap1 > 1e-12 and (1, index) not in used:
                violations.append((gap1, 1, index))
        if not violations:
            # вершина действительно принадлежит куску: оценка неулучшаема
            return PieceReport(tuple(shift), best, len(vertices),
                               len(active_rows), iteration)
        violations.sort(reverse=True)
        for _, layer, index in violations[:batch]:
            used.add((layer, index))
            active_rows.append(rows[index])
            active_rhs.append(rhs[index] if layer == 0 else shifted_rhs[index])

    raise RuntimeError(f"кусок {shift}: не сошлось за {max_iterations} итераций")


def certify_covering_radius(
    geometry: LayeredGeometry,
    target_r2: Fr | None,
    *,
    verbose: bool = True,
    progress_every: int = 50,
) -> dict:
    """Доказывает ``R^2 < target_r2`` для слоёной решётки (или сообщает провал)."""
    start = time.time()
    geometry = geometry.with_relevant()
    bound = geometry.coordinate_bound()
    shifts = geometry.piece_shifts()
    rows, rhs, _ = _halfspace_rows(geometry, [Fr(0)] * geometry.dim)

    worst = Fr(0)
    worst_shift = None
    reports: list[PieceReport] = []
    empty = 0
    failures: list[tuple[Fr, ...]] = []
    for number, shift in enumerate(shifts):
        shifted_rhs = [rhs[i] + sum(rows[i][k] * shift[k]
                                    for k in range(geometry.dim))
                       for i in range(len(rows))]
        report = certify_piece(geometry, shift, target_r2, rows, rhs,
                               shifted_rhs, coordinate_bound=bound)
        reports.append(report)
        if report.bound is None:
            empty += 1
        else:
            if target_r2 is not None and report.bound >= target_r2:
                failures.append(tuple(shift))
            if report.bound > worst:
                worst = report.bound
                worst_shift = tuple(shift)
        if verbose and (number + 1) % progress_every == 0:
            print(f"    {number + 1}/{len(shifts)} кусков, пустых {empty}, "
                  f"max phi {float(worst):.9f} [{time.time() - start:.0f}s]",
                  flush=True)

    return {
        "certified": not failures,
        "target_r2": target_r2,
        "max_phi": worst,
        "worst_shift": worst_shift,
        "pieces": len(shifts),
        "empty_pieces": empty,
        "failed_pieces": failures,
        "max_constraints": max(r.constraints for r in reports),
        "max_vertices": max(r.vertices for r in reports),
        "seconds": round(time.time() - start, 1),
    }
