"""Точный рациональный сертификат для chi(R^7, [1, ell]) <= 1029.

Переводит ламинирование E6*/343 с трёхкратным слоем из статуса **[Ч]** в
**[С]**: всё, что раньше считалось в плавающей точке (вершины кусочного
сертификата диаметра, проекции на ячейку), пересчитывается над Z и Q.

Схема:

1. Фиксируется рациональная матрица Грама Q = G / 10000 -- близкая
   к численному оптимуму форма. Дальше проверяется именно ЭТА решётка,
   поэтому происхождение G на доказательство не влияет.
2. Подрешётка Gamma = C Z^7 задаётся целыми столбцами, |det C| = 1029
   считается по Барейссу.
3. Voronoi-relevant векторы -- по теореме Вороного о Lambda/2Lambda:
   v релевантен тогда и только тогда, когда +-v -- единственная пара
   кратчайших представителей своего ненулевого класса. Граница перебора
   max_{s in {0,1}^7} Q(s,s) полна, так как у каждого класса есть
   представитель из нулей и единиц.
4. Вершины ячейки: Qhull лишь ПРЕДЛАГАЕТ точки, каждая переводится в точную
   рациональную по N независимым активным фасетам. Полнота доказывается
   точным замыканием по 1-скелету: у каждой вершины перебираются все
   (N-1)-подмножества активных фасет, и если такое подмножество задаёт
   допустимый экстремальный луч, второй конец ребра обязан быть в наборе.
   Многогранник НЕ предполагается простым -- у этой решётки 1600 вершин из
   30368 имеют по девять активных фасет.
5. R^2 -- максимум по вершинам, diam^2 = 4 R^2 (ячейка центрально
   симметрична).
6. Полный конечный набор кандидатов на D_min: из D(v) >= |v| - diam
   улучшить рекорд может лишь |v| < D_best + diam; чтобы не извлекать
   корней, берётся рациональная подстраховка (a+b)^2 <= 2(a^2+b^2).
7. Каждое D(v)^2 = 4 dist(v/2, V0)^2 -- точный KKT-сертификат: множители
   mu >= 0 и допустимость проекции, чего для выпуклой задачи достаточно.

Результат: D_min^2 = 7 ровно (минимум достигается на 27 горизонтальных
векторах -- это пол sqrt(7/3) lambda_1 теоремы об эйзенштейновых
конструкциях), d^2 = 103332736237500000/96858928789031597,
d = 1.03287825..., и замкнутый интервал [1, 103/100] проходит с точным
запасом 5750986852163787427/147618194625000000000.

NumPy и SciPy используются ТОЛЬКО как оракулы (Qhull подсказывает вершины,
SLSQP -- активный набор проекции). Ни одна их величина не входит в
доказательство: decimal-печать нужна лишь для чтения, а доказательные
radius_squared, diameter_squared, minimum_distance_squared,
normalized_distance_squared и margin -- это Fraction.

Происхождение: верификатор написан Н. Глушковой (август 2026), см.
journal/REVIEW-notes-v6.md. Здесь он приведён к соглашениям репозитория.

Usage::

    python -m chromatic_research.campaigns.dim7_1029_exact
    python -m chromatic_research.campaigns.dim7_1029_exact --ell 103/100
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction as Fr
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import HalfspaceIntersection

from chromatic_research.paths import results_path


# ============================================================================
# 1. ДАННЫЕ РАЦИОНАЛЬНОЙ КОНСТРУКЦИИ 1029
# ============================================================================

N = 7
DENOMINATOR = 10_000

# Рациональная матрица Грама Q = INTEGER_GRAM / DENOMINATOR.
# Это близкая рациональная форма к численной dim7_1029_rational.json.
# После фиксации G проверяется именно ЭТА рациональная решётка.
INTEGER_GRAM = [
    [ 30000,  15000,      0, -22500,  15000,   7500,  -8632],
    [ 15000,  30000,  22500,      0,   7500,  15000,  -4197],
    [     0,  22500,  45000,  22500,      0,  22500,   1601],
    [-22500,      0,  22500,  45000, -22500,      0,   6617],
    [ 15000,   7500,      0, -22500,  30000,  15000,  -5275],
    [  7500,  15000,  22500,      0,  15000,  30000,   4434],
    [ -8632,  -4197,   1601,   6617,  -5275,   4434,  17897],
]

# В репозитории kernel_rows возвращает СТРОКОВЫЙ базис
#
#   K =
#   [ 7  0  0  0  0  0  0]
#   [-5  1  0  0  0  0  0]
#   [ 0  0  7  0  0  0  0]
#   [ 0  0 -5  1  0  0  0]
#   [ 0  0  0  0  7  0  0]
#   [ 0  0  0  0 -5  1  0]
#   [ 0  0  0  0  0  0  3]
#
# Здесь соглашение Gamma = C Z^7, т.е. базис задаётся СТОЛБЦАМИ,
# поэтому C = K^T.
SUBLATTICE_COLUMNS = [
    [ 7, -5,  0,  0,  0,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  7, -5,  0,  0,  0],
    [ 0,  0,  0,  1,  0,  0,  0],
    [ 0,  0,  0,  0,  7, -5,  0],
    [ 0,  0,  0,  0,  0,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  3],
]

EXPECTED_INDEX = 1029


def configure(*, n: int, denominator: int, integer_gram, sublattice_columns,
              expected_index: int) -> None:
    """Перенастроить верификатор на другую рациональную решётку.

    Все процедуры читают данные из глобалей модуля в момент вызова, поэтому
    достаточно их переприсвоить. Нужно, чтобы тот же самый -- уже проверенный --
    код можно было натравить на другую размерность, не копируя его.
    """
    global N, DENOMINATOR, INTEGER_GRAM, SUBLATTICE_COLUMNS, EXPECTED_INDEX
    if len(integer_gram) != n or any(len(row) != n for row in integer_gram):
        raise ValueError("матрица Грама не согласована с размерностью")
    if len(sublattice_columns) != n or any(len(col) != n for col in sublattice_columns):
        raise ValueError("матрица подрешётки не согласована с размерностью")
    N = int(n)
    DENOMINATOR = int(denominator)
    INTEGER_GRAM = [[int(x) for x in row] for row in integer_gram]
    SUBLATTICE_COLUMNS = [[int(x) for x in col] for col in sublattice_columns]
    EXPECTED_INDEX = int(expected_index)

# По умолчанию проверяем закрытый интервал [1, 1.03].
# Само d^2 вычисляется независимо от этого ell.
DEFAULT_ELL = Fr(103, 100)


# ============================================================================
# 2. БАЗОВАЯ ТОЧНАЯ АРИФМЕТИКА
# ============================================================================


def qdot_int(u: Sequence[int], v: Sequence[int]) -> int:
    """Точное u^T G v для целых координат."""
    return sum(
        int(u[i]) * INTEGER_GRAM[i][j] * int(v[j])
        for i in range(N)
        for j in range(N)
    )


def qdot_frac(u: Sequence[Fr], v: Sequence[Fr]) -> Fr:
    """Точное u^T G v для рациональных координат."""
    return sum(
        Fr(u[i]) * INTEGER_GRAM[i][j] * Fr(v[j])
        for i in range(N)
        for j in range(N)
    )


def gram_row(vector: Sequence[int]) -> tuple[int, ...]:
    """Строка v^T G."""
    return tuple(
        sum(int(vector[i]) * INTEGER_GRAM[i][j] for i in range(N))
        for j in range(N)
    )


def det_bareiss_int(matrix: Sequence[Sequence[int]]) -> int:
    """Точный определитель целой квадратной матрицы методом Bareiss."""
    n = len(matrix)
    if n == 0:
        return 1
    if any(len(row) != n for row in matrix):
        raise ValueError("det_bareiss_int: matrix must be square")
    if n == 1:
        return int(matrix[0][0])

    a = [list(map(int, row)) for row in matrix]
    sign = 1
    previous_pivot = 1

    for k in range(n - 1):
        if a[k][k] == 0:
            pivot_row = next(
                (r for r in range(k + 1, n) if a[r][k] != 0),
                None,
            )
            if pivot_row is None:
                return 0
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign *= -1

        pivot = a[k][k]

        for i in range(k + 1, n):
            aik = a[i][k]
            for j in range(k + 1, n):
                numerator = pivot * a[i][j] - aik * a[k][j]
                if numerator % previous_pivot != 0:
                    raise ArithmeticError("Bareiss exact division failed")
                a[i][j] = numerator // previous_pivot
            a[i][k] = 0

        previous_pivot = pivot

    return sign * a[n - 1][n - 1]


def positive_definite_leading_minors() -> list[int]:
    """Точная проверка G>0 по критерию Сильвестра."""
    for i in range(N):
        for j in range(N):
            if INTEGER_GRAM[i][j] != INTEGER_GRAM[j][i]:
                raise RuntimeError("G is not symmetric")

    minors = [
        det_bareiss_int([row[:k] for row in INTEGER_GRAM[:k]])
        for k in range(1, N + 1)
    ]
    if any(value <= 0 for value in minors):
        raise RuntimeError(
            "G is not positive definite by Sylvester criterion: "
            f"{minors}"
        )
    return minors


def solve_linear_fraction(
    matrix: Sequence[Sequence[int | Fr]],
    rhs: Sequence[int | Fr],
) -> list[Fr] | None:
    """Точное решение A x=b над Q методом Гаусса."""
    n = len(matrix)
    if n == 0:
        return []
    if len(rhs) != n or any(len(row) != n for row in matrix):
        raise ValueError("solve_linear_fraction: incompatible dimensions")

    aug = [
        [Fr(matrix[i][j]) for j in range(n)] + [Fr(rhs[i])]
        for i in range(n)
    ]

    for col in range(n):
        pivot_row = next(
            (r for r in range(col, n) if aug[r][col] != 0),
            None,
        )
        if pivot_row is None:
            return None

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def solve_integer_system_common_denominator(
    matrix: Sequence[Sequence[int]],
    rhs: Sequence[int],
) -> tuple[tuple[int, ...], int] | None:
    """Решает A x=b точно и возвращает x=num/den, den>0."""
    n = len(matrix)
    if n == 0:
        return tuple(), 1
    if len(rhs) != n or any(len(row) != n for row in matrix):
        raise ValueError("solve_integer_system_common_denominator: bad shape")

    aug = [list(map(int, matrix[i])) + [int(rhs[i])] for i in range(n)]
    previous_pivot = 1

    for k in range(n - 1):
        if aug[k][k] == 0:
            pivot_row = next(
                (r for r in range(k + 1, n) if aug[r][k] != 0),
                None,
            )
            if pivot_row is None:
                return None
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]

        pivot = aug[k][k]
        for i in range(k + 1, n):
            aik = aug[i][k]
            for j in range(k + 1, n + 1):
                numerator = pivot * aug[i][j] - aik * aug[k][j]
                if numerator % previous_pivot != 0:
                    raise ArithmeticError("Bareiss solve: non-exact division")
                aug[i][j] = numerator // previous_pivot
            aug[i][k] = 0
        previous_pivot = pivot

    if aug[n - 1][n - 1] == 0:
        return None

    solution = [Fr(0)] * n
    for i in range(n - 1, -1, -1):
        value = Fr(aug[i][n]) - sum(
            Fr(aug[i][j]) * solution[j]
            for j in range(i + 1, n)
        )
        if aug[i][i] == 0:
            return None
        solution[i] = value / aug[i][i]

    denominator = 1
    for value in solution:
        denominator = math.lcm(denominator, value.denominator)

    numerators = tuple(
        value.numerator * (denominator // value.denominator)
        for value in solution
    )

    common = denominator
    for value in numerators:
        common = math.gcd(common, abs(value))
    if common > 1:
        numerators = tuple(value // common for value in numerators)
        denominator //= common

    if denominator < 0:
        denominator = -denominator
        numerators = tuple(-value for value in numerators)

    return numerators, denominator


def floor_fraction(value: Fr) -> int:
    value = Fr(value)
    return value.numerator // value.denominator


def ceil_fraction(value: Fr) -> int:
    return -floor_fraction(-Fr(value))


def floor_sqrt_fraction(value: Fr) -> int:
    """floor(sqrt(value)) для value>=0 без float."""
    value = Fr(value)
    if value < 0:
        raise ValueError("square root of negative rational")
    return math.isqrt(value.numerator // value.denominator)


def ldl_decomposition_exact(
    matrix: Sequence[Sequence[int]],
) -> tuple[list[list[Fr]], list[Fr]]:
    """Точное M=L D L^T для положительно определённой целой M."""
    n = len(matrix)
    L = [[Fr(0)] * n for _ in range(n)]
    D = [Fr(0)] * n

    for i in range(n):
        L[i][i] = Fr(1)

    for j in range(n):
        D[j] = Fr(matrix[j][j]) - sum(
            L[j][k] * L[j][k] * D[k]
            for k in range(j)
        )
        if D[j] <= 0:
            raise RuntimeError("LDL: matrix is not positive definite")

        for i in range(j + 1, n):
            L[i][j] = (
                Fr(matrix[i][j])
                - sum(
                    L[i][k] * L[j][k] * D[k]
                    for k in range(j)
                )
            ) / D[j]

    return L, D


def enumerate_quadratic_form_exact(
    matrix: Sequence[Sequence[int]],
    bound: int | Fr,
) -> list[tuple[int, ...]]:
    """Все ненулевые z с z^T M z <= bound, полностью точно."""
    n = len(matrix)
    L, D = ldl_decomposition_exact(matrix)
    bound = Fr(bound)
    if bound < 0:
        return []

    coefficients = [0] * n
    result: list[tuple[int, ...]] = []

    def recurse(j: int, accumulated: Fr) -> None:
        if accumulated > bound:
            return
        if j < 0:
            vector = tuple(coefficients)
            if any(vector):
                result.append(vector)
            return

        center_shift = sum(
            L[i][j] * coefficients[i]
            for i in range(j + 1, n)
        )
        remaining = bound - accumulated
        ratio = remaining / D[j]
        root_floor = floor_sqrt_fraction(ratio)

        safe_radius = root_floor + 1
        lower = ceil_fraction(-center_shift - safe_radius)
        upper = floor_fraction(-center_shift + safe_radius)

        for value in range(lower, upper + 1):
            shifted = Fr(value) + center_shift
            new_accumulated = accumulated + D[j] * shifted * shifted
            if new_accumulated <= bound:
                coefficients[j] = value
                recurse(j - 1, new_accumulated)

        coefficients[j] = 0

    recurse(n - 1, Fr(0))
    return result


def canonical_sign(vector: Sequence[int]) -> tuple[int, ...]:
    """Канонический представитель пары {v,-v}."""
    result = tuple(int(x) for x in vector)
    for value in result:
        if value != 0:
            return result if value > 0 else tuple(-x for x in result)
    return result


def matrix_vector_columns(
    matrix: Sequence[Sequence[int]],
    coefficients: Sequence[int],
) -> tuple[int, ...]:
    """C z для матрицы, чьи столбцы являются базисом подрешётки."""
    n = len(matrix)
    return tuple(
        sum(int(matrix[i][j]) * int(coefficients[j]) for j in range(n))
        for i in range(n)
    )


def sublattice_integer_gram() -> list[list[int]]:
    """Точная матрица C^T G C без общего знаменателя."""
    C = SUBLATTICE_COLUMNS
    return [
        [
            sum(
                C[i][a] * INTEGER_GRAM[i][j] * C[j][b]
                for i in range(N)
                for j in range(N)
            )
            for b in range(N)
        ]
        for a in range(N)
    ]


# ============================================================================
# 3. VORONOI-RELEVANT ВЕКТОРЫ ЧЕРЕЗ Lambda / 2 Lambda
# ============================================================================


def exact_relevant_vectors() -> tuple[list[tuple[int, ...]], dict]:
    """Находит все пары ±v Voronoi-relevant точно.

    Теорема: v relevant тогда и только тогда, когда ±v — единственные
    кратчайшие представители своего ненулевого класса v+2Lambda.

    Если в parity-классе кратчайших представителей больше одной пары,
    этот класс просто НЕ создаёт relevant-фасету. В отличие от старого кода
    для 1323 здесь это не считается ошибкой.
    """
    parity_seeds = list(itertools.product((0, 1), repeat=N))
    global_upper = max(
        qdot_int(seed, seed)
        for seed in parity_seeds
        if any(seed)
    )

    vectors = enumerate_quadratic_form_exact(INTEGER_GRAM, global_upper)

    best_norm: dict[tuple[int, ...], int] = {}
    minimizers: dict[tuple[int, ...], list[tuple[int, ...]]] = {}

    for vector in vectors:
        parity = tuple(value & 1 for value in vector)
        if not any(parity):
            continue

        norm = qdot_int(vector, vector)
        if parity not in best_norm or norm < best_norm[parity]:
            best_norm[parity] = norm
            minimizers[parity] = [vector]
        elif norm == best_norm[parity]:
            minimizers[parity].append(vector)

    expected_classes = 2**N - 1
    if len(best_norm) != expected_classes:
        raise RuntimeError(
            f"parity coverage failed: {len(best_norm)} of {expected_classes}"
        )

    relevant: list[tuple[int, ...]] = []
    non_relevant_classes: list[dict] = []

    for parity in sorted(minimizers):
        unique = sorted(set(minimizers[parity]))
        if (
            len(unique) == 2
            and unique[0] == tuple(-x for x in unique[1])
        ):
            relevant.append(canonical_sign(unique[0]))
        else:
            non_relevant_classes.append({
                "parity": parity,
                "number_of_shortest_representatives": len(unique),
                "shortest_norm_integer": best_norm[parity],
            })

    stats = {
        "global_upper_integer": global_upper,
        "parent_vectors_enumerated": len(vectors),
        "parity_classes": len(best_norm),
        "non_relevant_parity_classes": non_relevant_classes,
    }
    return sorted(set(relevant)), stats


def all_oriented_facets(
    relevant: Iterable[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    facets: set[tuple[int, ...]] = set()
    for raw in relevant:
        vector = tuple(map(int, raw))
        facets.add(vector)
        facets.add(tuple(-x for x in vector))
    return tuple(sorted(facets))


# ============================================================================
# 4. ТОЧНАЯ ФАСЕТНАЯ СИСТЕМА
# ============================================================================


class FacetSystem:
    """Неравенства v^T G x <= (v^T G v)/2."""

    def __init__(self, facets: Sequence[Sequence[int]]):
        self.facets = tuple(tuple(map(int, f)) for f in facets)
        self.normals = tuple(gram_row(f) for f in self.facets)
        self.qnorms = tuple(qdot_int(f, f) for f in self.facets)
        self.offsets = tuple(Fr(value, 2) for value in self.qnorms)

    def exact_active_fraction(
        self,
        point: Sequence[Fr],
    ) -> tuple[bool, tuple[int, ...] | None]:
        active: list[int] = []
        for i, normal in enumerate(self.normals):
            lhs = sum(Fr(normal[k]) * Fr(point[k]) for k in range(N))
            rhs = self.offsets[i]
            if lhs > rhs:
                return False, None
            if lhs == rhs:
                active.append(i)
        return True, tuple(active)

    def exact_active_y_common_denominator(
        self,
        numerators: Sequence[int],
        denominator: int,
    ) -> tuple[bool, tuple[int, ...] | None]:
        """Проверяет x=y/2, y=numerators/denominator, только над Z."""
        if denominator <= 0:
            raise ValueError("denominator must be positive")

        active: list[int] = []
        for i, normal in enumerate(self.normals):
            lhs = sum(normal[k] * int(numerators[k]) for k in range(N))
            rhs = self.qnorms[i] * denominator
            if lhs > rhs:
                return False, None
            if lhs == rhs:
                active.append(i)
        return True, tuple(active)


# ============================================================================
# 5. ТОЧНЫЕ ВЕРШИНЫ, В ТОМ ЧИСЛЕ НЕПРОСТЫЕ
# ============================================================================


@dataclass(frozen=True)
class ExactVertexCertificate:
    # Полный точный набор активных фасет; может содержать >N фасет.
    active_facets: tuple[int, ...]
    # y = 2x = numerators / denominator
    y_numerators: tuple[int, ...]
    y_denominator: int
    radius_squared: Fr

    def coordinates(self) -> tuple[Fr, ...]:
        return tuple(
            Fr(value, 2 * self.y_denominator)
            for value in self.y_numerators
        )


def numerical_halfspaces(
    facets: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Только float-оракул: Q, A, b для A x <= b."""
    q = np.asarray(INTEGER_GRAM, dtype=float) / DENOMINATOR
    facet_array = np.asarray(facets, dtype=float)
    a = facet_array @ q
    b = 0.5 * np.einsum("ij,ij->i", a, facet_array)
    return q, a, b


def certify_vertex_from_independent_facets(
    basis_ids: Sequence[int],
    fs: FacetSystem,
) -> ExactVertexCertificate | None:
    """Сертифицирует вершину по N независимым активным фасетам.

    У истинной вершины может быть больше N активных фасет. N фасет нужны
    только чтобы однозначно решить систему. После решения точной системы
    вычисляется ПОЛНЫЙ активный набор по всем фасетам.
    """
    basis_ids = tuple(sorted(map(int, basis_ids)))
    if len(basis_ids) != N or len(set(basis_ids)) != N:
        return None

    matrix = [list(fs.normals[i]) for i in basis_ids]
    rhs = [fs.qnorms[i] for i in basis_ids]

    solved = solve_integer_system_common_denominator(matrix, rhs)
    if solved is None:
        return None

    numerators, denominator = solved
    feasible, exact_active = fs.exact_active_y_common_denominator(
        numerators,
        denominator,
    )
    if not feasible or exact_active is None:
        return None
    if not set(basis_ids).issubset(exact_active):
        return None

    # Для вершины ранг активных нормалей должен быть N; выбранный basis_ids
    # уже невырожден, значит это выполнено.
    gram_numerator = qdot_int(numerators, numerators)
    radius_squared = Fr(
        gram_numerator,
        4 * denominator * denominator * DENOMINATOR,
    )

    return ExactVertexCertificate(
        active_facets=tuple(exact_active),
        y_numerators=numerators,
        y_denominator=denominator,
        radius_squared=radius_squared,
    )


def certify_float_vertex_hint(
    candidate: np.ndarray,
    fs: FacetSystem,
    a_float: np.ndarray,
    b_float: np.ndarray,
) -> ExactVertexCertificate:
    """Qhull-точка лишь предлагает фасеты; результат полностью exact."""
    slack = np.abs(b_float - a_float @ candidate)
    order = [int(i) for i in np.argsort(slack)]

    # У части вершин 1029-конструкции 9 активных фасет, поэтому пробуем
    # N-подмножества немного расширенного пула ближайших численных фасет.
    for pool_size in (N, N + 1, N + 2, N + 4, N + 7):
        pool = sorted(set(order[:pool_size]))
        if len(pool) < N:
            continue
        for basis_ids in itertools.combinations(pool, N):
            certificate = certify_vertex_from_independent_facets(basis_ids, fs)
            if certificate is None:
                continue

            exact_point = np.array(
                [float(v) for v in certificate.coordinates()],
                dtype=float,
            )
            # Защита от случайного сертификата другой вершины из того же пула.
            if np.max(np.abs(exact_point - candidate)) <= 1e-6:
                return certificate

    raise RuntimeError(
        "A Qhull vertex hint could not be converted to an exact vertex"
    )


def primitive_integer_direction(vector: Sequence[int]) -> tuple[int, ...]:
    values = tuple(map(int, vector))
    g = 0
    for value in values:
        g = math.gcd(g, abs(value))
    if g == 0:
        raise ValueError("zero direction")
    values = tuple(value // g for value in values)
    # Канонический знак только для дедупликации линии; ориентация луча
    # будет выбрана позже по активным неравенствам.
    for value in values:
        if value != 0:
            if value < 0:
                values = tuple(-x for x in values)
            break
    return values


def null_direction_of_rank_n_minus_1(
    rows: Sequence[Sequence[int]],
) -> tuple[int, ...] | None:
    """Точный целый вектор ядра (N-1)xN матрицы ранга N-1.

    Используется вектор алгебраических дополнений:
        d_j = (-1)^j det(M without column j).
    """
    if len(rows) != N - 1 or any(len(row) != N for row in rows):
        raise ValueError("expected an (N-1) x N matrix")

    cofactors = []
    for j in range(N):
        square = [
            [int(row[k]) for k in range(N) if k != j]
            for row in rows
        ]
        value = det_bareiss_int(square)
        if j % 2:
            value = -value
        cofactors.append(value)

    if all(value == 0 for value in cofactors):
        return None
    return primitive_integer_direction(cofactors)


def tangent_extreme_rays_exact(
    certificate: ExactVertexCertificate,
    fs: FacetSystem,
) -> tuple[tuple[int, ...], ...]:
    """Все экстремальные лучи касательного конуса в вершине, точно.

    Касательный конус задаётся active normals * d <= 0.
    В R^N каждый экстремальный луч лежит в пересечении как минимум N-1
    линейно независимых активных гиперплоскостей. Перебираем такие наборы,
    вычисляем 1D-ядро и оставляем только направления, допустимые для ВСЕХ
    активных неравенств.
    """
    active = certificate.active_facets
    rays: set[tuple[int, ...]] = set()

    for ids in itertools.combinations(active, N - 1):
        rows = [fs.normals[i] for i in ids]
        line = null_direction_of_rank_n_minus_1(rows)
        if line is None:
            continue

        dots = [
            sum(fs.normals[i][k] * line[k] for k in range(N))
            for i in active
        ]

        if all(value <= 0 for value in dots) and any(value < 0 for value in dots):
            ray = tuple(line)
        elif all(value >= 0 for value in dots) and any(value > 0 for value in dots):
            ray = tuple(-x for x in line)
        else:
            continue

        # Здесь знак уже важен: это ориентированный луч от данной вершины.
        g = 0
        for value in ray:
            g = math.gcd(g, abs(value))
        ray = tuple(value // g for value in ray)
        rays.add(ray)

    if not rays:
        raise RuntimeError(
            f"no tangent extreme rays at vertex with active set {active}"
        )
    return tuple(sorted(rays))


def next_vertex_along_ray_exact(
    certificate: ExactVertexCertificate,
    ray: Sequence[int],
    fs: FacetSystem,
) -> tuple[tuple[Fr, ...], tuple[int, ...]]:
    """Идёт от вершины вдоль экстремального луча до следующей вершины."""
    x = certificate.coordinates()
    ray = tuple(map(int, ray))

    best_t: Fr | None = None

    for i, normal in enumerate(fs.normals):
        directional = sum(normal[k] * ray[k] for k in range(N))
        if directional <= 0:
            continue

        lhs = sum(Fr(normal[k]) * x[k] for k in range(N))
        remaining = fs.offsets[i] - lhs
        if remaining < 0:
            raise RuntimeError("current exact vertex is infeasible")

        t = remaining / directional
        if t < 0:
            continue
        if t == 0:
            # Для допустимого касательного луча активная фасета не может
            # иметь положительную производную. Если это случилось — ошибка.
            raise RuntimeError(
                "ray leaves the polytope immediately; tangent test failed"
            )
        if best_t is None or t < best_t:
            best_t = t

    if best_t is None:
        raise RuntimeError("unbounded ray in a Voronoi cell")

    endpoint = tuple(x[k] + best_t * ray[k] for k in range(N))
    feasible, active = fs.exact_active_fraction(endpoint)
    if not feasible or active is None:
        raise RuntimeError("computed edge endpoint is not feasible")

    return endpoint, active


def build_and_certify_all_vertices(
    fs: FacetSystem,
    a_float: np.ndarray,
    b_float: np.ndarray,
) -> dict:
    """Qhull даёт кандидатов; exact edge closure доказывает полноту."""
    halfspaces = np.column_stack((a_float, -b_float))
    hull = HalfspaceIntersection(
        halfspaces,
        np.zeros(N),
        qhull_options="Qx",
    )

    exact_vertices: dict[tuple[Fr, ...], ExactVertexCertificate] = {}
    maximum_radius_squared: Fr | None = None
    farthest_vertices: list[ExactVertexCertificate] = []

    total_hints = len(hull.intersections)
    print(f"      Qhull hints: {total_hints}", flush=True)

    for number, candidate in enumerate(hull.intersections, 1):
        certificate = certify_float_vertex_hint(
            candidate,
            fs,
            a_float,
            b_float,
        )
        key = certificate.coordinates()

        previous = exact_vertices.get(key)
        if previous is None:
            exact_vertices[key] = certificate
            r2 = certificate.radius_squared
            if maximum_radius_squared is None or r2 > maximum_radius_squared:
                maximum_radius_squared = r2
                farthest_vertices = [certificate]
            elif r2 == maximum_radius_squared:
                farthest_vertices.append(certificate)
        else:
            if previous.active_facets != certificate.active_facets:
                raise RuntimeError("same exact vertex got inconsistent active sets")

        if number % 5000 == 0 or number == total_hints:
            print(
                f"      exact vertex hints certified: {number}/{total_hints}; "
                f"unique={len(exact_vertices)}",
                flush=True,
            )

    if not exact_vertices:
        raise RuntimeError("no exact vertex was certified")

    # ----------------------------------------------------------------------
    # Exact edge closure for a possibly NON-SIMPLE polytope.
    #
    # Для каждой вершины перебираем все (N-1)-подмножества её активных
    # фасет. Если такое множество S имеет ранг N-1 и задаёт допустимый
    # касательный луч, то оно определяет ребро, выходящее из вершины.
    # Другой конец этого ребра обязан быть второй вершиной, содержащей все
    # фасеты S. Поэтому строим индекс S -> список найденных вершин.
    #
    # Если для потенциального ребра индекс содержит только одну вершину,
    # Qhull пропустил соседнюю вершину и сертификат отвергается.
    # Это даёт точное замыкание по 1-скелету без предположения простоты.
    # ----------------------------------------------------------------------
    print("      checking exact edge closure (non-simple safe)...", flush=True)

    active_size_hist: dict[int, int] = {}
    subset_to_vertices: dict[tuple[int, ...], list[int]] = {}
    vertex_list = list(exact_vertices.values())

    for vertex_id, certificate in enumerate(vertex_list):
        active = certificate.active_facets
        active_size_hist[len(active)] = active_size_hist.get(len(active), 0) + 1
        for subset in itertools.combinations(active, N - 1):
            subset_to_vertices.setdefault(tuple(subset), []).append(vertex_id)

    singleton_subsets = [
        (subset, ids[0])
        for subset, ids in subset_to_vertices.items()
        if len(ids) == 1
    ]

    # Для простой вершины (ровно N активных независимых фасет) каждое
    # (N-1)-подмножество действительно задаёт одно ребро. Поэтому singleton
    # там сразу означает пропущенного соседа. Для непростой вершины сначала
    # точным cofactor-вектором проверяем, действительно ли subset задаёт
    # допустимый экстремальный луч.
    bad_singletons: list[dict] = []
    checked_nonsimple_singletons = 0

    for number, (subset, vertex_id) in enumerate(singleton_subsets, 1):
        certificate = vertex_list[vertex_id]
        active = certificate.active_facets

        if len(active) == N:
            bad_singletons.append({
                "subset": list(subset),
                "vertex_active": list(active),
                "reason": "simple vertex edge has no second endpoint",
            })
        else:
            rows = [fs.normals[i] for i in subset]
            line = null_direction_of_rank_n_minus_1(rows)
            if line is None:
                # rank < N-1: этот subset не выделяет линию/ребро.
                continue

            dots = [
                sum(fs.normals[i][k] * line[k] for k in range(N))
                for i in active
            ]
            feasible_ray = (
                (all(v <= 0 for v in dots) and any(v < 0 for v in dots))
                or
                (all(v >= 0 for v in dots) and any(v > 0 for v in dots))
            )
            checked_nonsimple_singletons += 1
            if feasible_ray:
                bad_singletons.append({
                    "subset": list(subset),
                    "vertex_active": list(active),
                    "reason": "rank-(N-1) feasible edge ray has no second endpoint",
                })

        if bad_singletons:
            break

        if number % 10000 == 0 or number == len(singleton_subsets):
            print(
                f"      singleton ridge checks: {number}/{len(singleton_subsets)}",
                flush=True,
            )

    if bad_singletons:
        raise RuntimeError(
            "exact edge closure failed; first missing-edge certificate: "
            f"{bad_singletons[0]}"
        )

    # Для информации считаем число различных пар вершин, которые имеют
    # общий (N-1)-набор активных фасет. Для rank=N-1 это реальные рёбра;
    # при rank<N-1 такая пара может дать лишний кандидат, поэтому это число
    # называется edge_pair_candidates, а не exact edge count.
    edge_pair_candidates: set[tuple[int, int]] = set()
    for ids in subset_to_vertices.values():
        if len(ids) == 2:
            a_id, b_id = sorted(ids)
            edge_pair_candidates.add((a_id, b_id))

    print(
        f"      edge closure exact: OK; subset keys={len(subset_to_vertices)}, "
        f"singleton keys={len(singleton_subsets)}",
        flush=True,
    )

    assert maximum_radius_squared is not None

    return {
        "vertices": len(exact_vertices),
        "edge_pair_candidates": len(edge_pair_candidates),
        "radius_squared": maximum_radius_squared,
        "farthest_count": len(farthest_vertices),
        "farthest_example": farthest_vertices[0],
        "qhull_hints": total_hints,
        "active_size_histogram": active_size_hist,
        "edge_closure_exact": True,
        "checked_nonsimple_singletons": checked_nonsimple_singletons,
    }


# ============================================================================
# 6. ТОЧНЫЙ ПЕРЕБОР ВЕКТОРОВ ПОДРЕШЁТКИ
# ============================================================================


def enumerate_sublattice_vectors_upto_squared(
    cutoff_squared: Fr,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], Fr]:
    """Все ненулевые v=Cz с ||v||^2 <= cutoff_squared, точно."""
    sub_gram = sublattice_integer_gram()
    integer_form_bound = Fr(DENOMINATOR) * Fr(cutoff_squared)

    coefficients = enumerate_quadratic_form_exact(
        sub_gram,
        integer_form_bound,
    )

    vectors: list[tuple[int, ...]] = []
    for z in coefficients:
        vector = matrix_vector_columns(SUBLATTICE_COLUMNS, z)
        physical_norm_squared = Fr(qdot_int(vector, vector), DENOMINATOR)
        if physical_norm_squared > cutoff_squared:
            raise RuntimeError("sublattice coordinate convention mismatch")
        vectors.append(vector)

    vectors = sorted(set(vectors))
    representatives = sorted({canonical_sign(v) for v in vectors})
    return vectors, representatives, integer_form_bound


# ============================================================================
# 7. ТОЧНЫЕ KKT-ПРОЕКЦИИ; SLSQP ТОЛЬКО ОРАКУЛ
# ============================================================================


@dataclass(frozen=True)
class ProjectionCertificate:
    vector: tuple[int, ...]
    active_facets: tuple[int, ...]
    projection: tuple[Fr, ...]
    distance_squared: Fr


def exact_kkt_projection_for_active_set(
    vector: Sequence[int],
    active_ids: Sequence[int],
    fs: FacetSystem,
) -> ProjectionCertificate | None:
    """Точный KKT-сертификат проекции p=v/2 на V0."""
    active = tuple(sorted(set(map(int, active_ids))))
    if not active or len(active) > N:
        return None

    point = [Fr(int(value), 2) for value in vector]
    F = [fs.facets[i] for i in active]
    m = len(F)

    middle = [
        [qdot_int(F[i], F[j]) for j in range(m)]
        for i in range(m)
    ]
    rhs = [
        qdot_frac(F[i], point) - fs.offsets[active[i]]
        for i in range(m)
    ]

    multipliers = solve_linear_fraction(middle, rhs)
    if multipliers is None or any(mu < 0 for mu in multipliers):
        return None

    projection = [
        point[k] - sum(
            multipliers[i] * F[i][k]
            for i in range(m)
        )
        for k in range(N)
    ]

    feasible, _ = fs.exact_active_fraction(projection)
    if not feasible:
        return None

    difference = [point[k] - projection[k] for k in range(N)]

    # D(v) = 2 dist(v/2,V0), поэтому D(v)^2 = 4 ||difference||_Q^2.
    distance_squared = Fr(4) * qdot_frac(
        difference,
        difference,
    ) / DENOMINATOR

    return ProjectionCertificate(
        vector=tuple(map(int, vector)),
        active_facets=active,
        projection=tuple(projection),
        distance_squared=distance_squared,
    )


def exact_projection(
    vector: Sequence[int],
    fs: FacetSystem,
    q_float: np.ndarray,
    a_float: np.ndarray,
    b_float: np.ndarray,
) -> ProjectionCertificate:
    """Находит глобальную точную проекцию; float лишь предлагает active set."""
    vector = tuple(map(int, vector))
    point_exact = tuple(Fr(value, 2) for value in vector)

    feasible, _ = fs.exact_active_fraction(point_exact)
    if feasible:
        return ProjectionCertificate(
            vector=vector,
            active_facets=tuple(),
            projection=point_exact,
            distance_squared=Fr(0),
        )

    point_float = np.asarray(vector, dtype=float) / 2.0

    result = minimize(
        lambda x: float((x - point_float) @ q_float @ (x - point_float)),
        np.zeros(N),
        jac=lambda x: 2.0 * q_float @ (x - point_float),
        constraints=[LinearConstraint(a_float, -np.inf, b_float)],
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 3000},
    )

    if not result.success:
        raise RuntimeError(
            f"SLSQP oracle failed for {vector}: {result.message}"
        )

    slack = np.abs(b_float - a_float @ result.x)
    order = [int(i) for i in np.argsort(slack)]

    # KKT в R^N допускает сертификат максимум с N линейно независимыми
    # активными нормалями. Пробуем подмножества небольшого пула ближайших граней.
    for pool_size in (N, N + 1, N + 3, N + 5, N + 9):
        pool = sorted(set(order[:pool_size]))
        for size in range(1, min(N, len(pool)) + 1):
            for active in itertools.combinations(pool, size):
                certificate = exact_kkt_projection_for_active_set(
                    vector,
                    active,
                    fs,
                )
                if certificate is not None:
                    return certificate

    raise RuntimeError(
        f"no exact KKT certificate found for {vector}; "
        "the numerical active-set oracle was insufficient"
    )


# ============================================================================
# 8. ТОЧНЫЙ ГЛОБАЛЬНЫЙ D_min
# ============================================================================


def exact_global_minimum_same_color_distance_squared(
    diameter_squared: Fr,
    fs: FacetSystem,
    q_float: np.ndarray,
    a_float: np.ndarray,
    b_float: np.ndarray,
) -> dict:
    """Вычисляет настоящий глобальный D_min^2, а не только ell-сертификат.

    Шаг A. Перебираем ||v|| <= 2 diam. Это даёт хотя бы один точный кандидат
    D_best (для данной конструкции таких векторов достаточно много).

    Шаг B. Любой вектор, способный улучшить D_best, обязан удовлетворять
        ||v|| < D_best + diam,
    потому что D(v) >= ||v|| - diam.

    Чтобы НЕ извлекать корней, используем безопасную рациональную границу
        (D_best + diam)^2 <= 2(D_best^2 + diam^2).
    Поэтому полный второй перебор с
        ||v||^2 <= 2(D_best^2 + diam^2)
    гарантированно содержит каждый возможный глобальный минимизатор.
    """

    preliminary_cutoff_squared = 4 * diameter_squared
    prelim_vectors, prelim_reps, prelim_integer_bound = (
        enumerate_sublattice_vectors_upto_squared(preliminary_cutoff_squared)
    )
    if not prelim_reps:
        raise RuntimeError("preliminary sublattice set is empty")

    print(
        f"      preliminary set: vectors={len(prelim_vectors)}, "
        f"pairs={len(prelim_reps)}",
        flush=True,
    )

    cache: dict[tuple[int, ...], ProjectionCertificate] = {}

    for number, vector in enumerate(prelim_reps, 1):
        cert = exact_projection(vector, fs, q_float, a_float, b_float)
        cache[vector] = cert
        print(
            f"      preliminary projection {number:2d}/{len(prelim_reps)}: "
            f"v={vector}, D^2={cert.distance_squared} "
            f"(≈ {float(cert.distance_squared):.12f})",
            flush=True,
        )

    candidate_best_squared = min(c.distance_squared for c in cache.values())

    # Полная exact-граница для любого потенциального улучшателя.
    global_cutoff_squared = 2 * (
        diameter_squared + candidate_best_squared
    )

    all_vectors, all_reps, global_integer_bound = (
        enumerate_sublattice_vectors_upto_squared(global_cutoff_squared)
    )

    print(
        f"      global completeness set: vectors={len(all_vectors)}, "
        f"pairs={len(all_reps)}",
        flush=True,
    )

    for number, vector in enumerate(all_reps, 1):
        if vector not in cache:
            cert = exact_projection(vector, fs, q_float, a_float, b_float)
            cache[vector] = cert
            print(
                f"      added projection {number:2d}/{len(all_reps)}: "
                f"v={vector}, D^2={cert.distance_squared} "
                f"(≈ {float(cert.distance_squared):.12f})",
                flush=True,
            )

    relevant_cache = {v: cache[v] for v in all_reps}
    minimum_distance_squared = min(
        cert.distance_squared for cert in relevant_cache.values()
    )
    minimizers = sorted(
        v for v, cert in relevant_cache.items()
        if cert.distance_squared == minimum_distance_squared
    )

    # Строгая полнота: если ||v||^2 > global_cutoff_squared, то
    # ||v|| > D_candidate + diam и, следовательно, D(v)>D_candidate>=D_min.
    # Мы использовали ещё более широкую границу 2(a^2+b^2).

    return {
        "minimum_distance_squared": minimum_distance_squared,
        "minimizers": minimizers,
        "projection_certificates": relevant_cache,
        "preliminary_cutoff_squared": preliminary_cutoff_squared,
        "preliminary_integer_bound": prelim_integer_bound,
        "preliminary_vectors": len(prelim_vectors),
        "preliminary_pairs": len(prelim_reps),
        "candidate_best_squared": candidate_best_squared,
        "global_cutoff_squared": global_cutoff_squared,
        "global_integer_bound": global_integer_bound,
        "global_vectors": len(all_vectors),
        "global_pairs": len(all_reps),
    }


# ============================================================================
# 9. JSON HELPERS
# ============================================================================


def vertex_json(certificate: ExactVertexCertificate) -> dict:
    return {
        "active_facets": list(certificate.active_facets),
        "coordinates": [str(value) for value in certificate.coordinates()],
        "radius_squared": str(certificate.radius_squared),
    }


def projection_json(certificate: ProjectionCertificate) -> dict:
    return {
        "vector": list(certificate.vector),
        "active_facets": list(certificate.active_facets),
        "projection": [str(value) for value in certificate.projection],
        "distance_squared": str(certificate.distance_squared),
    }


# ============================================================================
# 10. ГЛАВНАЯ ПРОВЕРКА
# ============================================================================


def verify(ell: Fr = DEFAULT_ELL, json_path: Path | None = None) -> dict:
    started = time.perf_counter()
    ell = Fr(ell)

    if ell < 1:
        raise ValueError("ell must satisfy ell >= 1")

    print("=" * 78, flush=True)
    print("EXACT VERIFIER: rational R^7 lattice, index 1029", flush=True)
    print("=" * 78, flush=True)

    # ----------------------------------------------------------------------
    # 1. G > 0 и индекс
    # ----------------------------------------------------------------------
    minors = positive_definite_leading_minors()
    index = abs(det_bareiss_int(SUBLATTICE_COLUMNS))

    if index != EXPECTED_INDEX:
        raise RuntimeError(f"index {index} != {EXPECTED_INDEX}")

    print(f"[1/7] G > 0 exactly; |det C| = {index}", flush=True)
    print(f"      Q = G/{DENOMINATOR}", flush=True)

    # ----------------------------------------------------------------------
    # 2. Relevant vectors / facets
    # ----------------------------------------------------------------------
    relevant, relevant_stats = exact_relevant_vectors()
    facets = all_oriented_facets(relevant)

    if not relevant or not facets:
        raise RuntimeError("no Voronoi relevant vectors found")

    print(
        f"[2/7] exact Lambda/2Lambda: pairs={len(relevant)}, "
        f"facets={len(facets)}, "
        f"parent vectors enumerated={relevant_stats['parent_vectors_enumerated']}",
        flush=True,
    )
    print(
        "      parity classes without a unique ± shortest pair = "
        f"{len(relevant_stats['non_relevant_parity_classes'])}",
        flush=True,
    )

    fs = FacetSystem(facets)
    q_float, a_float, b_float = numerical_halfspaces(facets)

    # ----------------------------------------------------------------------
    # 3. Все вершины и точный диаметр
    # ----------------------------------------------------------------------
    geometry = build_and_certify_all_vertices(fs, a_float, b_float)

    radius_squared = Fr(geometry["radius_squared"])
    diameter_squared = 4 * radius_squared

    print(
        f"[3/7] exact geometry: vertices={geometry['vertices']}, "
        f"edge-pair candidates={geometry['edge_pair_candidates']}",
        flush=True,
    )
    print(
        f"      active-facet histogram = {geometry['active_size_histogram']}",
        flush=True,
    )
    print(f"      R^2 = {radius_squared}", flush=True)
    print(f"      diam^2 = {diameter_squared}", flush=True)
    print(
        f"      diam ≈ {math.sqrt(float(diameter_squared)):.15f} "
        "(display only)",
        flush=True,
    )

    # ----------------------------------------------------------------------
    # 4-5. Полный exact D_min
    # ----------------------------------------------------------------------
    distance_data = exact_global_minimum_same_color_distance_squared(
        diameter_squared,
        fs,
        q_float,
        a_float,
        b_float,
    )

    minimum_distance_squared = Fr(distance_data["minimum_distance_squared"])
    minimizers = distance_data["minimizers"]

    print(f"[4/7] global finite set certified", flush=True)
    print(
        f"      vectors={distance_data['global_vectors']}, "
        f"pairs={distance_data['global_pairs']}",
        flush=True,
    )
    print(
        f"      cutoff^2 = {distance_data['global_cutoff_squared']}",
        flush=True,
    )

    print(f"[5/7] D_min^2 = {minimum_distance_squared}", flush=True)
    print(
        f"      D_min ≈ {math.sqrt(float(minimum_distance_squared)):.15f} "
        "(display only)",
        flush=True,
    )
    print(f"      minimizer(s) = {minimizers}", flush=True)

    # ----------------------------------------------------------------------
    # 6. Точное нормированное расстояние
    # ----------------------------------------------------------------------
    normalized_distance_squared = minimum_distance_squared / diameter_squared
    normalized_distance_display = math.sqrt(float(normalized_distance_squared))

    print(
        f"[6/7] d^2 = D_min^2/diam^2 = {normalized_distance_squared}",
        flush=True,
    )
    print(
        f"      d ≈ {normalized_distance_display:.15f} (display only)",
        flush=True,
    )
    print(
        f"      exact d^2 > 1 : {normalized_distance_squared > 1}",
        flush=True,
    )

    # ----------------------------------------------------------------------
    # 7. Точный закрытый интервал [1, ell]
    # ----------------------------------------------------------------------
    margin = minimum_distance_squared - ell * ell * diameter_squared
    interval_valid = margin > 0

    print(
        f"[7/7] ell = {ell}; exact margin "
        f"D_min^2-ell^2*diam^2 = {margin}",
        flush=True,
    )
    print(f"      CLOSED interval valid = {interval_valid}", flush=True)

    elapsed = time.perf_counter() - started

    projection_certificates = distance_data["projection_certificates"]

    result = {
        "dimension": N,
        "index": index,
        "gram_denominator": DENOMINATOR,
        "integer_gram": INTEGER_GRAM,
        "sublattice_columns": SUBLATTICE_COLUMNS,
        "ell": str(ell),
        "positive_definite_leading_minors": minors,
        "relevant_pairs": len(relevant),
        "facets": len(facets),
        "parent_vectors_enumerated_for_relevant_search": (
            relevant_stats["parent_vectors_enumerated"]
        ),
        "non_relevant_parity_classes": (
            relevant_stats["non_relevant_parity_classes"]
        ),
        "vertices": int(geometry["vertices"]),
        "edge_pair_candidates": int(geometry["edge_pair_candidates"]),
        "qhull_hints": int(geometry["qhull_hints"]),
        "active_size_histogram": {
            str(k): int(v)
            for k, v in geometry["active_size_histogram"].items()
        },
        "edge_closure_exact": bool(geometry["edge_closure_exact"]),
        "radius_squared_exact": str(radius_squared),
        "diameter_squared_exact": str(diameter_squared),
        "diameter_float_display": math.sqrt(float(diameter_squared)),
        "farthest_vertex_count": int(geometry["farthest_count"]),
        "farthest_vertex_example": vertex_json(geometry["farthest_example"]),
        "preliminary_vectors": int(distance_data["preliminary_vectors"]),
        "preliminary_pairs": int(distance_data["preliminary_pairs"]),
        "preliminary_cutoff_squared_exact": str(
            distance_data["preliminary_cutoff_squared"]
        ),
        "global_vectors": int(distance_data["global_vectors"]),
        "global_pairs": int(distance_data["global_pairs"]),
        "global_cutoff_squared_exact": str(
            distance_data["global_cutoff_squared"]
        ),
        "minimum_distance_squared_exact": str(minimum_distance_squared),
        "minimum_distance_float_display": math.sqrt(
            float(minimum_distance_squared)
        ),
        "minimizers": [list(v) for v in minimizers],
        "normalized_distance_squared_exact": str(normalized_distance_squared),
        "normalized_distance_float_display": normalized_distance_display,
        "margin_exact": str(margin),
        "interval_valid": bool(interval_valid),
        "projection_certificates": [
            projection_json(projection_certificates[v])
            for v in sorted(projection_certificates)
        ],
        "elapsed_seconds": elapsed,
    }

    if json_path is not None:
        Path(json_path).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON certificate written to: {json_path}", flush=True)

    print(f"Elapsed: {elapsed:.2f} s", flush=True)
    return result


# ============================================================================
# 11. CLI
# ============================================================================


def parse_rational(text: str) -> Fr:
    try:
        return Fr(text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "ell must be rational/decimal, e.g. 1.03 or 103/100"
        ) from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ell",
        type=parse_rational,
        default=DEFAULT_ELL,
        help="right endpoint of the closed forbidden interval [1,ell]",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="куда писать сертификат (по умолчанию results/dim7_1029_exact.json)",
    )
    args = parser.parse_args(argv)

    if args.ell < 1:
        parser.error("ell must satisfy ell >= 1")

    result = verify(args.ell, args.output or results_path("dim7_1029_exact.json"))
    return 0 if result["interval_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
