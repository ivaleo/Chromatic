"""Доказанный пол для родителя ``E₆*`` в ℝ⁶: любая пригодная подрешётка имеет
индекс не меньше **305**.

Что доказывается.  Пусть ``Λ = E₆*`` (нормировка ``λ₁² = 4/3``,
``R_cov² = 2/3``, ``det Λ = 3^{-1/2}``) и ``Γ ⊆ Λ`` — подрешётка, задающая
корректную раскраску при ``ℓ = 1``, то есть ``D(v) ≥ diam V₀ = 2R`` для всех
``v ∈ Γ \\ {0}``.  Тогда ``[Λ : Γ] ≥ 305``.

Действующий рекорд — ``Γ = (3+ω)E₆*`` с индексом 343.  Значит на этом родителе
методу осталось не больше 11 %, а окно возможных улучшений — ровно
``[305, 342]``.

Доказательство в четыре шага.

1. **Инрадиусная лемма** ``D(v) ≤ |v| − λ₁`` (из ``B(0, λ₁/2) ⊆ V₀``) даёт
   ``|v| ≥ 2R + λ₁`` для всех ``v ∈ Γ\\{0}``, то есть
   ``λ₁(Γ)² ≥ (2R + λ₁)² = 4 + 8√2/3 = 7.7712…``

2. **Оболочка ``|v|² = 8`` запрещена ЦЕЛИКОМ.**  Это единственная оболочка
   ``E₆*`` в промежутке ``[(2R+λ₁)², 28/3)``, и все её 468 векторов имеют
   ``D(v) < 2R``.  Для каждого предъявляется ТОЧНЫЙ рациональный свидетель:
   точка ``x ∈ V₀`` с ``4|v/2 − x|² < 8/3``.  Принадлежность ``x ∈ V₀``
   проверяется всеми 126 опорными неравенствами в дробях, неравенство на
   расстояние — тоже в дробях.  Ни одного округления.

3. Следовательно ``λ₁(Γ)² ≥ 28/3`` (ближайшая сверху оболочка).

4. **Неравенство Эрмита** ``λ₁(Γ)² ≤ γ₆ (det Γ)^{1/3}`` с
   ``γ₆ = (64/3)^{1/6}`` (Блихфельдт; максимум достигается на ``E₆``) и
   ``det Γ = k·3^{-1/2}`` даёт

       (28/3)³ ≤ γ₆³ · k·3^{-1/2} = (8/√3)(k/√3) = 8k/3,
       k ≥ 21952·3/(27·8) = 21952/72 = 304.888…,

   то есть ``k ≥ 305``.  ∎

Почему это сильнее прежних полов.  Композиция «инрадиус + упаковка» из
``core/paradigm_floor.py`` даёт для ``E₆*`` только ``k ≥ 176``: она
останавливается на шаге 1.  Весь выигрыш — в шаге 2, то есть в том, что
запрещённой оказывается ЦЕЛАЯ оболочка, а не отдельные векторы.

Usage::

    python -m chromatic_research.campaigns.dim6_e6star_floor
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np

from chromatic_research.core.exact_layer_cert import enumerate_shifted
from chromatic_research.paths import results_path

# Целочисленная матрица Грама E₆ (минимум 2, det 3).
E6_GRAM = [
    [2, 0, -1, 0, 0, 0],
    [0, 2, 0, -1, 0, 0],
    [-1, 0, 2, -1, 0, 0],
    [0, -1, -1, 2, -1, 0],
    [0, 0, 0, -1, 2, -1],
    [0, 0, 0, 0, -1, 2],
]
DIM = 6

# Точные инварианты E₆* = (E₆)*: грамиан = E6_GRAM^{-1}.
LAMBDA1_SQ = Fr(4, 3)
COVERING_SQ = Fr(2, 3)          # R²
DIAM_SQ = Fr(8, 3)              # (2R)²
SHELL_SQ = Fr(8)                # запрещаемая оболочка
NEXT_SHELL_SQ = Fr(28, 3)       # первая разрешённая оболочка
CHAMPION_INDEX = 343


def dual_gram() -> list[list[Fr]]:
    """Грамиан ``E₆*`` в точных дробях (обратная к целочисленной ``E₆``)."""
    size = DIM
    work = [[Fr(E6_GRAM[i][j]) for j in range(size)]
            + [Fr(1) if i == j else Fr(0) for j in range(size)]
            for i in range(size)]
    for col in range(size):
        pivot = next(r for r in range(col, size) if work[r][col] != 0)
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [v / scale for v in work[col]]
        for row in range(size):
            if row != col and work[row][col] != 0:
                factor = work[row][col]
                work[row] = [work[row][k] - factor * work[col][k]
                             for k in range(2 * size)]
    return [[work[i][size + j] for j in range(size)] for i in range(size)]


def dot(gram, u, v) -> Fr:
    return sum(Fr(u[i]) * gram[i][j] * Fr(v[j])
               for i in range(DIM) for j in range(DIM))


def relevant_vectors(gram) -> list[tuple[int, ...]]:
    """Векторы ``|v|² ≤ 4R²`` — надмножество релевантных, задаёт ``V₀`` точно."""
    return [v for v in enumerate_shifted(gram, [Fr(0)] * DIM, 4 * COVERING_SQ)
            if any(v)]


def shell(gram, norm_squared: Fr) -> list[tuple[int, ...]]:
    return [v for v in enumerate_shifted(gram, [Fr(0)] * DIM, norm_squared)
            if any(v) and dot(gram, v, v) == norm_squared]


def shells_in_window(gram, low: float, high: Fr) -> dict:
    """Все оболочки с ``low ≤ |v|² ≤ high`` и их размеры."""
    counts: dict[Fr, int] = {}
    for v in enumerate_shifted(gram, [Fr(0)] * DIM, high):
        if not any(v):
            continue
        value = dot(gram, v, v)
        if float(value) < low - 1e-12:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def exact_witness(gram, facets, vector, denominator: int = 2520):
    """Рациональная точка ``x ∈ V₀`` с ``4|v/2 − x|² < 8/3``, либо ``None``.

    Оракул — численная проекция; принятая точка полностью пересчитывается
    в дробях, поэтому ошибка оракула стоит неудачи попытки, но не корректности.
    """
    from scipy.optimize import linprog

    # Работаем прямо в координатах базиса: неравенства ⟨x, a⟩ ≤ |a|²/2 линейны
    # по координатам, а квадрат расстояния — квадратичная форма с грамом.
    matrix = np.array([[float(v) for v in row] for row in gram])
    rows = np.array([[float(sum(gram[i][j] * Fr(normal[j]) for j in range(DIM)))
                      for i in range(DIM)] for normal, _ in facets])
    offsets = np.array([float(value) / 2 for _, value in facets])
    half_float = np.array([float(c) / 2 for c in vector])

    # ближайшая точка ячейки — задача наименьших квадратов с линейными
    # ограничениями; решаем последовательными ЛП-шагами Франка–Вульфа с
    # точным одномерным минимумом (устойчиво и без внешних QP-решателей)
    point = np.zeros(DIM)                      # 0 ∈ V₀ всегда
    for _ in range(300):
        residual = point - half_float
        gradient = 2.0 * matrix @ residual
        step = linprog(gradient, A_ub=rows, b_ub=offsets,
                       bounds=[(None, None)] * DIM, method="highs")
        if not step.success:
            break
        direction = step.x - point
        denominator_step = float(direction @ matrix @ direction)
        if denominator_step <= 1e-18:
            break
        alpha = -float(residual @ matrix @ direction) / denominator_step
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha <= 1e-14:
            break
        point = point + alpha * direction

    # Проекция лежит НА границе ячейки, поэтому округление её выталкивает
    # наружу.  Сдвигаем точку внутрь к нулю (``0 ∈ V₀`` всегда, ячейка
    # выпукла, значит ``(1−ε)x ∈ V₀``); запас по расстоянию это позволяет.
    half = [Fr(c, 2) for c in vector]
    for shrink in (0, 1, 3, 10, 30, 100, 300):
        scale = 1.0 - shrink / 1000.0
        rational = [Fr(round(value * scale * denominator), denominator)
                    for value in point]
        if any(dot(gram, rational, normal) * 2 > norm_squared
               for normal, norm_squared in facets):
            continue
        delta = [half[i] - rational[i] for i in range(DIM)]
        distance_squared = 4 * dot(gram, delta, delta)
        if distance_squared < DIAM_SQ:
            return rational, distance_squared
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--denominator", type=int, default=2520)
    args = parser.parse_args(argv)

    gram = dual_gram()
    print("Λ = E₆*: λ₁² =", dot(gram, [1] + [0] * 5, [1] + [0] * 5),
          "(ожидается 4/3 после приведения)")
    facet_vectors = relevant_vectors(gram)
    facets = [(v, dot(gram, v, v)) for v in facet_vectors]
    print(f"опорных полупространств (надмножество релевантных): {len(facets)}")

    low = float((2 * math.sqrt(float(COVERING_SQ)) + math.sqrt(float(LAMBDA1_SQ))) ** 2)
    window = shells_in_window(gram, low, 4 * DIAM_SQ)
    print(f"окно норм² = [{low:.6f}, {float(4*DIAM_SQ):.6f}]; оболочки:")
    for value, count in window.items():
        print(f"    |v|² = {value} = {float(value):.6f} : {count} векторов")

    target = shell(gram, SHELL_SQ)
    print(f"\nоболочка |v|² = {SHELL_SQ}: {len(target)} векторов — "
          f"строим точные свидетели запрета")
    witnesses = []
    failures = []
    worst = Fr(0)
    for vector in target:
        found = exact_witness(gram, facets, vector, args.denominator)
        if found is None:
            failures.append(list(vector))
            continue
        point, distance_squared = found
        witnesses.append({"v": list(vector),
                          "d2_upper": [int(distance_squared.numerator),
                                       int(distance_squared.denominator)]})
        worst = max(worst, distance_squared)
    print(f"свидетелей построено: {len(witnesses)}/{len(target)}; "
          f"худшая верхняя оценка D² = {worst} = {float(worst):.6f} "
          f"< {DIAM_SQ} = {float(DIAM_SQ):.6f}")
    if failures:
        print(f"НЕ ЗАКРЫТО векторов: {len(failures)} — доказательство неполно")

    # шаг 4: Эрмит
    bound = Fr(28, 3) ** 3 * 3 / 8
    floor_index = math.ceil(float(bound))
    print(f"\nЭрмит: (28/3)³ ≤ (8/√3)·k·3^(−1/2) = 8k/3  ⇒  "
          f"k ≥ {bound} = {float(bound):.6f}  ⇒  k ≥ {floor_index}")
    print(f"рекорд на этом родителе: {CHAMPION_INDEX} ((3+ω)E₆*)")
    print(f"ОКНО ВОЗМОЖНЫХ УЛУЧШЕНИЙ: [{floor_index}, {CHAMPION_INDEX - 1}] "
          f"— не более {100*(1 - floor_index/CHAMPION_INDEX):.1f} % по числу цветов")

    payload = {
        "parent": "E6*",
        "lambda1_sq": str(LAMBDA1_SQ), "covering_sq": str(COVERING_SQ),
        "diam_sq": str(DIAM_SQ), "det": "3^(-1/2)",
        "window_low": low,
        "shells_in_window": {str(k): v for k, v in window.items()},
        "forbidden_shell": str(SHELL_SQ),
        "forbidden_shell_size": len(target),
        "witnesses_built": len(witnesses),
        "witness_failures": failures,
        "worst_witness_d2": str(worst),
        "first_allowed_shell": str(NEXT_SHELL_SQ),
        "hermite_bound": str(bound),
        "index_floor": floor_index,
        "champion_index": CHAMPION_INDEX,
        "open_window": [floor_index, CHAMPION_INDEX - 1],
        "proved": not failures,
    }
    out = args.output or results_path("dim6_e6star_floor.json")
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"сохранено: {out}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
