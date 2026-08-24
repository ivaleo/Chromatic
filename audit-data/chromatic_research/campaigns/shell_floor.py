"""Оболочечный пол: сколько цветов метод НЕ может опустить на данном родителе.

Идея.  Прежние полы проекта (объёмный и инрадиусный) останавливаются на
неравенстве ``λ₁(Γ) ≥ 2R + λ₁``.  Но норма вектора решётки пробегает не
континуум, а ОБОЛОЧКИ, и первая оболочка над ``(2R+λ₁)²`` часто оказывается
запрещённой целиком.  Тогда ``λ₁(Γ)²`` скачком поднимается до следующей
оболочки, и неравенство Эрмита даёт заметно больший пол.

Запрет оболочки доказывается радиальным свидетелем.  Для ``|v|² = N`` берём
точку ``x = s·v`` с рациональным ``s`` и проверяем ``x ∈ V₀``, то есть

    s·⟨v, u⟩ ≤ |u|²/2   для всех u ∈ Λ \\ {0}.

По Коши–Шварцу ``⟨v,u⟩ ≤ √(N·|u|²)``, поэтому условие выполняется автоматически,
как только ``|u|² ≥ 4s²N``.  Остаются КОНЕЧНО МНОГО малых оболочек, и на них
работает целочисленность: скалярные произведения решётки лежат в ``(1/e)ℤ``,
поэтому ``⟨v,u⟩`` не превосходит наибольшего кратного ``1/e``, не большего
``√(N·|u|²)``.  Тогда

    D(v) = 2·dist(v/2, V₀) ≤ 2|v/2 − s v| = 2(1/2 − s)√N,

и если это меньше ``diam = 2R``, вся оболочка запрещена.

Доказательство получается проверяемым вручную: ни одного перебора векторов,
только Коши–Шварц и целочисленность.

Результаты (подробности — в выводе программы):

    Λ = E₈   : оболочка 12 запрещена (D ≤ √3 < 2)  ⇒  λ₁(Γ)² ≥ 14
               ⇒  k ≥ (14/2)⁴ = 2401 = РОВНО рекорд ⇒ 2401 ОПТИМАЛЕН
    Λ = E₆*  : оболочка 8 запрещена (D² ≤ 200/81 < 8/3) ⇒ λ₁(Γ)² ≥ 28/3
               ⇒  k ≥ 2744/9 ⇒ k ≥ 305 (рекорд 343, окно [305, 342])

Usage::

    python -m chromatic_research.campaigns.shell_floor
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from fractions import Fraction as Fr
from pathlib import Path

from chromatic_research.core.exact_layer_cert import enumerate_shifted
from chromatic_research.paths import results_path

# Целочисленный грам E₈ (min 2, det 1).
E8_GRAM = [
    [2, -1, 0, 0, 0, 0, 0, 0],
    [-1, 2, -1, 0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0, 0, 0, -1],
    [0, 0, -1, 2, -1, 0, 0, 0],
    [0, 0, 0, -1, 2, -1, 0, 0],
    [0, 0, 0, 0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0, -1, 2, 0],
    [0, 0, -1, 0, 0, 0, 0, 2],
]


@dataclass(frozen=True)
class Parent:
    """Родительская решётка с точными инвариантами."""

    name: str
    gram: tuple[tuple[Fr, ...], ...]
    lambda1_sq: Fr
    covering_sq: Fr                  # R²
    ip_denominator: int              # ⟨Λ,Λ⟩ ⊆ (1/e)ℤ
    hermite_power: Fr                # γ_n^n · (det Λ)² — всегда рационально
    hermite_note: str
    record_index: int
    record_name: str
    dimension: int = 0               # если gram пуст — задаётся явно
    analytic_shells: tuple[Fr, ...] = ()   # спектр норм, если перечислять нельзя
    spectrum_note: str = ""
    conditional: str = ""            # на чём условен результат (пусто = ни на чём)

    @property
    def dim(self) -> int:
        return self.dimension or len(self.gram)

    @property
    def diam_sq(self) -> Fr:
        return 4 * self.covering_sq


def dot(gram, u, v) -> Fr:
    size = len(gram)
    return sum(Fr(u[i]) * gram[i][j] * Fr(v[j])
               for i in range(size) for j in range(size))


def shells_up_to(parent: Parent, bound: Fr) -> dict[Fr, int]:
    """Спектр норм до ``bound``: перечислением либо из аналитического списка.

    В размерностях 12 и 24 перечислять нечего: число векторов оболочки там
    астрономическое, зато сам СПЕКТР известен (чётные нормы от минимума), а
    доказательству нужен только он — граница Коши–Шварца с целочисленностью
    векторов не считает.
    """
    if parent.analytic_shells:
        return {value: 0 for value in parent.analytic_shells if value <= bound}
    counts: dict[Fr, int] = {}
    zero = [Fr(0)] * parent.dim
    for v in enumerate_shifted(parent.gram, zero, bound):
        if not any(v):
            continue
        value = dot(parent.gram, v, v)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def max_inner_product(norm_v: Fr, norm_u: Fr, denominator: int) -> Fr:
    """Наибольшее кратное 1/e, не превосходящее √(N·M) — граница Коши–Шварца."""
    product = norm_v * norm_u
    scaled = product * denominator * denominator          # (e√(NM))²
    root = math.isqrt(int(scaled.numerator) // int(scaled.denominator))
    while Fr((root + 1) ** 2, denominator ** 2) <= product:
        root += 1
    while Fr(root ** 2, denominator ** 2) > product:
        root -= 1
    return Fr(root, denominator)


def radial_witness(parent: Parent, norm_v: Fr, shells: dict[Fr, int]):
    """Рациональное ``s`` со свидетелем ``s·v ∈ V₀`` и оценкой ``D²``.

    Перебираются знаменатели; берётся ``s``, дающее наилучшую (наименьшую)
    оценку ``D²``.  Возвращает ``None``, если оболочку так закрыть не удалось.
    """
    best = None
    for denominator in range(2, 60):
        for numerator in range(1, denominator):
            s = Fr(numerator, denominator)
            if 2 * s >= 1:
                continue
            limit = 4 * s * s * norm_v
            ok = True
            for norm_u in shells:
                if norm_u >= limit:
                    break                     # дальше Коши–Шварц закрывает сам
                bound = max_inner_product(norm_v, norm_u,
                                          parent.ip_denominator)
                if s * bound > norm_u / 2:
                    ok = False
                    break
            if not ok:
                continue
            distance_sq = 4 * (Fr(1, 2) - s) ** 2 * norm_v
            if distance_sq >= parent.diam_sq:
                continue
            if best is None or distance_sq < best[1]:
                best = (s, distance_sq, limit)
    return best


def analyse(parent: Parent) -> dict:
    lam1 = math.sqrt(float(parent.lambda1_sq))
    radius = math.sqrt(float(parent.covering_sq))
    low = (2 * radius + lam1) ** 2
    high = parent.diam_sq                     # (2·diam)²/... см. ниже
    ceiling = 4 * parent.diam_sq              # |v| ≥ 2·diam разрешён всегда
    shells = shells_up_to(parent, ceiling)

    print(f"\n=== {parent.name} (ℝ^{parent.dim}) ===")
    print(f"λ₁² = {parent.lambda1_sq}, R² = {parent.covering_sq}, "
          f"diam² = {parent.diam_sq}")
    print(f"инрадиусная лемма: λ₁(Γ)² ≥ (2R+λ₁)² = {low:.6f}")
    window = {k: v for k, v in shells.items() if float(k) >= low - 1e-12}
    if parent.spectrum_note:
        print(f"спектр норм: {parent.spectrum_note}")
    print(f"оболочки в окне [{low:.4f}, {float(ceiling):.4f}]:")
    for value, count in window.items():
        tail = f": {count} векторов" if count else ""
        print(f"    |v|² = {value} = {float(value):.5f}{tail}")

    forbidden = []
    first_allowed = None
    for value in window:
        witness = radial_witness(parent, value, shells)
        if witness is None:
            first_allowed = value
            break
        s, distance_sq, limit = witness
        forbidden.append({
            "shell": str(value), "s": str(s),
            "d2_bound": str(distance_sq),
            "d2_bound_float": float(distance_sq),
            "cauchy_schwarz_covers_from": str(limit),
        })
        print(f"    оболочка {value} ЗАПРЕЩЕНА: свидетель s = {s}, "
              f"D² ≤ {distance_sq} = {float(distance_sq):.6f} "
              f"< {parent.diam_sq} = {float(parent.diam_sq):.6f}")
    if first_allowed is None:
        first_allowed = ceiling
    print(f"    ⇒ λ₁(Γ)² ≥ {first_allowed} = {float(first_allowed):.6f}")

    # Эрмит: λ₁² ≤ γ_n (det Γ)^{2/n}, det Γ = k·det Λ.  Возведение в степень n
    # убирает корни: k² ≥ (λ₁²)^n / (γ_n^n · (det Λ)²) — всё рационально при
    # любой чётности размерности.
    squared = first_allowed ** parent.dim / parent.hermite_power
    bound = math.sqrt(float(squared))
    floor_index = math.ceil(bound - 1e-12)
    print(f"    Эрмит ({parent.hermite_note}): "
          f"k² ≥ ({first_allowed})^{parent.dim} / {parent.hermite_power} "
          f"= {squared} ⇒ k ≥ {bound:.4f}  ⇒  k ≥ {floor_index}")
    print(f"    рекорд: {parent.record_index} ({parent.record_name})")
    if parent.conditional:
        print(f"    ВНИМАНИЕ: результат условен — {parent.conditional}")
    if floor_index >= parent.record_index:
        print(f"    *** ПОЛ СОВПАЛ С РЕКОРДОМ: {parent.record_index} ОПТИМАЛЕН "
              f"на этом родителе ***")
    else:
        print(f"    окно возможных улучшений: "
              f"[{floor_index}, {parent.record_index - 1}]; запас "
              f"{parent.record_index / floor_index:.2f}×")

    return {
        "parent": parent.name, "dim": parent.dim,
        "lambda1_sq": str(parent.lambda1_sq),
        "covering_sq": str(parent.covering_sq),
        "inradius_bound": low,
        "shells_in_window": {str(k): v for k, v in window.items()},
        "forbidden_shells": forbidden,
        "first_allowed_shell": str(first_allowed),
        "hermite_squared_bound": str(squared),
        "hermite_denominator": str(parent.hermite_power),
        "hermite_note": parent.hermite_note,
        "index_floor": floor_index,
        "record_index": parent.record_index,
        "record_is_optimal": floor_index >= parent.record_index,
        "conditional": parent.conditional,
    }


def e6_star_parent() -> Parent:
    from chromatic_research.campaigns.dim6_e6star_floor import dual_gram

    gram = tuple(tuple(row) for row in dual_gram())
    # γ₆⁶ = 64/3, (det Λ)² = 1/3  ⇒  произведение 64/9
    return Parent("E6*", gram, Fr(4, 3), Fr(2, 3), 3, Fr(64, 9),
                  "γ₆⁶·(detΛ)² = (64/3)·(1/3) = 64/9", 343, "(3+ω)E₆*")


def e8_parent() -> Parent:
    gram = tuple(tuple(Fr(x) for x in row) for row in E8_GRAM)
    # γ₈⁸ = 2⁸ = 256, (det Λ)² = 1
    return Parent("E8", gram, Fr(2), Fr(1), 1, Fr(256),
                  "γ₈⁸·(detΛ)² = 2⁸·1 = 256", 2401, "(3+ω)E₈")


def a3_star_parent() -> Parent:
    """A₃* = ОЦК: базис (±½,±½,±½)-типа, λ₁² = 3/4, R² = 5/16, det = 1/2."""
    # Базис ОЦК: e₁ = (1,0,0), e₂ = (0,1,0), e₃ = (½,½,½).
    gram = (
        (Fr(1), Fr(0), Fr(1, 2)),
        (Fr(0), Fr(1), Fr(1, 2)),
        (Fr(1, 2), Fr(1, 2), Fr(3, 4)),
    )
    # γ₃³ = 2, (det Λ)² = 1/4  ⇒  произведение 1/2
    return Parent("A3* (ОЦК)", gram, Fr(3, 4), Fr(5, 16), 4, Fr(1, 2),
                  "γ₃³·(detΛ)² = 2·(1/4) = 1/2", 15, "решётчатая раскраска Кулсона")


def k12_parent() -> Parent:
    """K₁₂ (Коксетер–Тодд): λ₁² = 4, 2R/λ₁ = √(8/3) ⇒ R² = 8/3, det Λ = 27.

    Чётная решётка с минимумом 4, поэтому спектр норм — чётные ≥ 4.  Перечислять
    оболочки в ℝ¹² бессмысленно (их размеры астрономические), а нужен только
    спектр.  Результат УСЛОВЕН: точное значение γ₁₂ неизвестно, берётся
    гипотетическое γ₁₂ = 4/√3, достигаемое на самой K₁₂.
    """
    # γ₁₂¹²·(det Λ)² = (4/√3)¹²·27² = (4¹²/3⁶)·3⁶ = 4¹² — тройки сокращаются
    return Parent("K12", (), Fr(4), Fr(8, 3), 1, Fr(4) ** 12,
                  "γ₁₂¹²·(detΛ)² = (4/√3)¹²·27² = 4¹²", 3 ** 12, "3·K₁₂",
                  dimension=12,
                  analytic_shells=tuple(Fr(2 * m) for m in range(2, 40)),
                  spectrum_note="чётная, минимум 4 ⇒ нормы 4, 6, 8, …",
                  conditional="γ₁₂ = 4/√3 (оптимальность K₁₂ в ℝ¹² открыта)")


def leech_parent() -> Parent:
    """Λ₂₄ (Лич): λ₁² = 4, R² = 2, det Λ = 1; γ₂₄ = 4 ДОКАЗАНА (Кон–Кумар).

    Спектр — чётные нормы ≥ 4.  Результат безусловен.
    """
    # γ₂₄²⁴·(det Λ)² = 4²⁴·1
    return Parent("Leech Λ24", (), Fr(4), Fr(2), 1, Fr(4) ** 24,
                  "γ₂₄²⁴·(detΛ)² = 4²⁴·1", 7 ** 12, "(3+ω)Λ₂₄",
                  dimension=24,
                  analytic_shells=tuple(Fr(2 * m) for m in range(2, 40)),
                  spectrum_note="чётная, минимум 4 ⇒ нормы 4, 6, 8, …")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    reports = [analyse(e8_parent()), analyse(e6_star_parent()),
               analyse(a3_star_parent()), analyse(k12_parent()),
               analyse(leech_parent())]

    out = args.output or results_path("shell_floor.json")
    out.write_text(json.dumps({"parents": reports}, indent=1,
                              ensure_ascii=False) + "\n")
    print(f"\nсохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
