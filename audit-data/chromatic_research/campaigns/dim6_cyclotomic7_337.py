"""ℝ⁶, индекс 337: полный поиск в циклотомическом семействе ℤ[ζ₇].

Замысел.  Рекорд ℝ⁴ = 43 нашёлся не перебором по всем девяти параметрам формы,
а СИМMЕТРИЙНЫМ сужением: решётка оказалась ℤ[ω]-модулем, и 43 = N(7+ω).  Здесь
тот же приём в размерности 6 на индексе 337.  Простое 337 ≡ 1 (mod 7), поэтому
оно ПОЛНОСТЬЮ расщепляется в ℚ(ζ₇), и у ℤ[ζ₇] ровно шесть идеалов нормы 337.

Модель.  ``Λ = ℤ[ζ₇]`` (ранг 1 над кольцом целых, вещественная размерность 6) с
инвариантной следовой формой

    Q_α(x) = Tr_{K/ℚ}(α x x̄),   α ∈ K⁺ = ℚ(ζ₇)⁺ вполне положительно.

Матрица Грама в базисе ``1, ζ, …, ζ⁵`` имеет вид ``G_ij = m_{|i-j|}``, где
``m_k = Tr_{K⁺/ℚ}(α η_k)``, ``η_k = ζ^k + ζ^{-k}``; из ``Σ_{k=1}^{6} ζ^k = -1``
следует ``m_1 + m_2 + m_3 = -m_0/2``.  Итого ТРИ параметра, а после исключения
масштаба — **два**.  Это на порядок меньше, чем 20 существенных параметров
общей шестимерной формы.

Полнота поиска.  Замена ``α ↦ α u ū`` для единицы ``u ∈ ℤ[ζ₇]`` даёт ту же
решётку с точностью до изометрии (``Q_{αuū}(x) = Q_α(ux)``), а по теореме
Куммера для простого проводника ``E = W · E⁺``, поэтому действует ровно группа
квадратов вещественных единиц ``(E⁺)²``.  В логарифмическом вложении это
решётка сдвигов двумерной плоскости ``Σ log α_j = 0``; сканируется её
фундаментальный параллелограмм (с запасом), то есть покрывается ВСЁ семейство,
а не его окрестность.  Группа Галуа переставляет шесть идеалов вместе с
сопряжениями ``α``, поэтому проверяются все шесть, а область по ``α`` берётся
одна.

Что означает отрицательный результат.  Закрыто будет всё двупараметрическое
семейство целиком, а не одна метрика: это утверждение о классе конструкций.

Usage::

    python -m chromatic_research.campaigns.dim6_cyclotomic7_337 --grid 60
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from chromatic_research.core.eisenstein4 import evaluate
from chromatic_research.paths import results_path

P = 7
INDEX = 337
DIM = P - 1

# Сопряжения θ = ζ + ζ^{-1}: θ_j = 2 cos(2πj/7), j = 1, 2, 3.
THETA = np.array([2 * math.cos(2 * math.pi * j / P) for j in (1, 2, 3)])
# η_k в трёх вещественных вложениях: η_k^{(j)} = 2 cos(2π k j / 7).
ETA = np.array([[2 * math.cos(2 * math.pi * k * j / P) for j in (1, 2, 3)]
                for k in range(4)])

# Циклотомические единицы ξ_a = sin(2πa/7)/sin(2π/7), a = 2, 3 — фундаментальные
# для ℚ(ζ₇)⁺ (число классов h⁺ = 1).  Действует группа их КВАДРАТОВ.
def _unit_logs() -> np.ndarray:
    rows = []
    for a in (2, 3):
        conj = np.array([math.sin(2 * math.pi * a * j / P)
                         / math.sin(2 * math.pi * j / P) for j in (1, 2, 3)])
        rows.append(2.0 * np.log(np.abs(conj)))     # квадрат единицы
    return np.array(rows)


UNIT_LOGS = _unit_logs()


def gram_from_alpha(alpha: np.ndarray) -> np.ndarray:
    """Матрица Грама Q_α в базисе 1, ζ, …, ζ⁵ (шесть на шесть)."""
    m = ETA @ alpha                       # m_k, k = 0..3
    gram = np.empty((DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            k = (i - j) % P
            gram[i, j] = m[min(k, P - k)]
    return 0.5 * (gram + gram.T)


def prime_ideals_of_norm(modulus: int = INDEX) -> list[np.ndarray]:
    """ℤ-базисы шести простых идеалов нормы 337 (строки — координаты в ℤ[ζ])."""
    roots = [r for r in range(1, modulus)
             if pow(r, P, modulus) == 1 and r != 1]
    assert len(roots) == P - 1, f"ожидалось {P-1} корней, найдено {len(roots)}"
    bases = []
    for r in roots:
        # p = ker(ℤ[ζ] → F_337, ζ ↦ r): x = Σ x_i ζ^i ↦ Σ x_i r^i
        weights = [pow(r, i, modulus) for i in range(DIM)]
        rows = [[modulus if i == 0 else 0 for i in range(DIM)]]
        for i in range(1, DIM):
            # ζ^i - r^i лежит в ядре
            row = [0] * DIM
            row[i] = 1
            row[0] = -weights[i]
            rows.append(row)
        basis = _hermite_normal_form(rows, DIM)
        det = round(abs(np.linalg.det(np.array(basis, float))))
        assert det == modulus, f"индекс {det} вместо {modulus}"
        bases.append(np.array(basis, dtype=int))
    return bases


def _hermite_normal_form(rows, dim):
    """HNF (нижнетреугольная) целочисленной системы образующих."""
    work = [list(map(int, row)) for row in rows]
    basis = [[0] * dim for _ in range(dim)]
    for col in range(dim):
        while True:
            nonzero = [r for r in work if r[col] != 0]
            if len(nonzero) <= 1:
                break
            nonzero.sort(key=lambda r: abs(r[col]))
            pivot = nonzero[0]
            for row in nonzero[1:]:
                factor = row[col] // pivot[col]
                for k in range(dim):
                    row[k] -= factor * pivot[k]
        nonzero = [r for r in work if r[col] != 0]
        if nonzero:
            basis[col] = list(nonzero[0])
            work = [r for r in work if r is not nonzero[0]]
            for row in work:
                if row[col]:
                    factor = row[col] // basis[col][col]
                    for k in range(dim):
                        row[k] -= factor * basis[col][k]
    return basis


def covering_radius(facets, dim: int, directions: int, seed: int) -> float:
    """R_cov ячейки, заданной опорными полупространствами.

    Максимум ``|x|`` по ячейке достигается в вершине; каждое направление даёт
    вершину (ЛП), а повтор с направлением ``x/|x|`` быстро сходится к локально
    самой дальней. Оценка идёт СНИЗУ, поэтому ``d = D/(2R)`` получается сверху —
    для экрана «порог не взят» это безопасная сторона.
    """
    from scipy.optimize import linprog

    normals = np.array([f[0] for f in facets], float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.array([f[1] for f in facets], float)
    rng = np.random.default_rng(seed)
    seeds = list(normals) + [rng.standard_normal(dim) for _ in range(directions)]
    best = 0.0
    for start in seeds:
        direction = start
        for _ in range(4):
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                break
            res = linprog(-direction / norm, A_ub=normals, b_ub=offsets,
                          bounds=[(None, None)] * dim, method="highs")
            if not res.success:
                break
            point = res.x
            value = float(np.linalg.norm(point))
            if value > best:
                best = value
            if np.linalg.norm(point - direction) < 1e-12:
                break
            direction = point
    return best


def _sublattice_points(basis: np.ndarray, bound: float):
    """Точки подрешётки в шаре радиуса ``bound`` (LLL-приведённый базис)."""
    import combigeo
    reduced = np.array(combigeo.lll_reduce(basis.tolist()))
    from chromatic_research.core.eisenstein4 import lattice_points_within
    return lattice_points_within(reduced, bound)


def evaluate_alpha(alpha: np.ndarray, ideals, *, directions: int = 220,
                   seed: int = 0) -> dict | None:
    """d = D/diam для каждого из шести идеалов при данном α.

    Ячейка Вороного целиком НЕ строится: нужны только опорные полупространства
    (``relevant_facets``) и проекция на них (``dist_to_halfspaces``).  Для
    шестимерной следовой формы ℤ[ζ₇] построение ячейки занимает минуты, а этот
    путь — доли секунды.
    """
    import combigeo
    gram = gram_from_alpha(alpha)
    if np.linalg.eigvalsh(gram)[0] <= 1e-9:
        return None
    chol = np.linalg.cholesky(gram)
    try:
        basis = np.array(combigeo.lll_reduce(chol.tolist()))
        facets = combigeo.relevant_facets(basis.tolist())
    except Exception:                                   # noqa: BLE001
        return None
    radius = covering_radius(facets, DIM, directions, seed)
    if radius <= 0:
        return None
    diam = 2.0 * radius
    # координаты идеала в LLL-базисе: строки идеала в ℤ[ζ] -> декартовы -> базис
    change = np.linalg.solve(basis.T, chol.T).T
    best = -1.0
    best_index = None
    for number, ideal in enumerate(ideals):
        sub = np.rint(ideal @ change).astype(int)
        if abs(round(np.linalg.det(sub.astype(float)))) != INDEX:
            continue
        cartesian = sub.astype(float) @ basis
        points = _sublattice_points(cartesian, 2.02 * diam)
        value = float("inf")
        for vector in sorted(points, key=lambda v: v @ v):
            if float(np.linalg.norm(vector)) - diam >= value:
                break
            value = min(value, 2.0 * combigeo.dist_to_halfspaces(
                (0.5 * vector).tolist(), facets))
        value /= diam
        if value > best:
            best, best_index = value, number
    if best_index is None:
        return None
    return {"d": float(best), "ideal": best_index, "diam": float(diam)}


def scan(grid: int, span: float, verbose: bool = True) -> dict:
    """Скан фундаментального параллелограмма действия (E⁺)² на log-плоскости."""
    ideals = prime_ideals_of_norm()
    start = time.time()
    best = {"d": -1.0}
    samples = 0
    failures = 0
    records = []
    for i in range(grid):
        for j in range(grid):
            # координаты в базисе решётки единиц, с запасом span
            u = (i + 0.5) / grid * span - (span - 1) / 2
            v = (j + 0.5) / grid * span - (span - 1) / 2
            logs = u * UNIT_LOGS[0] + v * UNIT_LOGS[1]
            logs = logs - logs.mean()                  # нормировка N(α) = 1
            alpha = np.exp(logs)
            result = evaluate_alpha(alpha, ideals)
            samples += 1
            if result is None:
                failures += 1
                continue
            records.append({"u": u, "v": v, **result})
            if result["d"] > best["d"]:
                best = {**result, "u": u, "v": v, "alpha": alpha.tolist()}
                if verbose:
                    print(f"  новый максимум d = {result['d']:.9f} "
                          f"при (u,v) = ({u:+.4f}, {v:+.4f}), идеал "
                          f"{result['ideal']} [{time.time()-start:.0f}s]",
                          flush=True)
        if verbose and (i + 1) % 5 == 0:
            print(f"  строка {i+1}/{grid}, лучшее {best['d']:.9f} "
                  f"[{time.time()-start:.0f}s]", flush=True)
    return {"best": best, "samples": samples, "failures": failures,
            "records": records, "seconds": round(time.time() - start, 1)}


def polish(best: dict, ideals, steps: int = 400) -> dict:
    """Локальная доводка лучшей точки (Нелдер–Мид по двум параметрам)."""
    from scipy.optimize import minimize

    def objective(point):
        logs = point[0] * UNIT_LOGS[0] + point[1] * UNIT_LOGS[1]
        logs = logs - logs.mean()
        result = evaluate_alpha(np.exp(logs), ideals)
        return -result["d"] if result else 1.0

    res = minimize(objective, [best["u"], best["v"]], method="Nelder-Mead",
                   options={"maxiter": steps, "xatol": 1e-10, "fatol": 1e-12})
    logs = res.x[0] * UNIT_LOGS[0] + res.x[1] * UNIT_LOGS[1]
    logs = logs - logs.mean()
    alpha = np.exp(logs)
    result = evaluate_alpha(alpha, ideals)
    return {"u": float(res.x[0]), "v": float(res.x[1]),
            "alpha": alpha.tolist(), **(result or {})}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--span", type=float, default=1.4,
                        help="во сколько раз область больше фундаментального "
                             "параллелограмма (запас на краевые эффекты)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"ℤ[ζ₇], индекс {INDEX}; действие группы квадратов единиц, "
          f"сетка {args.grid}x{args.grid}, запас x{args.span}")
    ideals = prime_ideals_of_norm()
    print(f"простых идеалов нормы {INDEX}: {len(ideals)}")

    report = scan(args.grid, args.span)
    best = report["best"]
    print(f"\nлучшее по сетке: d = {best['d']:.9f} (идеал {best['ideal']})")

    refined = polish(best, ideals)
    print(f"после доводки:   d = {refined.get('d', float('nan')):.9f}")
    verdict = ("ПОРОГ ВЗЯТ: d > 1 — кандидат на chi(R^6) <= 337"
               if refined.get("d", 0) > 1.0 else
               "порог не взят: всё циклотомическое семейство ранга 1 при "
               "k = 337 даёт d < 1")
    print(verdict)

    out = args.output or results_path("dim6_cyclotomic7_337.json")
    payload = {
        "index": INDEX, "field": "Q(zeta_7)", "module_rank": 1,
        "grid": args.grid, "span": args.span,
        "best_grid": {k: v for k, v in best.items() if k != "records"},
        "refined": refined,
        "threshold_reached": bool(refined.get("d", 0) > 1.0),
        "samples": report["samples"], "failures": report["failures"],
        "seconds": report["seconds"],
        "verdict": verdict,
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
