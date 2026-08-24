"""ℝ⁶, индекс 337: ИСЧЕРПЫВАЮЩИЙ экран эйзенштейновых подмодулей.

Почему именно эта ветвь.  Действующий чемпион размерности 6 — конструкция
``(3+ω)E₆*`` с 343 цветами — сам лежит в этом семействе: ``E₆*`` есть
ℤ[ω]-модуль ранга 3.  То есть речь не о новой экзотике, а о том же семействе на
индексе, который на шесть меньше.  Арифметическая зацепка:
``337 = N(21 + 8ω) = 441 − 168 + 64`` — норма эйзенштейнова простого, а
``337 ≡ 1 (mod 3)``, поэтому 337 расщепляется в ℤ[ω].

Модель.  Берётся вещественная решётка ``Λ ⊂ ℝ⁶`` вместе с автоморфизмом ``S``
порядка 3 с характеристическим многочленом ``(x²+x+1)³`` — это и есть структура
ℤ[ω]-модуля ранга 3 (``ω`` действует как ``S``).  Пространство ``S``-инвариантных
форм девятимерно (против 21 у общей шестимерной формы), и, что важно, определено
над ℚ: рационализация не выводит из семейства.

Полнота перебора подмодулей.  Индекс 337 прост, поэтому у ЛЮБОГО
``S``-инвариантного подмодуля ``M`` индекса 337 фактор ``Λ/M`` имеет порядок 337,
и ``ω`` действует на нём умножением на корень ``r`` уравнения ``r²+r+1 ≡ 0``.
Значит ``M = ker(x ↦ x·c mod 337)``, где ``c`` — собственный вектор ``S`` с
собственным значением ``r`` (``Sc = rc``).  Собственное подпространство трёхмерно, поэтому
подмодулей ровно ``|ℙ²(𝔽₃₃₇)| = 1 + 337 + 337² = 113 907`` на каждый из двух
корней, итого **227 814** — и это ВЕСЬ список, а не выборка.

Как перебор делается быстро.  Проверять 227 814 подмодулей по одному
неподъёмно.  Работает двойственность: ``x ∈ M_c`` равносильно ``c·x ≡ 0``, то
есть ЛИНЕЙНОМУ условию на ``c``.  Поэтому каждый ЗАПРЕЩЁННЫЙ вектор (тот, у
которого ``D(x) < ℓ·diam``) вычёркивает ровно одну прямую в ``ℙ²`` — 338 точек.
Векторы обрабатываются по возрастанию нормы; экран заканчивается либо непустым
множеством выживших (кандидат на ``χ(ℝ⁶) ≤ 337``), либо полным вычёркиванием —
и это доказательство, что при данной форме индекс 337 недостижим.

Достаточно смотреть только на ``|x| < diam + ℓ·diam``: из ``D(x) ≥ |x| − diam``
длинные векторы разрешены автоматически.

Usage::

    python -m chromatic_research.campaigns.dim6_eisenstein_337 --trials 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chromatic_research.paths import results_path

INDEX = 337
RANK = 3
DIM = 2 * RANK


# ------------------------------------------------------- структура ℤ[ω]-модуля


def cube_roots_of_unity(modulus: int = INDEX) -> list[int]:
    """Корни ``x² + x + 1 ≡ 0 (mod p)``: два образа ``ω``."""
    return [r for r in range(1, modulus) if (r * r + r + 1) % modulus == 0]


def minimal_vectors(basis: np.ndarray) -> np.ndarray:
    """Целочисленные координаты всех минимальных векторов (обе полусферы)."""
    from chromatic_research.core.eisenstein4 import lattice_points_within

    gram = basis @ basis.T
    shortest = float(np.min(np.diag(gram)))
    points = lattice_points_within(basis, np.sqrt(shortest) + 1e-7)
    inverse = np.linalg.inv(basis)
    coords = np.rint(np.array(points) @ inverse).astype(int)
    both = np.vstack([coords, -coords])
    return np.unique(both, axis=0)


def find_order3_automorphism(basis: np.ndarray) -> np.ndarray | None:
    """Автоморфизм порядка 3 с характеристическим многочленом ``(x²+x+1)³``.

    Строки ``S`` — координаты образов базисных векторов; они обязаны быть
    векторами той же нормы, поэтому кандидаты берутся среди минимальных
    векторов (базис после LLL из них и состоит).  Отсечение по скалярным
    произведениям делает перебор мгновенным.
    """
    gram = basis @ basis.T
    candidates = minimal_vectors(basis)
    values = candidates @ gram @ candidates.T
    norms = np.diag(values)
    target = np.diag(gram)
    rows: list[int] = []

    def descend(level: int) -> np.ndarray | None:
        if level == DIM:
            matrix = candidates[rows]
            if not np.array_equal(matrix @ matrix @ matrix, np.eye(DIM, dtype=int)):
                return None
            if np.array_equal(matrix, np.eye(DIM, dtype=int)):
                return None
            # характеристический многочлен обязан быть (x²+x+1)³:
            # это равносильно S² + S + I = 0
            if np.any(matrix @ matrix + matrix + np.eye(DIM, dtype=int)):
                return None
            return matrix
        for index in range(len(candidates)):
            if abs(norms[index] - target[level]) > 1e-7:
                continue
            if any(abs(values[index, rows[k]] - gram[level, k]) > 1e-7
                   for k in range(level)):
                continue
            rows.append(index)
            found = descend(level + 1)
            if found is not None:
                return found
            rows.pop()
        return None

    return descend(0)


def invariant_form_basis(automorphism: np.ndarray) -> list[np.ndarray]:
    """Базис пространства симметричных ``G`` с ``S G Sᵀ = G`` (обязано быть 9-мерным).

    Соглашение: координаты — СТРОКИ, образ вектора ``x`` есть ``x S``, поэтому
    условие изометрии имеет вид ``S G Sᵀ = G``.
    """
    pairs = [(i, j) for i in range(DIM) for j in range(i, DIM)]
    columns = []
    for i, j in pairs:
        template = np.zeros((DIM, DIM))
        template[i, j] = template[j, i] = 1.0
        image = automorphism @ template @ automorphism.T - template
        columns.append([image[a, b] for a, b in pairs])
    matrix = np.array(columns).T
    _, singular, vectors = np.linalg.svd(matrix)
    kernel = vectors[np.sum(singular > 1e-9):]
    out = []
    for row in kernel:
        form = np.zeros((DIM, DIM))
        for value, (i, j) in zip(row, pairs):
            form[i, j] = form[j, i] = value
        out.append(form)
    return out


# ---------------------------------------------------------------- проективное


def canonical(point, modulus: int = INDEX):
    """Каноническое представление точки ``ℙ²`` (первая ненулевая = 1)."""
    for k in range(RANK):
        if point[k] % modulus:
            inverse = pow(int(point[k]) % modulus, modulus - 2, modulus)
            return tuple(int(v * inverse % modulus) for v in point)
    return None


def hyperplane_points(functional, modulus: int = INDEX):
    """Все точки ``ℙ²`` с ``a · functional ≡ 0`` — их ровно ``p + 1``."""
    y = [int(v) % modulus for v in functional]
    pivot = next(k for k in range(RANK) if y[k])
    basis = []
    for k in range(RANK):
        if k == pivot:
            continue
        vector = [0] * RANK
        vector[k] = 1
        vector[pivot] = (-y[k] * pow(y[pivot], modulus - 2, modulus)) % modulus
        basis.append(vector)
    first, second = basis
    points = [canonical(first, modulus)]
    for t in range(modulus):
        points.append(canonical([(second[k] + t * first[k]) % modulus
                                 for k in range(RANK)], modulus))
    return points


def eigenspace(automorphism: np.ndarray, root: int,
               modulus: int = INDEX) -> list[list[int]]:
    """Базис пространства ``{c : Sc ≡ rc}`` над ``𝔽_p`` (обязано быть 3-мерным).

    Именно эти ``c`` задают подмодули: при ``Sc = rc`` условие ``x·c ≡ 0``
    сохраняется действием ``ω`` (``(xS)·c = x(Sc) = r(x·c)``), то есть
    ``M_c = {x : x·c ≡ 0}`` — ℤ[ω]-подмодуль индекса 337.
    """
    matrix = [[int(automorphism[i][j]) % modulus for j in range(DIM)]
              for i in range(DIM)]
    for i in range(DIM):
        matrix[i][i] = (matrix[i][i] - root) % modulus
    rows = [row[:] for row in matrix]
    pivots = []
    row = 0
    for col in range(DIM):
        target = next((r for r in range(row, DIM) if rows[r][col] % modulus), None)
        if target is None:
            continue
        rows[row], rows[target] = rows[target], rows[row]
        inverse = pow(rows[row][col], modulus - 2, modulus)
        rows[row] = [v * inverse % modulus for v in rows[row]]
        for r in range(DIM):
            if r != row and rows[r][col] % modulus:
                factor = rows[r][col]
                rows[r] = [(rows[r][k] - factor * rows[row][k]) % modulus
                           for k in range(DIM)]
        pivots.append(col)
        row += 1
    free = [c for c in range(DIM) if c not in pivots]
    kernel = []
    for f in free:
        vector = [0] * DIM
        vector[f] = 1
        for r, col in enumerate(pivots):
            vector[col] = (-rows[r][f]) % modulus
        kernel.append(vector)
    return kernel


# ------------------------------------------------------------------- геометрия


def covering_radius(facets, directions: int, seed: int) -> float:
    """``R_cov`` по опорным полупространствам; оценка снизу — безопасная сторона.

    Проверено на каталоге: E₆ → 1.632993, E₆* → 1.414214, D₆ → 1.732051,
    A₆* → 1.632993, то есть точные значения ``2R/λ₁``.
    """
    from scipy.optimize import linprog

    normals = np.array([f[0] for f in facets], float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.array([f[1] for f in facets], float)
    rng = np.random.default_rng(seed)
    seeds = list(normals) + [rng.standard_normal(DIM) for _ in range(directions)]
    best = 0.0
    for start in seeds:
        direction = start
        for _ in range(4):
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                break
            res = linprog(-direction / norm, A_ub=normals, b_ub=offsets,
                          bounds=[(None, None)] * DIM, method="highs")
            if not res.success:
                break
            point = res.x
            best = max(best, float(np.linalg.norm(point)))
            if np.linalg.norm(point - direction) < 1e-12:
                break
            direction = point
    return best


def screen(basis: np.ndarray, automorphism: np.ndarray, *, ell: float = 1.0,
           directions: int = 200, seed: int = 0, verbose: bool = False) -> dict:
    """Исчерпывающий экран всех 227 814 подмодулей индекса 337 при данной форме."""
    import combigeo
    from chromatic_research.core.eisenstein4 import lattice_points_within

    reduced = np.array(combigeo.lll_reduce(basis.tolist()))
    facets = combigeo.relevant_facets(reduced.tolist())
    radius = covering_radius(facets, directions, seed)
    diameter = 2.0 * radius
    threshold = ell * diameter
    window = diameter + threshold + 1e-9

    # координаты берём в ИСХОДНОМ базисе: в нём задан автоморфизм S
    inverse = np.linalg.inv(basis)
    points = lattice_points_within(reduced, window)
    records = []
    for vector in points:
        coords = np.rint(vector @ inverse).astype(int)
        records.append((float(vector @ vector), coords, vector))
    records.sort(key=lambda item: item[0])

    roots = cube_roots_of_unity(INDEX)
    eigen = [eigenspace(automorphism, root, INDEX) for root in roots]
    for space in eigen:
        if len(space) != RANK:
            return {"ok": False, "reason": f"собственное подпространство "
                                           f"размерности {len(space)}"}

    total = INDEX * INDEX + INDEX + 1
    killed: list[set | None] = [set(), set()]
    forbidden = 0
    checked = 0
    start = time.time()
    for _, coords, vector in records:
        checked += 1
        distance = 2.0 * combigeo.dist_to_halfspaces((0.5 * vector).tolist(), facets)
        if distance >= threshold:
            continue
        forbidden += 1
        for which in (0, 1):
            if killed[which] is None:
                continue
            image = [int(sum(eigen[which][k][i] * int(coords[i])
                             for i in range(DIM)) % INDEX) for k in range(RANK)]
            if all(v == 0 for v in image):
                killed[which] = None      # вектор лежит во ВСЕХ подмодулях
                continue
            killed[which].update(hyperplane_points(image, INDEX))
        if all(k is None or len(k) >= total for k in killed):
            break
        if verbose and forbidden % 500 == 0:
            sizes = ["все" if k is None else len(k) for k in killed]
            print(f"    запрещённых {forbidden}/{checked}, вычеркнуто {sizes} "
                  f"из {total} [{time.time() - start:.0f}s]", flush=True)

    survivors = [0 if k is None else total - len(k) for k in killed]
    return {
        "ok": True, "radius": radius, "diameter": diameter,
        "checked": checked, "total_window": len(records), "forbidden": forbidden,
        "survivors": survivors,
        "any_survivor": bool(survivors[0] > 0 or survivors[1] > 0),
        "seconds": round(time.time() - start, 1),
    }


def best_ratio(basis: np.ndarray, automorphism: np.ndarray, *,
               low: float = 0.5, high: float = 1.25, steps: int = 9,
               directions: int = 150, seed: int = 0) -> float:
    """Максимум ``d = D/diam`` ПО ВСЕМ 227 814 подмодулям при данной форме.

    Множество выживших монотонно убывает по порогу, поэтому точная верхняя
    грань порогов, при которых выжившие есть, и есть искомый максимум.  Двоичный
    поиск: каждый шаг — один исчерпывающий экран.
    """
    if not screen(basis, automorphism, ell=low, directions=directions,
                  seed=seed).get("any_survivor"):
        return low
    if screen(basis, automorphism, ell=high, directions=directions,
              seed=seed).get("any_survivor"):
        return high
    for _ in range(steps):
        middle = 0.5 * (low + high)
        report = screen(basis, automorphism, ell=middle,
                        directions=directions, seed=seed)
        if report.get("any_survivor"):
            low = middle
        else:
            high = middle
    return low


# ----------------------------------------------------------------- кампания


def seed_lattice() -> tuple[np.ndarray, np.ndarray]:
    """E₆* из каталога вместе с найденным ω-автоморфизмом."""
    import combigeo
    from chromatic_research.core.lattices import CATALOG

    basis = np.array(combigeo.lll_reduce(np.array(CATALOG["E6*"]()).tolist()))
    automorphism = find_order3_automorphism(basis)
    if automorphism is None:
        raise RuntimeError("у E6* не найден автоморфизм порядка 3")
    return basis, automorphism


def optimize(basis: np.ndarray, automorphism: np.ndarray, forms, *,
             restarts: int, sigma: float, seed: int, steps: int,
             directions: int, output: Path) -> dict:
    """Максимизирует ``max_submodules d`` по девятимерному инвариантному конусу.

    Каждое вычисление целевой функции — двоичный поиск из ``steps`` ИСЧЕРПЫВАЮЩИХ
    экранов, то есть учитывает все 227 814 подмодулей.  Поэтому найденный
    максимум — это максимум по ПАРАМ (форма, подмодуль), а не по одной паре.
    """
    gram0 = basis @ basis.T
    scale = np.trace(gram0) / DIM
    rng = np.random.default_rng(seed)
    history = []
    best = {"ratio": -1.0}
    start = time.time()

    def evaluate(coefficients) -> float:
        gram = gram0 + scale * sum(c * f for c, f in zip(coefficients, forms))
        gram = 0.5 * (gram + gram.T)
        try:
            if np.linalg.eigvalsh(gram)[0] <= 1e-9:
                return -1.0
            current = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            return -1.0
        return best_ratio(current, automorphism, steps=steps,
                          directions=directions, seed=seed)

    from scipy.optimize import minimize

    for restart in range(restarts):
        origin = (np.zeros(len(forms)) if restart == 0
                  else rng.normal(0, sigma, len(forms)))
        result = minimize(lambda c: -evaluate(c), origin, method="Nelder-Mead",
                          options={"maxiter": 220, "fatol": 1e-6,
                                   "xatol": 1e-5})
        value = -float(result.fun)
        history.append({"restart": restart, "ratio": value,
                        "coefficients": [float(v) for v in result.x]})
        marker = ""
        if value > best["ratio"]:
            best = {"ratio": value, "coefficients": [float(v) for v in result.x],
                    "restart": restart}
            marker = "  <-- лучшее"
        print(f"  рестарт {restart}: max d = {value:.6f}{marker} "
              f"[{time.time() - start:.0f}s]", flush=True)
        output.write_text(json.dumps(
            {"index": INDEX, "best": best, "history": history,
             "restarts_done": restart + 1,
             "submodules_per_form": 2 * (INDEX ** 2 + INDEX + 1),
             "seconds": round(time.time() - start, 1)},
            indent=1, ensure_ascii=False) + "\n")
    return {"best": best, "history": history,
            "seconds": round(time.time() - start, 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--ell", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--optimize", action="store_true",
                        help="искать максимум d по инвариантному конусу "
                             "(двоичный поиск порога на каждой форме)")
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--directions", type=int, default=120)
    args = parser.parse_args(argv)

    if args.optimize:
        basis, automorphism = seed_lattice()
        forms = invariant_form_basis(automorphism)
        out = args.output or results_path("dim6_eisenstein_337_optimum.json")
        print(f"максимизация max_подмодули d по {len(forms)}-мерному "
              f"инвариантному конусу; на каждой форме — {2*(INDEX**2+INDEX+1)} "
              f"подмодулей")
        report = optimize(basis, automorphism, forms, restarts=args.restarts,
                          sigma=args.sigma, seed=args.seed, steps=args.steps,
                          directions=args.directions, output=out)
        ratio = report["best"]["ratio"]
        print(f"\nмаксимум по семейству: d = {ratio:.6f} "
              f"({'ПОРОГ ВЗЯТ' if ratio >= 1.0 else 'порог не взят'})")
        print(f"сохранено: {out}")
        return 0

    basis, automorphism = seed_lattice()
    forms = invariant_form_basis(automorphism)
    print(f"эйзенштейнов экран, индекс {INDEX}, порог d >= {args.ell}")
    print(f"корни x²+x+1 mod {INDEX}: {cube_roots_of_unity()}")
    print(f"подмодулей: 2 × {INDEX**2 + INDEX + 1} = {2*(INDEX**2+INDEX+1)}")
    print(f"инвариантных форм: {len(forms)}-мерное пространство")

    gram0 = basis @ basis.T
    rng = np.random.default_rng(args.seed)
    records = []
    found = None
    start = time.time()
    for trial in range(args.trials):
        if trial == 0:
            gram = gram0
        else:
            gram = gram0.copy()
            for form in forms:
                gram = gram + rng.normal(0, args.sigma) * form * np.trace(gram) / DIM
            gram = 0.5 * (gram + gram.T)
            if np.linalg.eigvalsh(gram)[0] <= 1e-9:
                continue
        try:
            current = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError:
            continue
        # автоморфизм задан в координатах базиса, поэтому переносится как есть
        report = screen(current, automorphism, ell=args.ell,
                        seed=args.seed + trial)
        if not report.get("ok"):
            continue
        records.append({"trial": trial, **{k: v for k, v in report.items()
                                           if k != "ok"}})
        tag = "E6*" if trial == 0 else f"проба {trial}"
        print(f"  {tag}: diam={report['diameter']:.6f}, запрещённых "
              f"{report['forbidden']}/{report['checked']} из "
              f"{report['total_window']}, выжило {report['survivors']} "
              f"[{report['seconds']}s]", flush=True)
        if report["any_survivor"]:
            found = records[-1]
            print("  !!! ЕСТЬ ВЫЖИВШИЕ — кандидат на chi(R^6) <= 337")
            break

    verdict = ("НАЙДЕН кандидат индекса 337" if found else
               f"на всех {len(records)} инвариантных формах ни один из "
               f"{2*(INDEX**2+INDEX+1)} подмодулей не проходит порог d >= "
               f"{args.ell}")
    print(f"\n{verdict}  [{time.time()-start:.0f}s]")

    out = args.output or results_path("dim6_eisenstein_337_screen.json")
    out.write_text(json.dumps({
        "index": INDEX, "ell": args.ell, "forms_tested": len(records),
        "submodules_per_form": 2 * (INDEX ** 2 + INDEX + 1),
        "invariant_form_dimension": len(forms),
        "found": found, "verdict": verdict, "records": records,
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"сохранено: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
