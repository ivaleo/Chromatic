"""Скан эйзенштейновой области форм в R^4: поиск раскрасок с k < 43.

Мотив. Рекорд chi(R^4) <= 43 (23.08.2026) достигается на решётке, у которой
Aut ~ Z/6 и элемент порядка 3 имеет характеристический многочлен (x^2+x+1)^2:
Lambda --- это Z[w]-модуль ранга 2 (эйзенштейнова решётка), а подрешётка Gamma
индекса 43 = N(7+w) --- Z[w]-подмодуль. Прежние кампании искали CMA-ES по всем
10 параметрам формы Грама и нашли рекорд лишь случайно (у k=43 было 3 рестарта).
Здесь пространство поиска сужено симметрией до ТРЁХ параметров, что позволяет
провести плотный скан плюс локальную доводку --- практически исчерпывающе.

Область: эрмитовы формы [[a, c], [conj c, b]] над Z[w], нормировка det = 1
(величина d инвариантна к масштабу), приведение по Эйзенштейну a <= b,
|c| <= a/sqrt(3), arg c in [0, pi/3]. Верх a <= sqrt(3/2) --- следствие
приведения; низ a >= A_MIN --- отсечение вырожденного (каспового) конца, где
решётка вытягивается и d падает (контролируется отдельной проверкой).

Индексы: только нормы идеалов Z[w] (индекс Z[w]-подмодуля обязан быть нормой);
в диапазоне 7..43 это 7, 9, 12, 13, 16, 19, 21, 25, 27, 28, 31, 36, 37, 39, 43.

Запуск:
    python -m chromatic_research.campaigns.dim4_eisenstein_scan [индексы] [na,nr,nt]
"""

import json
import math
import sys
import time

import numpy as np
import combigeo
from scipy.optimize import minimize

from chromatic_research.core.eisenstein4 import (evaluate, hermitian_gram,
                                                 zw_norm_indices, zw_submodules)
from chromatic_research.paths import results_path

A_MIN, A_MAX = 0.40, math.sqrt(1.5)


def make_lattice(a, rho, th):
    """Решётка области по (a, |c|, arg c); None вне области определения."""
    if not (A_MIN * 0.5 <= a <= A_MAX + 1e-9) or rho < 0 or rho > a / math.sqrt(3) + 1e-9:
        return None
    c = rho * complex(math.cos(th), math.sin(th))
    b = (1 + abs(c) ** 2) / a
    if b < a - 1e-12:
        return None
    try:
        B = np.linalg.cholesky(hermitian_gram(a, b, c))
    except np.linalg.LinAlgError:
        return None
    return B


def best_d(B, subs):
    """max по подмодулям от d = D/diam; 0.0 на вырожденных решётках.

    combigeo.voronoi_cell бросает RuntimeError, когда ячейка вырождена
    (нарушено соотношение Эйлера) --- такие точки на границе области поиска
    встречаются при доводке и должны просто отбрасываться, а не ронять прогон.
    """
    try:
        cell = combigeo.voronoi_cell(B.tolist())
    except RuntimeError:
        return 0.0
    diam = cell.diameter
    if not np.isfinite(diam) or diam <= 0:
        return 0.0
    try:
        return max(evaluate(cell, B, S, diam) for S in subs)
    except RuntimeError:
        return 0.0


def scan(subs, na, nr, nt, keep=6):
    """Грубый скан области; возвращает (лучшее, список лучших точек)."""
    top = []
    for a in np.linspace(A_MIN, A_MAX, na):
        for rho in np.linspace(0.0, a / math.sqrt(3), nr):
            for th in np.linspace(0.0, math.pi / 3, nt):
                B = make_lattice(a, rho, th)
                if B is None:
                    continue
                d = best_d(B, subs)
                top.append((d, (float(a), float(rho), float(th))))
    top.sort(key=lambda r: -r[0])
    return top[0], top[:keep]


def refine(subs, start, tol=1e-10):
    """Локальная доводка Нелдером-Мидом по (a, |c|, arg c)."""
    def neg(p):
        B = make_lattice(*p)
        return 1.0 if B is None else -best_d(B, subs)
    res = minimize(neg, np.array(start), method="Nelder-Mead",
                   options={"xatol": tol, "fatol": tol, "maxiter": 4000, "maxfev": 4000})
    return -res.fun, [float(x) for x in res.x]


def main():
    indices = ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1
               else zw_norm_indices(7, 43))
    na, nr, nt = ((int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2
                  else (40, 16, 10))
    out = {}
    for k in indices:
        subs = zw_submodules(k)
        t0 = time.time()
        (d_grid, p_grid), tops = scan(subs, na, nr, nt)
        d_best, p_best = d_grid, p_grid
        for _, p in tops:
            d, p2 = refine(subs, p)
            if d > d_best:
                d_best, p_best = d, p2
        flag = "ГОДНАЯ РАСКРАСКА" if d_best >= 1.0 else ""
        print(f"k={k:3d}  подмодулей {len(subs):4d}  сетка {d_grid:.7f}  "
              f"доводка {d_best:.9f}  {flag}  [{time.time()-t0:.0f}s]", flush=True)
        out[str(k)] = {"submodules": len(subs), "d_grid": d_grid,
                       "d_refined": d_best, "params": p_best,
                       "admissible": bool(d_best >= 1.0)}
    path = results_path("dim4_eisenstein_scan.json")
    json.dump({"domain": {"a_min": A_MIN, "a_max": A_MAX, "grid": [na, nr, nt],
                          "normalization": "det H = 1, a<=b, |c|<=a/sqrt3, arg c in [0,pi/3]"},
               "results": out}, open(path, "w"), ensure_ascii=False, indent=1)
    print("записано:", path)


if __name__ == "__main__":
    main()
