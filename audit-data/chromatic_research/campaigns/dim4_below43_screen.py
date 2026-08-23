"""Экран <<ниже 43>> в R^4: симметричные семейства, но ВСЕ подрешётки.

Кампания dim4_eisenstein_scan ограничивает и решётку (симметрией), и
подрешётку (инвариантностью). Здесь второе ограничение снято: для каждой
решётки семейства запускается штатный исчерпывающий перебор всех эрмитовых
нормальных форм заданного индекса (combigeo.find_optimal), так что
неинвариантные Gamma тоже покрыты. Это делает результат экраном для
индексов, не являющихся нормами, --- в частности для 33, 35, 38, 42, у
которых инвариантных подрешёток нет вовсе.

Область: эйзенштейново (Z[w]) и гауссово (Z[i]) семейства, три существенных
параметра каждое, приведённая область, нормировка det H = 1.

Статус результата --- отрицательный экран [Э]: он говорит, что в двух
симметричных семействах порог d = 1 при k <= 42 не достигается, и НЕ является
доказательством невозможности.

Запуск: python -m chromatic_research.campaigns.dim4_below43_screen [kmin-kmax] [na,nr,nt]
"""

import json
import math
import sys
import time

import numpy as np
import combigeo
from scipy.optimize import minimize

from chromatic_research.core.eisenstein4 import gauss_gram, hermitian_gram
from chromatic_research.paths import results_path

FAMILIES = {
    # имя: (грам, верх приведения по a, предел |c|/a, диапазон arg c)
    "eisenstein": (hermitian_gram, math.sqrt(1.5), 1 / math.sqrt(3), math.pi / 3),
    "gauss": (gauss_gram, math.sqrt(2.0), 0.5, math.pi / 2),
}
A_MIN = 0.40


def lattice(family, a, rho, th):
    gram, a_max, c_ratio, _ = FAMILIES[family]
    if not (A_MIN * 0.5 <= a <= a_max + 1e-9) or rho < 0 or rho > c_ratio * a + 1e-9:
        return None
    c = rho * complex(math.cos(th), math.sin(th))
    b = (1 + abs(c) ** 2) / a
    if b < a - 1e-12:
        return None
    try:
        return np.linalg.cholesky(gram(a, b, c))
    except np.linalg.LinAlgError:
        return None


def d_range(B, lo, hi):
    """Исчерпывающий максимум d по ВСЕМ подрешёткам каждого индекса из [lo, hi]."""
    try:
        res = combigeo.find_optimal_range(B.tolist(), lo, hi)
    except RuntimeError:
        return {k: 0.0 for k in range(lo, hi + 1)}
    return {k: res[k].normalized for k in range(lo, hi + 1)}


def d_one(B, k):
    try:
        return combigeo.find_optimal(B.tolist(), k).normalized
    except RuntimeError:
        return 0.0


def main():
    lo, hi = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "31-43").split("-"))
    na, nr, nt = ((int(x) for x in sys.argv[2].split(","))
                  if len(sys.argv) > 2 else (16, 8, 6))
    # бюджет доводки: у составных индексов (40, 42) перебор ЭНФ на порядок
    # больше, чем у простых, и щедрая доводка растягивает прогон на часы;
    # для экрана хватает сетки плюс лёгкого локального спуска
    starts, maxfev = ((int(x) for x in sys.argv[3].split(","))
                      if len(sys.argv) > 3 else (3, 200))
    path = results_path("dim4_below43_screen.json")
    out = {}
    if path.exists():                     # докатываем прерванный прогон
        prev = json.load(open(path))
        if prev.get("grid") == [na, nr, nt] and prev.get("range") == [lo, hi]:
            out = prev.get("families", {})
            print(f"продолжаем: готовых семейств {len(out)}", flush=True)

    def flush():
        json.dump({"grid": [na, nr, nt], "a_min": A_MIN, "range": [lo, hi],
                   "refine": [starts, maxfev],
                   "note": "исчерпывающий перебор подрешёток на симметричных семействах",
                   "families": out}, open(path, "w"), ensure_ascii=False, indent=1)
    for family, (_, a_max, c_ratio, th_max) in FAMILIES.items():
        if family in out:
            continue
        t0 = time.time()
        rows = []
        for a in np.linspace(A_MIN, a_max, na):
            for rho in np.linspace(0.0, c_ratio * a, nr):
                for th in np.linspace(0.0, th_max, nt):
                    B = lattice(family, a, rho, th)
                    if B is None:
                        continue
                    rows.append((d_range(B, lo, hi), (float(a), float(rho), float(th))))
        print(f"[{family}] сетка: {len(rows)} решёток за {time.time()-t0:.0f}s", flush=True)
        fam = {}
        for k in range(lo, hi + 1):
            rows.sort(key=lambda r: -r[0][k])
            d_best, p_best = rows[0][0][k], rows[0][1]
            for _, p in rows[:starts]:
                res = minimize(lambda q: 0.0 if lattice(family, *q) is None
                               else -d_one(lattice(family, *q), k),
                               np.array(p), method="Nelder-Mead",
                               options={"xatol": 1e-10, "fatol": 1e-11,
                                        "maxiter": maxfev, "maxfev": maxfev})
                if -res.fun > d_best:
                    d_best, p_best = -res.fun, [float(x) for x in res.x]
            fam[str(k)] = {"d": d_best, "params": list(p_best),
                           "admissible": bool(d_best >= 1.0)}
            print(f"[{family}] k={k:3d}  d={d_best:.7f}"
                  f"  {'ГОДНАЯ' if d_best >= 1 else ''}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
        out[family] = fam
        flush()
        print(f"[{family}] всего {time.time()-t0:.0f}s", flush=True)
    flush()
    print("записано:", path)


if __name__ == "__main__":
    main()
