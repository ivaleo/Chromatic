"""Спуск ниже 43 в R^4: поиск по ВСЕМ формам с посевом из симметричной области.

Кампания dim4_below43_screen перебирает подрешётки исчерпывающе, но решётку
держит внутри симметричного семейства. Здесь наоборот: решётка свободна (все
десять коэффициентов формы Грама, нормировка det = 1), а стартовые точки
берутся из трёх источников:

  1. оптимум эйзенштейнова семейства для этого k (новая область);
  2. оптимум гауссова семейства;
  3. чемпион кампании 2026-07 (CMA-ES по общим формам) --- прежняя область.

Смысл: проверить, не открывает ли окрестность новой (симметричной) точки
чего-то в полном пространстве --- ровно так в R^5 барьер индекса 134 был снят
переходом к другой форме, а не уточнением прежней.

Форма параметризуется нижнетреугольным L (Q = L L^T), что автоматически даёт
положительную определённость; масштаб убирается нормировкой det = 1.

Запуск: python -m chromatic_research.campaigns.dim4_below43_general [kmin-kmax] [maxfev]
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

IDX = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
       (3, 0), (3, 1), (3, 2), (3, 3)]


def to_basis(par):
    """10 параметров -> базис B решётки с det = 1 (Q = B B^T)."""
    L = np.zeros((4, 4))
    for v, (i, j) in zip(par, IDX):
        L[i, j] = v
    for i in range(4):
        L[i, i] = abs(L[i, i]) + 1e-9
    det = abs(np.linalg.det(L))
    if det < 1e-12:
        return None
    return L / det ** 0.25


def from_gram(Q):
    """Обратно: форма Грама -> параметры (фактор Холецкого, нормированный)."""
    L = np.linalg.cholesky(np.asarray(Q, float))
    L = L / abs(np.linalg.det(L)) ** 0.25
    return [L[i, j] for (i, j) in IDX]


def d_of(par, k):
    B = to_basis(par)
    if B is None:
        return 0.0
    try:
        return combigeo.find_optimal(B.tolist(), k).normalized
    except RuntimeError:
        return 0.0


def seeds(k, screen, cma):
    """Стартовые точки для индекса k."""
    out = []
    for fam, gram, ratio in (("eisenstein", hermitian_gram, 1 / math.sqrt(3)),
                             ("gauss", gauss_gram, 0.5)):
        row = screen["families"].get(fam, {}).get(str(k))
        if not row:
            continue
        a, rho, th = row["params"]
        c = rho * complex(math.cos(th), math.sin(th))
        b = (1 + abs(c) ** 2) / a
        try:
            out.append((fam, from_gram(gram(a, b, c))))
        except np.linalg.LinAlgError:
            pass
    row = cma.get(f"k{k}")
    if row and "Q" in row:
        try:
            out.append(("cma2026-07", from_gram(row["Q"])))
        except np.linalg.LinAlgError:
            pass
    return out


def main():
    lo, hi = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "31-42").split("-"))
    maxfev = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    screen = json.load(open(results_path("dim4_below43_screen.json")))
    cma = json.load(open(results_path("n8_cma44_ladder.json")))

    out = {}
    for k in range(lo, hi + 1):
        t0, best, tag = time.time(), 0.0, None
        for name, par in seeds(k, screen, cma):
            d0 = d_of(par, k)
            res = minimize(lambda p: -d_of(p, k), np.array(par), method="Nelder-Mead",
                           options={"xatol": 1e-9, "fatol": 1e-10,
                                    "maxiter": maxfev, "maxfev": maxfev})
            d = max(d0, -res.fun)
            if d > best:
                best, tag = d, name
            print(f"  k={k:3d} посев {name:12s}: {d0:.6f} -> {d:.7f}", flush=True)
        out[str(k)] = {"d": best, "seed": tag, "admissible": bool(best >= 1.0)}
        print(f"k={k:3d}  ЛУЧШЕЕ {best:.7f} (из {tag})"
              f"  {'ГОДНАЯ' if best >= 1 else ''}  [{time.time()-t0:.0f}s]", flush=True)
    path = results_path("dim4_below43_general.json")
    json.dump({"range": [lo, hi], "maxfev": maxfev, "results": out,
               "note": "поиск по всем формам Грама с посевом из симметричных семейств"},
              open(path, "w"), ensure_ascii=False, indent=1)
    print("записано:", path)


if __name__ == "__main__":
    main()
