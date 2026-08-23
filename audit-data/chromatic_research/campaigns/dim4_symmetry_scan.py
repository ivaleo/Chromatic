"""Скан симметрийно-ограниченных семейств решёток в R^4 (общий движок).

Принцип кампании 23.08.2026: рекорд chi(R^4) <= 43 нашёлся на решётке с
Aut ~ Z/6, то есть на Z[w]-модуле ранга 2. Конечный порядок элемента
S in GL_4(Z) фиксирует структуру O-модуля, а пространство S-инвариантных форм
резко сужается:

    порядок 3 или 6, char = Phi_3^2 / Phi_6^2  -> Z[w]-модуль ранга 2, 4 параметра
    порядок 4,        char = Phi_4^2           -> Z[i]-модуль ранга 2, 4 параметра
    порядок 5/10/8/12, char = Phi_m (deg 4)    -> Z[zeta_m]-модуль ранга 1, 2 параметра

С точностью до масштаба это 3, 3 и 1 существенный параметр против 9 у общей
формы. Поэтому скан здесь плотный, а не выборочный, как CMA-ES.

Индекс O-подмодуля обязан быть нормой идеала O, поэтому перебираются только
такие k. Циклотомические семейства (ранг 1) дополнительно прогоняются штатным
исчерпывающим перебором ВСЕХ подрешёток (combigeo.find_optimal), который
покрывает и неинвариантные Gamma.

Запуск:
    python -m chromatic_research.campaigns.dim4_symmetry_scan gauss  [k1,k2,...]
    python -m chromatic_research.campaigns.dim4_symmetry_scan cyclo  [kmin-kmax]
"""

import json
import math
import sys
import time

import numpy as np
import combigeo
from scipy.optimize import minimize

from chromatic_research.core.eisenstein4 import (evaluate, gauss_gram,
                                                 zi_elements_of_norm, zi_submodules)
from chromatic_research.paths import results_path

A_MIN, A_MAX = 0.40, math.sqrt(2.0)   # приведение по Гауссу: |c| <= a/2, a <= b


def gauss_lattice(a, rho, th):
    """Базис гауссовой решётки по (a, |c|, arg c) при нормировке det H = 1."""
    if rho < 0 or rho > a / 2 + 1e-9:
        return None
    c = rho * complex(math.cos(th), math.sin(th))
    b = (1 + abs(c) ** 2) / a
    if b < a - 1e-12:
        return None
    try:
        return np.linalg.cholesky(gauss_gram(a, b, c))
    except np.linalg.LinAlgError:
        return None


def best_d(B, subs):
    """max по подмодулям от d; 0.0 на вырожденных решётках."""
    try:
        cell = combigeo.voronoi_cell(B.tolist())
        diam = cell.diameter
        if not np.isfinite(diam) or diam <= 0:
            return 0.0
        return max(evaluate(cell, B, S, diam) for S in subs)
    except RuntimeError:
        return 0.0


def run_gauss(indices, grid=(30, 12, 9)):
    na, nr, nt = grid
    out = {}
    for k in indices:
        subs = zi_submodules(k)
        if not subs:
            continue
        t0, top = time.time(), []
        for a in np.linspace(A_MIN, A_MAX, na):
            for rho in np.linspace(0.0, a / 2, nr):
                for th in np.linspace(0.0, math.pi / 2, nt):
                    B = gauss_lattice(a, rho, th)
                    if B is not None:
                        top.append((best_d(B, subs), (float(a), float(rho), float(th))))
        top.sort(key=lambda r: -r[0])
        d_best, p_best = top[0]
        for _, p in top[:6]:
            res = minimize(lambda q: 1.0 if gauss_lattice(*q) is None
                           else -best_d(gauss_lattice(*q), subs),
                           np.array(p), method="Nelder-Mead",
                           options={"xatol": 1e-11, "fatol": 1e-12,
                                    "maxiter": 4000, "maxfev": 4000})
            if -res.fun > d_best:
                d_best, p_best = -res.fun, [float(x) for x in res.x]
        print(f"[Z[i]] k={k:3d}  подмодулей {len(subs):4d}  сетка {top[0][0]:.7f}  "
              f"доводка {d_best:.9f}  {'ГОДНАЯ' if d_best >= 1 else ''}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        out[str(k)] = {"submodules": len(subs), "d_grid": top[0][0],
                       "d_refined": d_best, "params": p_best,
                       "admissible": bool(d_best >= 1.0)}
    return out


# --------------------------------------------------------------------------------
# циклотомические семейства ранга 1: Lambda = Z[zeta_m], форма Tr(alpha x conj(y))
# --------------------------------------------------------------------------------

def cyclotomic_gram(m, t):
    """Грам Z-базиса (1, z, z^2, z^3) кольца Z[zeta_m] в вложении Минковского.

    Поле Q(zeta_m) степени 4 имеет ДВЕ пары комплексных вложений; решётка
    живёт в C^2 = R^4, а инвариантная форма --- это
        h(x, y) = a1 * s1(x) conj(s1(y)) + a2 * s2(x) conj(s2(y)),
    где (a1, a2) --- вполне положительный элемент вещественного подполя.
    Масштаб несуществен, поэтому a1 = 1, a2 = t --- единственный параметр
    (симметрия Галуа даёт t ~ 1/t, достаточно t in (0, 1]).
    """
    exps = {5: (1, 2), 8: (1, 3), 12: (1, 5)}[m]
    sig = [[complex(math.cos(2 * math.pi * e * j / m),
                    math.sin(2 * math.pi * e * j / m)) for j in range(4)]
           for e in exps]
    G = np.array([[ (sig[0][i] * sig[0][j].conjugate()).real
                    + t * (sig[1][i] * sig[1][j].conjugate()).real
                   for j in range(4)] for i in range(4)])
    return 0.5 * (G + G.T)


def run_cyclotomic(kmin, kmax, samples=26):
    out = {}
    for m in (5, 8, 12):
        rows = []
        for t in np.linspace(0.06, 1.0, samples):
            G = cyclotomic_gram(m, float(t))
            if np.linalg.eigvalsh(G).min() <= 1e-9:
                continue
            B = np.linalg.cholesky(G)
            try:
                res = combigeo.find_optimal_range(B.tolist(), kmin, kmax)
            except RuntimeError:
                continue
            rows.append((float(t), {k: res[k].normalized for k in sorted(res)}))
        best = {k: max((r[1][k], r[0]) for r in rows) for k in range(kmin, kmax + 1)}
        print(f"[Z[zeta_{m}]] лучшие d по k: " +
              " ".join(f"{k}:{best[k][0]:.4f}" for k in sorted(best)), flush=True)
        out[f"zeta{m}"] = {str(k): {"d": best[k][0], "t": best[k][1]} for k in best}
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gauss"
    if mode == "gauss":
        indices = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
                   else [k for k in range(8, 43) if zi_elements_of_norm(k)])
        res = run_gauss(indices)
        path = results_path("dim4_gauss_scan.json")
    else:
        lo, hi = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "31-44").split("-"))
        res = run_cyclotomic(lo, hi)
        path = results_path("dim4_cyclotomic_scan.json")
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
    print("записано:", path)


if __name__ == "__main__":
    main()
