r"""Прицельный плотный скан одного класса симметрии по одному индексу.

Атлас (dim4_symmetry_atlas) проходит по двенадцати классам Z[S]-модулей ранга 4
случайной выборкой --- этого хватает на экран, но мало для рекорда. Здесь один
класс сканируется плотно: направления инвариантного конуса берутся сеткой на
сфере вокруг канонической формы Q0 = sum_j S^j (S^j)^T, для каждой формы
запускается ШТАТНЫЙ исчерпывающий перебор всех подрешёток индекса k, а лучшие
точки доводятся Нелдером--Мидом.

Повод: класс Z[C_3] (+) Z --- регулярное представление C_3 плюс тривиальное
слагаемое, то есть склеенный модуль, не разлагающийся в прямую сумму
подрешёток, --- дал при k = 42 ширину 0,9583 против 0,9489 у прежнего
чемпиона по всем формам. Это лучший известный результат для индекса, который
принципиально недоступен инвариантным подрешёткам (42 --- не норма ни в Z[w],
ни в Z[i]).

Запуск:
    python -m chromatic_research.campaigns.dim4_glued_focus [класс] [k1,k2] [N]
"""

import json
import sys
import time

import numpy as np
import combigeo
from scipy.optimize import minimize

from chromatic_research.campaigns.dim4_symmetry_atlas import (CANDIDATES, COND_MAX,
                                                              canonical,
                                                              invariant_basis,
                                                              order_of)
from chromatic_research.paths import runs_path

SEED = 20260823


def basis_of(name):
    S = CANDIDATES[name]
    return S, invariant_basis(S)


def width(t, basis, k):
    """d при индексе k для формы sum t_i B_i; 0.0 на вырожденных."""
    Q = sum(c * B for c, B in zip(t, basis))
    Q = 0.5 * (Q + Q.T)
    ev = np.linalg.eigvalsh(Q)
    if ev.min() <= 1e-9:
        return 0.0
    Q = Q / np.linalg.det(Q) ** 0.25
    try:
        B = np.array(combigeo.lll_reduce(np.linalg.cholesky(Q).tolist()))
    except (RuntimeError, np.linalg.LinAlgError):
        return 0.0
    G = B @ B.T
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    if ev.min() <= 1e-9 or ev.max() / ev.min() > COND_MAX:
        return 0.0
    try:
        cell = combigeo.voronoi_cell(B.tolist())
        if len(cell.facets) < 8 or not np.isfinite(cell.diameter):
            return 0.0
        return combigeo.find_optimal(B.tolist(), k).normalized
    except RuntimeError:
        return 0.0


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Z[C3] (+) Z (склейка)"
    ks = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
          else [40, 41, 42])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    S, basis = basis_of(name)
    r = len(basis)
    base = canonical(S, basis)
    scale = max(1e-9, float(np.abs(base).max()))
    print(f"класс {name}: порядок {order_of(S)}, параметров {r}", flush=True)

    rng = np.random.default_rng(SEED)
    # сетка направлений: равномерная выборка на сфере в конусе + разные радиусы
    dirs = rng.normal(size=(N, r))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    radii = np.linspace(0.05, 1.6, 12)

    out = {}
    for k in ks:
        t0, rows = time.time(), []
        for i, u in enumerate(dirs):
            t = base + radii[i % len(radii)] * scale * u
            d = width(t, basis, k)
            if d > 0:
                rows.append((d, t))
        rows.sort(key=lambda z: -z[0])
        d_best, t_best = (rows[0] if rows else (0.0, base))
        print(f"  k={k}: сетка {d_best:.7f} ({len(rows)} годных форм, "
              f"{time.time()-t0:.0f}s)", flush=True)
        for _, t in rows[:6]:
            res = minimize(lambda x: -width(x, basis, k), np.array(t),
                           method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-11,
                                    "maxiter": 500, "maxfev": 500})
            if -res.fun > d_best:
                d_best, t_best = -res.fun, [float(x) for x in res.x]
        print(f"  k={k}: доводка {d_best:.7f}  "
              f"{'ГОДНАЯ' if d_best >= 1 else ''}  [{time.time()-t0:.0f}s]",
              flush=True)
        out[str(k)] = {"d": d_best, "t": list(map(float, t_best)),
                       "admissible": bool(d_best >= 1.0), "samples": len(rows)}
        json.dump({"class": name, "samples": N, "seed": SEED, "results": out,
                   "note": "прицельный плотный скан одного класса симметрии"},
                  open(runs_path("dim4_glued_focus.json"), "w"),
                  ensure_ascii=False, indent=1)
    print("записано:", runs_path("dim4_glued_focus.json"))


if __name__ == "__main__":
    main()
