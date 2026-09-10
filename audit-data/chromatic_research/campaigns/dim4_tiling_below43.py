r"""Мозаичная атака на индексы 40--42 в R^4.

Мотив (\S«границы метода» статьи). Индексы 40, 41, 42 выпали из
симметрийного перебора не по геометрической, а по АРИФМЕТИЧЕСКОЙ причине:
42 не является нормой ни в Z[w], ни в Z[i], поэтому подрешётки нужной формы
там просто не существует. Мозаичная схема (степенная/лагеррова диаграмма
Gamma-периодического набора узлов с весами) этого ограничения не знает --- в
плоскости именно на нелёшиановых индексах она обгоняет решёточную. Пол
мозаичного принципа даёт в R^4 лишь 2^4 = 16, так что k = 42 им не запрещён.

Два типа посева, оба принципиально разные:

  (а) лучшая известная РЕШЁТОЧНАЯ раскраска этого индекса (мозаика с w = 0 и
      узлами-трансверсалью Lambda/Gamma) --- CMA стартует ровно из неё и
      может только улучшить;
  (б) чемпион индекса 43 МИНУС ОДИН УЗЕЛ (drop): раскраска индекса 43 годна с
      запасом 0,4 %, и если выбросить один узел, оставшиеся 42 ячейки
      разрастаются --- вопрос лишь в том, отыграет ли мозаичная свобода
      потерю. Этот посев стартует из заведомо хорошей области.

Запуск:
    python -m chromatic_research.campaigns.dim4_tiling_below43 [k1,k2] [бюджет]
"""

import json
import sys
import time

import numpy as np
import combigeo

from chromatic_research.campaigns import power_search as ps
from chromatic_research.core import power_coloring as pc
from chromatic_research.paths import load_json, results_path

N = 4
DROPS = 0           # сколько узлов выбрасывать из 43-раскраски (посев «б»);
                    # по умолчанию 0: замер показал, что такой посев стартует
                    # с d = 0,6797 и заведомо не конкурент решёточному чемпиону
                    # индекса 42 (0,9583), а стоит столько же

# Шаг CMA. Размерность задачи ~ n*(k-1) + (k-1) + 10 = 215 при k = 42, поэтому
# «обычная» sigma = 0.1 даёт смещение с нормой 0.1*sqrt(215) ~ 1.5 и попросту
# уничтожает посев: первое же поколение улетает в d ~ 0.45. Берём шаг обратно
# пропорционально корню из размерности — это локальная полировка посева, а
# не глобальный поиск (глобальный по 215 параметрам безнадёжен).
SIGMA_SCALE = 0.16


def headline_lattice():
    """Базис заголовочной решётки R^4/43 (эйзенштейнов оптимум)."""
    from fractions import Fraction as F
    src = load_json("r4_k43_eisenstein_rational.json")
    Q = np.array([[float(F(x)) for x in row] for row in src["Q_fractions"]])
    return np.linalg.cholesky(Q), np.array(src["transition"], float)


def best_sublattice(B, k):
    """Лучшая подрешётка индекса k: (целочисленная матрица H, ширина)."""
    res = combigeo.find_optimal(B.tolist(), k)
    H = np.rint(np.asarray(res.best.sub_basis, float) @ np.linalg.inv(B))
    assert abs(round(abs(np.linalg.det(H)))) == k, "индекс подрешётки не совпал"
    return H, res.normalized


def screen_champion(k):
    """Лучшая решётка симметричных семейств для индекса k (из артефактов экрана)."""
    import math
    from chromatic_research.campaigns.dim4_below43_screen import FAMILIES, lattice
    best = (0.0, None)
    try:
        screen = load_json("dim4_below43_screen.json")["families"]
    except FileNotFoundError:
        screen = {}
    for fam, rows in screen.items():
        row = rows.get(str(k))
        if row and row["d"] > best[0]:
            B = lattice(fam, *row["params"])
            if B is not None:
                best = (row["d"], B)
    # атлас: там же лежит рекорд k=42 (склеенный модуль Z[C3] (+) Z)
    try:
        atlas = load_json("dim4_symmetry_atlas.json")["atlas"]
    except FileNotFoundError:
        atlas = {}
    from chromatic_research.campaigns.dim4_symmetry_atlas import (CANDIDATES,
                                                                 invariant_basis)
    for name, blob in atlas.items():
        row = (blob.get("best") or {}).get(str(k))
        if not row or row["d"] <= best[0]:
            continue
        basis = invariant_basis(CANDIDATES[name])
        Q = sum(c * M for c, M in zip(row["t"], basis))
        Q = 0.5 * (Q + Q.T)
        if np.linalg.eigvalsh(Q).min() <= 1e-9:
            continue
        Q = Q / np.linalg.det(Q) ** 0.25
        try:
            best = (row["d"], np.array(combigeo.lll_reduce(
                np.linalg.cholesky(Q).tolist())))
        except (RuntimeError, np.linalg.LinAlgError):
            pass
    return best


def staged_polish(G, T, budget, label):
    """Постадийная полировка мозаики вокруг решёточной точки.

    Прямой CMA по всем 215 параметрам бесполезен: решёточная точка --- острый
    локальный максимум, и уже первое поколение при sigma = 0.01 теряет 8 %.
    Поэтому свобода включается по частям, от самой дешёвой к самой дорогой:

      W  --- только веса (k-1 параметр): ячейки меняют размер, узлы стоят;
      TW --- узлы и веса при неподвижной решётке периодов (n(k-1)+k-1);
      всё --- плюс сама решётка периодов.

    На каждой стадии стартом служит лучшая точка предыдущей.
    """
    import cma
    k = len(T)
    unit = (abs(np.linalg.det(G)) / k) ** (2.0 / N)
    scale = 0.25 * unit

    def width(sites, weights):
        try:
            rep = pc.evaluate(G, sites, weights)
        except Exception:
            return 0.0
        return rep.width if rep.sound and rep.width > 0 else 0.0

    base = width(T, np.zeros(k))
    best = (base, np.zeros(k), T.copy())
    print(f"    [{label}] точка посева: {base:.7f}", flush=True)

    def optimise(dim, x0, sigma, to_state, evals):
        nonlocal best
        es = cma.CMAEvolutionStrategy(list(x0), sigma,
                                      {"maxfevals": evals, "verbose": -9,
                                       "seed": 20260823})
        while not es.stop():
            xs = es.ask()
            fs = []
            for x in xs:
                sites, weights = to_state(np.asarray(x, float))
                d = width(sites, weights)
                fs.append(-d)
                if d > best[0]:
                    best = (d, weights.copy(), sites.copy())
            es.tell(xs, fs)
        return es

    # стадия W: только веса
    def state_w(x):
        w = np.concatenate([[0.0], x * scale])
        return T, w - w.mean()
    optimise(k - 1, np.zeros(k - 1), 0.4, state_w, budget)
    print(f"    [{label}] после весов: {best[0]:.7f}", flush=True)

    # стадия TW: узлы + веса при неподвижной решётке периодов
    w_best = best[1].copy()
    Ginv = np.linalg.inv(G)

    def state_tw(x):
        shift = x[:N * (k - 1)].reshape(k - 1, N)
        sites = T.copy()
        sites[1:] = T[1:] + shift @ G
        w = np.concatenate([[0.0], w_best[1:] + x[N * (k - 1):] * scale])
        return sites, w - w.mean()
    optimise(N * (k - 1) + (k - 1), np.zeros(N * (k - 1) + (k - 1)),
             0.02, state_tw, budget)
    print(f"    [{label}] после узлов+весов: {best[0]:.7f}", flush=True)
    return best[0]


def main():
    ks = ([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42])
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
    drops = int(sys.argv[3]) if len(sys.argv) > 3 else DROPS
    B43, T43 = headline_lattice()
    out = {}

    for k in ks:
        rows = []
        print(f"=== k = {k} ===", flush=True)

        d_lat, B = screen_champion(k)
        if B is not None:
            H, _ = best_sublattice(B, k)
            G, T = H @ B, pc.transversal(B, H)
            print(f"  посев (а): решёточный чемпион d={d_lat:.6f}", flush=True)
            t0 = time.time()
            d = staged_polish(G, T, budget, "а")
            print(f"  посев (а): итог {d:.7f}  [{time.time()-t0:.0f}s]", flush=True)
            rows.append({"seed": "lattice-champion", "d_seed": d_lat, "d": d})

        # (б) 43-раскраска минус один узел
        G43, T43full = T43 @ B43, pc.transversal(B43, T43)
        for drop in range(drops):
            T = np.delete(T43full, drop, axis=0)
            if len(T) != k:
                continue
            print(f"  посев (б) drop={drop}", flush=True)
            t0 = time.time()
            d = staged_polish(G43, T, budget, f"б{drop}")
            print(f"  посев (б) drop={drop}: итог {d:.7f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            rows.append({"seed": f"drop{drop}", "d": d})

        best = max((r["d"] for r in rows), default=0.0)
        out[str(k)] = {"best": best, "admissible": bool(best >= 1.0), "runs": rows}
        print(f"k={k}: ЛУЧШЕЕ {best:.7f}  {'ГОДНАЯ' if best >= 1 else ''}", flush=True)
        json.dump({"budget": budget, "results": out,
                   "note": "мозаичная атака на индексы ниже 43 в R^4 "
                           "(постадийная полировка решёточной точки)"},
                  open(results_path("dim4_tiling_below43.json"), "w"),
                  ensure_ascii=False, indent=1)
    print("записано:", results_path("dim4_tiling_below43.json"))


if __name__ == "__main__":
    main()
