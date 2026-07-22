"""Атака на 140 в 5D: min-conflicts (быстрый внутренний оценщик) + поиск формы.

Для составных индексов k<140 перебираем формы (возмущения A5*/D5 + случайные) и
запускаем min-conflicts по НЕциклическим структурам факторгруппы. Первый успех =>
валидная подрешётка индекса k => χ(R^5)<=k (при d>=1). Проверяем точную ширину.
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")


def norm_gram(L, n):
    G = L @ L.T
    return G / abs(np.linalg.det(G)) ** (1.0 / n)


def work(args):
    Qlist, n, k, budget, seed = args
    import combigeo
    from cyclic_csp import forbidden_coords
    from general_csp import invariant_factor_structures, index_and_check
    from minconf_csp import min_conflicts
    Q = np.array(Qlist)
    try:
        B = np.linalg.cholesky(Q)
        cell = combigeo.voronoi_cell(B.tolist())
    except Exception:
        return None
    F, diam = forbidden_coords(B, cell, 1.0, n)
    # только НЕциклические структуры (>=2 факторов) — цикл для симметричных не работает и медленный
    structs = [e for e in invariant_factor_structures(k) if len(e) >= 2]
    for e_list in structs:
        r = min_conflicts(F, e_list, n, max_steps=budget, restarts=6, seed=seed)
        if r is not None:
            idx, av = index_and_check(r, e_list, F, n)
            if idx == k and av:
                return {"k": k, "structure": e_list, "phi": r, "nF": len(F)}
    return None


if __name__ == "__main__":
    n = 5
    ks = [int(x) for x in sys.argv[1:]] or [138, 135, 132, 130, 128, 126, 124, 120]
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    A5 = norm_gram(np.linalg.cholesky(M.T @ M), n)
    D5 = norm_gram(np.array([[1,1,0,0,0],[1,-1,0,0,0],[0,1,-1,0,0],[0,0,1,-1,0],
                             [0,0,0,1,-1]], float), n)
    rng = np.random.default_rng(23)
    forms = [("A5*", A5), ("D5", D5)]
    for i in range(14):
        base = A5 if i % 2 == 0 else D5
        L = np.linalg.cholesky(base)
        P = L * (1 + rng.normal(scale=0.10, size=L.shape) * np.tri(n))
        forms.append((f"g{i}", norm_gram(P, n)))

    best = 140
    out = {}
    with Pool(8) as pool:
        for k in sorted(ks, reverse=True):
            jobs = [(Q.tolist(), n, k, 4000, si) for si, (_, Q) in enumerate(forms)]
            t0 = time.time()
            res = pool.map(work, jobs)
            hits = [(nm, r) for (nm, _), r in zip(forms, res) if r is not None]
            if hits:
                nm, r = hits[0]
                print(f"k={k}: ВАЛИДНАЯ подрешётка на {nm}! структура {r['structure']}  "
                      f"=> χ(R^5)<={k}  [{time.time()-t0:.0f}s]", flush=True)
                out[k] = {"form": nm, "structure": r["structure"], "phi": r["phi"]}
                best = min(best, k)
            else:
                print(f"k={k}: не найдено ни на одной из {len(forms)} форм [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"ИТОГ: минимальный найденный валидный индекс в 5D = {best} "
          f"({'НОВЫЙ РЕЗУЛЬТАТ <140!' if best < 140 else 'не ниже 140'})", flush=True)
    json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/mc_attack5d.json", "w"),
              indent=1)
    print("DONE", flush=True)
