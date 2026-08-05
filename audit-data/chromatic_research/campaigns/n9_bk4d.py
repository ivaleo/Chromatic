"""N9: диагностика best_killed в 4D ниже 45 (аналог жёсткого препятствия 5D/6D).

В 5D best_killed=1 (все k=120..139) и в 6D =2 (k=300..342) — «неустранимые»
запрещённые векторы, не снимаемые деформацией формы, что стало свидетельством
оптимальности 140/343. Вопрос: есть ли такая же картина в 4D при k=31..44,
или препятствие «мягкое» (best_killed скачет, форм-зависимо)?

Для форм-чемпионов и классики: min_conflicts_cost по всем структурам
факторгруппы, бюджет по калибровке (2000 шагов × 15 рестартов).
"""
import sys
import json
import time
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path
from chromatic_research.forms import norm_gram, unpack


def work(args):
    name, Qlist, k = args
    import combigeo
    from chromatic_research.core.cyclic_csp import forbidden_coords
    from chromatic_research.core.general_csp import invariant_factor_structures
    Q = np.array(Qlist)
    B = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
    cell = combigeo.voronoi_cell(B.tolist())
    F, diam = forbidden_coords(B, cell, 1.0, 4)
    best = None
    for e_list in invariant_factor_structures(k):
        found, bk, idx = combigeo.min_conflicts_cost(
            [list(f) for f in F], e_list, 4, max_steps=2000, restarts=15, seed=7)
        if best is None or bk < best[0]:
            best = (bk, e_list, found)
        if bk == 0:
            break
    return {"form": name, "k": k, "nF": len(F),
            "best_killed": best[0], "structure": best[1], "found": bool(best[2])}


if __name__ == "__main__":
    W45 = np.array(json.load(open(results_path("n6_push45.json")))["Q"])
    W46 = np.array(json.load(open(results_path("n4_push46.json")))["Q"])
    X44 = json.load(open(results_path("n7_push44.json")))["k44"]["x"]
    Q44 = unpack(np.asarray(X44))
    D4 = norm_gram(np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float))
    A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
    A4S = norm_gram(np.linalg.inv(np.linalg.cholesky(A4G)).T)

    forms = [("rec44", Q44), ("W45", W45), ("W46", W46), ("D4", D4), ("A4*", A4S)]
    ks = [44, 43, 42, 41, 40, 39, 36, 33, 31]
    jobs = [(nm, Q.tolist(), k) for k in ks for nm, Q in forms]
    t0 = time.time()
    out = []
    with Pool(2) as pool:  # основная кампания занимает 8 ядер
        for r in pool.imap(work, jobs):
            out.append(r)
            print(f"{r['form']:>5} k={r['k']}: best_killed={r['best_killed']}"
                  f" |F|={r['nF']} структура={r['structure']}", flush=True)
            json.dump(out, open(results_path("n9_bk4d.json"), "w"), indent=1)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)
