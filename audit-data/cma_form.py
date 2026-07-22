"""CMA-ES по формам Грама с ТОЧНОЙ оценкой (exhaustive find_optimal).

Замена Нелдеру-Миду: CMA-ES (адаптация ковариации) — сильный глобальный метод
для многоэкстремальных чёрных ящиков. Оценка d(Q,k) — точная (все подрешётки
индекса k), поэтому пригодно там, где перебор подрешёток дёшев (n<=4).

Параллельно 8 инстансов на пуле процессов; каждый — независимый CMA-ES с
рестартами (IPOP-стиль) из разных затравок.

Использование: python cma_form.py <dim> <k> [budget] [n_instances]
"""
import sys
import json
import time
import numpy as np
from multiprocessing import Pool


def cholesky_unpack(x, dim):
    L = np.zeros((dim, dim)); L[np.tril_indices(dim)] = x
    Q = L @ L.T; d = abs(np.linalg.det(Q))
    return Q / d ** (1.0 / dim) if d > 1e-12 else None


def bounds(dim):
    lb, ub = [], []
    for i in range(dim):
        for j in range(i + 1):
            if i == j: lb.append(0.15); ub.append(3.0)
            else: lb.append(-3.0); ub.append(3.0)
    return lb, ub


def warm_starts(dim):
    """Классические решётки как затравки."""
    outs = []
    if dim == 4:
        mats = {
            "D4": [[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],
            "A4s": None,
        }
        A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
        mats["A4s"] = np.linalg.inv(np.linalg.cholesky(A4G)).T.tolist()
        for M in mats.values():
            M = np.array(M, float); G = M @ M.T
            Gn = G / abs(np.linalg.det(G)) ** (1.0 / dim)
            outs.append(np.linalg.cholesky(Gn)[np.tril_indices(dim)])
    return outs


def one_instance(args):
    dim, k, budget, seed, x0 = args
    import combigeo, cma
    def f(x):
        Q = cholesky_unpack(np.asarray(x), dim)
        if Q is None: return 1.0
        try:
            B = np.linalg.cholesky(Q + 1e-12 * np.eye(dim))
            return -combigeo.find_optimal(B.tolist(), index=k, threads=1).normalized
        except Exception:
            return 1.0
    lb, ub = bounds(dim)
    m = dim * (dim + 1) // 2
    if x0 is None:
        x0 = np.zeros(m); idx = 0
        for i in range(dim):
            for j in range(i + 1):
                x0[idx] = 1.0 if i == j else 0.0; idx += 1
    es = cma.CMAEvolutionStrategy(list(x0), 0.6, {
        "bounds": [lb, ub], "maxfevals": budget, "seed": seed + 1,
        "verbose": -9, "popsize": 10})
    best_d, best_x = 0.0, None
    while not es.stop():
        X = es.ask()
        vals = [f(x) for x in X]
        es.tell(X, vals)
        i = int(np.argmin(vals))
        if -vals[i] > best_d:
            best_d, best_x = -vals[i], X[i]
    return (best_d, None if best_x is None else list(best_x))


if __name__ == "__main__":
    dim = int(sys.argv[1]); k = int(sys.argv[2])
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
    ninst = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    ws = warm_starts(dim)
    jobs = []
    for s in range(ninst):
        x0 = ws[s % len(ws)] if ws and s < 2 else None  # 2 тёплых, остальные холодные
        jobs.append((dim, k, budget, s, None if x0 is None else list(x0)))
    t0 = time.time()
    with Pool(min(ninst, 8)) as pool:
        res = pool.map(one_instance, jobs)
    bd, bx = max(res, key=lambda t: t[0])
    alld = sorted(round(d, 5) for d, _ in res)
    print(f"dim={dim} k={k}: CMA-ES по формам max d = {bd:.7f}  "
          f"{'>=1 ПРОБОЙ!' if bd >= 1 else '<1'}  инстансы={alld}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    Q = cholesky_unpack(np.asarray(bx), dim) if bx else None
    json.dump({"dim": dim, "k": k, "d": bd, "Q": None if Q is None else Q.tolist()},
              open(f"/Users/mac/Documents/_My_code/Chromatic/audit-data/cma_d{dim}_k{k}.json", "w"),
              indent=1)
    print("DONE", flush=True)
