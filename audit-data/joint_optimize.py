"""Вложенная совместная оптимизация (форма Q + циклическая подрешётка Γ) с CMA-ES.

Архитектура (снимает барьер больших размерностей):
  ВНЕШНИЙ уровень: CMA-ES по параметрам Холецкого формы Q (непрерывные).
  ВНУТРЕННИЙ уровень: для каждой Q ячейка Вороного строится ОДИН РАЗ, затем
     дёшево ищется лучшая циклическая подрешётка c (скан + локальное уточнение).
  Цель внешнего = max_c d(Q, c).

Стоимость одного внешнего вызова = одно построение ячейки + N дешёвых оценок Γ,
без перебора миллиардов подрешёток. В 5D ~1.6 c/вызов вместо ~50 мин.

Использование: python joint_optimize.py <dim> <k> [budget] [seed] [inner]
"""
import sys
import time
import numpy as np
import combigeo
from voronoi4d import lattice_points_within, lll_reduce, shortest_vector


def cholesky_unpack(x, dim):
    L = np.zeros((dim, dim)); L[np.tril_indices(dim)] = x
    Q = L @ L.T; d = abs(np.linalg.det(Q))
    return Q / d ** (1.0 / dim) if d > 1e-12 else None


def d_of_pair(B, cell, diam, c_int, k, dim):
    T = np.eye(dim); T[:dim - 1, dim - 1] = c_int; T[dim - 1, dim - 1] = k
    sub = T @ B
    sub_l = lll_reduce(sub)
    v0 = shortest_vector(sub_l)
    cur = 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(v0)).tolist(), cell)
    for v in sorted(lattice_points_within(sub_l, cur + diam), key=lambda w: float(w @ w)):
        if float(np.linalg.norm(v)) - diam >= cur:
            break
        cur = min(cur, 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(v)).tolist(), cell))
    return cur / diam


def best_cyclic(B, cell, k, dim, inner, rng, seed_cs=None):
    """Лучшая циклическая подрешётка индекса k для данной формы (скан + уточнение)."""
    diam = cell.diameter
    best_d, best_c = 0.0, None
    # затравки: переданные хорошие c + случайные
    cand = list(seed_cs or [])
    cand += [rng.integers(0, k, size=dim - 1) for _ in range(inner)]
    for c in cand:
        d = d_of_pair(B, cell, diam, np.asarray(c) % k, k, dim)
        if d > best_d:
            best_d, best_c = d, np.asarray(c) % k
    # локальное уточнение вокруг лучшей (±1 по каждой координате, несколько проходов)
    if best_c is not None:
        improved = True; passes = 0
        while improved and passes < 4:
            improved = False; passes += 1
            for i in range(dim - 1):
                for delta in (-1, 1, -2, 2):
                    c2 = best_c.copy(); c2[i] = (c2[i] + delta) % k
                    d = d_of_pair(B, cell, diam, c2, k, dim)
                    if d > best_d + 1e-9:
                        best_d, best_c, improved = d, c2, True
    return best_d, (None if best_c is None else best_c.tolist())


class Objective:
    def __init__(self, dim, k, inner, seed):
        self.dim, self.k, self.inner = dim, k, inner
        self.rng = np.random.default_rng(seed)
        self.m = dim * (dim + 1) // 2
        self.best_d, self.best_x, self.best_c = 0.0, None, None
        self.seed_cs = []  # переносим хорошие c между формами
    def __call__(self, x):
        Q = cholesky_unpack(np.asarray(x[:self.m]), self.dim)
        if Q is None:
            return 1.0
        try:
            B = np.linalg.cholesky(Q + 1e-12 * np.eye(self.dim))
            cell = combigeo.voronoi_cell(B.tolist())
        except Exception:
            return 1.0
        d, c = best_cyclic(B, cell, self.k, self.dim, self.inner, self.rng,
                           seed_cs=self.seed_cs[-3:])
        if d > self.best_d:
            self.best_d, self.best_x, self.best_c = d, np.asarray(x[:self.m]).copy(), c
            if c is not None:
                self.seed_cs.append(np.asarray(c))
            print(f"    d={d:.6f} c={c}", flush=True)
        return -d  # CMA минимизирует


def run(dim, k, budget=3000, seed=0, inner=150):
    import cma
    obj = Objective(dim, k, inner, seed)
    m = obj.m
    x0 = np.zeros(m)
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            x0[idx] = 1.0 if i == j else 0.0
            idx += 1
    lb = []; ub = []
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            if i == j: lb.append(0.2); ub.append(3.0)
            else: lb.append(-3.0); ub.append(3.0)
            idx += 1
    es = cma.CMAEvolutionStrategy(x0, 0.7, {
        "bounds": [lb, ub], "maxfevals": budget, "seed": seed + 1,
        "verbose": -9, "popsize": 12})
    while not es.stop():
        X = es.ask()
        es.tell(X, [obj(x) for x in X])
    return obj.best_d, obj.best_x, obj.best_c


if __name__ == "__main__":
    dim = int(sys.argv[1]); k = int(sys.argv[2])
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    inner = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    t0 = time.time()
    bd, bx, bc = run(dim, k, budget, seed, inner)
    print(f"dim={dim} k={k}: joint+CMA-ES max d = {bd:.7f}  c={bc}  "
          f"{'>=1 !!!' if bd >= 1 else '<1'}  ({time.time()-t0:.0f}s, budget={budget})",
          flush=True)
    import json
    Q = cholesky_unpack(bx, dim) if bx is not None else None
    json.dump({"dim": dim, "k": k, "d": bd, "c": bc,
               "Q": None if Q is None else Q.tolist()},
              open(f"/Users/mac/Documents/_My_code/Chromatic/audit-data/joint_d{dim}_k{k}.json", "w"),
              indent=1)
