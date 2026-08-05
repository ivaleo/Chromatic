"""N3: разведка в R^5 — можно ли пробить 140 (гипотеза Arman et al. для n=5)?

Дешёвый зонд: 300 случайных возмущений формы A5* (и D5, Z5-смесей); для каждой
формы оцениваем max d(k=139) по ОГРАНИЧЕННОМУ семейству подрешёток (HNF с 139
на позициях 1-2 диагонали: 1 + 139 = 140 матриц на форму) через python-конвейер
без перестройки ячейки. Если максимум ландшафта близок к 1 — цель для тяжёлой
атаки; если далеко — фиксируем отрицательную разведку.
"""
import json, math, time
import numpy as np
import combigeo

n, K = 5, 139
M = np.ones((n + 1, n), float)
for j in range(n):
    M[j, j] = -n
A5S = np.linalg.cholesky(M.T @ M)
A5S /= abs(np.linalg.det(A5S)) ** (1 / n)

def restricted_hnfs():
    """Выборка циклических HNF: 139 в ПОСЛЕДНЕЙ позиции диагонали (только такие
    семейства не содержат исходных базисных векторов), наддиагональ столбца 5 —
    случайные векторы c in [0,139)^4. Фиксированная выборка для всех форм."""
    rng0 = np.random.default_rng(97)
    mats = []
    for _ in range(800):
        T = np.eye(n)
        T[4, 4] = K
        T[0:4, 4] = rng0.integers(0, K, size=4)
        mats.append(T.copy())
    return mats

HNFS = restricted_hnfs()

from voronoi4d import lattice_points_within, lll_reduce, shortest_vector

def min_D(cell, diam, sub):
    """min_v 2*dist(v/2, V0) по подрешётке — с готовой ячейкой (без перестройки)."""
    sub_l = lll_reduce(sub)
    v0 = shortest_vector(sub_l)
    cur = 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(v0)).tolist(), cell)
    for v in sorted(lattice_points_within(sub_l, cur + diam),
                    key=lambda w: float(w @ w)):
        if float(np.linalg.norm(v)) - diam >= cur:
            break
        cur = min(cur, 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(v)).tolist(), cell))
    return cur

def eval_form(B):
    cell = combigeo.voronoi_cell(B.tolist())
    diam = cell.diameter
    best = 0.0
    for T in HNFS:
        best = max(best, min_D(cell, diam, T @ B) / diam)
    return best


def main():
    rng = np.random.default_rng(31)
    t0 = time.time()
    results = []
    base = eval_form(A5S)
    results.append(("A5*", base))
    print(f"A5* restricted d(139) = {base:.6f}", flush=True)
    best = (base, "A5*")
    for i in range(30):
        L = np.linalg.cholesky(A5S @ A5S.T)
        P = L * (1 + rng.normal(scale=0.10, size=L.shape) * np.tri(n))
        Q = P @ P.T
        Q /= abs(np.linalg.det(Q)) ** (1 / n)
        try:
            B = np.linalg.cholesky(Q)
            d = eval_form(B)
        except Exception:
            continue
        if d > best[0]:
            best = (d, f"pert{i}")
            print(f"  new best {d:.6f} at pert{i} [{time.time()-t0:.0f}s]", flush=True)
    print(f"probe done: best restricted d(139) = {best[0]:.6f} ({best[1]}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump({"best": best[0], "tag": best[1]},
              open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n3_5d_probe.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
