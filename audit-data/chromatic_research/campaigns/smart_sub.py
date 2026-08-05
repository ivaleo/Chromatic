"""Умный поиск подрешётки (в стиле Arman et al.) — БЕЗ полного перебора.

Для данной решётки Λ (форма Q, ячейка построена) ищем подрешётку Λ' минимального
индекса с d(Λ,Λ')>=1 (или макс. ширины d при индексе <= target). Идея: генераторы
берутся из коротких НЕзапрещённых векторов (d(v)>=1), сублаттис проверяется точно
(мин по границе достаточности). Стоимость — секунды, а не часы: снимает барьер 5D.
"""
import numpy as np
import combigeo
from voronoi4d import lattice_points_within, lll_reduce
from chromatic_research.paths import results_path


def d_of_sub(cell, diam, sub_basis):
    """d(Λ,Λ') для КОНКРЕТНОЙ подрешётки с готовой ячейкой."""
    sub_l = lll_reduce(np.asarray(sub_basis))
    # кратчайший вектор подрешётки
    v0 = min(lattice_points_within(sub_l, min(np.linalg.norm(r) for r in sub_l) + 1e-9),
             key=lambda w: float(w @ w))
    cur = 2.0 * combigeo.distance_to_cell((0.5 * v0).tolist(), cell)
    for v in sorted(lattice_points_within(sub_l, cur + diam), key=lambda w: float(w @ w)):
        if float(np.linalg.norm(v)) - diam >= cur:
            break
        cur = min(cur, 2.0 * combigeo.distance_to_cell((0.5 * v).tolist(), cell))
    return cur / diam


def smart_search(B, cell, dim, target_index, ntry=4000, ell=1.0, seed=0):
    """Ищет подрешётку минимального индекса с d>=ell (генераторы из незапрещённых).

    :return: (best_index, best_d, best_sub_basis) или (None,...) если не найдено.
    """
    diam = cell.diameter
    Binv = np.linalg.inv(B)
    # незапрещённые векторы (d(v)>=ell), в целочисленных координатах, по возрастанию |v|
    R = 2.3 * diam
    cand = []
    for v in lattice_points_within(B, R):
        d = 2.0 * combigeo.distance_to_cell((0.5 * v).tolist(), cell) / diam
        if d >= ell - 1e-9:
            c = np.rint(v @ Binv).astype(int)
            cand.append((float(v @ v), c))
    cand.sort(key=lambda t: t[0])
    coords = [c for _, c in cand]
    if len(coords) < dim:
        return None, 0.0, None
    C = np.array(coords, float)

    rng = np.random.default_rng(seed)
    best = (None, 0.0, None)   # (index, d, sub_basis)
    ncand = len(coords)
    # смещаем выбор к коротким (малый индекс): веса ~ 1/rank
    weights = 1.0 / (1.0 + np.arange(ncand))
    weights /= weights.sum()
    for _ in range(ntry):
        pick = rng.choice(ncand, size=dim, replace=False, p=weights)
        M = C[pick]
        det = abs(np.linalg.det(M))
        idx = int(round(det))
        if idx < 1 or abs(det - idx) > 1e-6:
            continue
        if idx > target_index:
            continue
        sub = M @ B
        d = d_of_sub(cell, diam, sub)
        if d >= ell - 1e-9:
            # приоритет: меньший индекс; при равном — большая ширина
            if best[0] is None or idx < best[0] or (idx == best[0] and d > best[1]):
                best = (idx, d, sub.tolist())
    return best


if __name__ == "__main__":
    import sys, time, json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    ntry = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    # A5* по умолчанию
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    B = np.linalg.cholesky(M.T @ M)
    B /= abs(np.linalg.det(B)) ** (1.0 / n)
    cell = combigeo.voronoi_cell(B.tolist())
    t = time.time()
    idx, d, sub = smart_search(B, cell, n, target, ntry=ntry, ell=1.0)
    print(f"A5* умный поиск: минимальный индекс с d>=1 = {idx}  (d={d:.5f})  "
          f"[{time.time()-t:.0f}s, {ntry} проб]", flush=True)
    print(f"(известный оптимум A5* = 140; <140 => СЕНСАЦИЯ, ~140 => метод валиден)")
    json.dump({"n": n, "index": idx, "d": d}, open(
        results_path(f"smart_a{n}star.json"), "w"))
    print("DONE")
