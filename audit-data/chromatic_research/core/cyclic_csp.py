"""Эффективный поиск ЦИКЛИЧЕСКОЙ подрешётки через сравнения по модулю k (CSP).

Ключ (снимает барьер больших размерностей): циклическая Γ_c индекса k задаётся
последним столбцом HNF c=(c_1..c_{n-1}). Вектор f∈Λ лежит в Γ_c  ⟺
    f_n - Σ c_i f_i ≡ 0 (mod k)   [f_i — координаты f в базисе Λ].
Значит Γ_c ИЗБЕГАЕТ запрещённого множества F_ℓ={v: D(v)<ℓ·diam}  ⟺
    Σ c_i f_i ≢ f_n (mod k)   для всех f∈F_ℓ.
Каждый f исключает гиперплоскость mod k; ищем c в дополнении их объединения.
Тогда d(Γ_c) ≥ ℓ ТОЧНО. F_ℓ конечно (D(v)≥|v|-diam ⟹ |v|<(ℓ+1)diam).

Стоимость: F_ℓ (короткие векторы, дёшево) + поиск c (проверка |F_ℓ| сравнений).
Никакого перебора миллиардов подрешёток.
"""
import numpy as np
import combigeo
from voronoi4d import lattice_points_within
from chromatic_research.paths import results_path


def forbidden_coords(B, cell, ell, dim):
    """Координаты (в базисе Λ) векторов с D(v) < ell*diam, плюс их точные D."""
    diam = cell.diameter
    Binv = np.linalg.inv(B)
    R = (ell + 1.0) * diam + 1e-6           # |v| < (ell+1)diam  (т.к. D>=|v|-diam)
    F = []
    for v in lattice_points_within(B, R):    # по одному из пары ±v
        D = 2.0 * combigeo.distance_to_cell((0.5 * v).tolist(), cell)
        if D < ell * diam - 1e-9:
            c = np.rint(v @ Binv).astype(np.int64)
            F.append(tuple(int(x) for x in c))
    return F, diam


def find_cyclic(F, k, dim, ntry=200000, seed=0):
    """Ищет c∈(Z/k)^{n-1} с Σc_i f_i ≢ f_n (mod k) для всех f∈F. None если не найдено."""
    if not F:
        return [0] * (dim - 1)              # запретов нет — любая c годится
    Farr = np.array(F, dtype=np.int64)       # (|F|, n)
    fhead = Farr[:, :dim - 1] % k            # коэффициенты при c
    ftail = Farr[:, dim - 1] % k
    rng = np.random.default_rng(seed)
    # сначала пробуем структурировано: c = (1, t, t^2, ...) mod k при разных t
    trials = [np.array([pow(t, i + 1, k) for i in range(dim - 1)], np.int64)
              for t in range(1, min(k, 400))]
    # затем случайные
    for _ in range(ntry - len(trials)):
        trials.append(rng.integers(0, k, size=dim - 1).astype(np.int64))
    for c in trials:
        resid = (fhead @ c - ftail) % k       # для каждого f: Σc_i f_i - f_n mod k
        if np.all(resid != 0):
            return [int(x) for x in c]
    return None


def best_cyclic_csp(B, cell, k, dim, ell_lo=1.0, ell_hi=None, steps=18, ntry=60000):
    """Максимизирует ℓ: бинарный поиск наибольшего ℓ, при котором ∃ циклическая Γ_c
    индекса k с d≥ℓ. Возвращает (best_ell, best_c)."""
    diam = cell.diameter
    if ell_hi is None:
        # верхняя оценка ширины: λ1(подрешётки) не больше ~ ... берём щедро
        ell_hi = 3.0
    best_ell, best_c = 0.0, None
    lo, hi = ell_lo, ell_hi
    # сначала проверим осуществимость при ell_lo
    F, _ = forbidden_coords(B, cell, lo, dim)
    c = find_cyclic(F, k, dim, ntry=ntry)
    if c is None:
        return 0.0, None
    best_ell, best_c = lo, c
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        F, _ = forbidden_coords(B, cell, mid, dim)
        c = find_cyclic(F, k, dim, ntry=ntry)
        if c is not None:
            best_ell, best_c, lo = mid, c, mid
        else:
            hi = mid
    return best_ell, best_c


if __name__ == "__main__":
    import sys, time, json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    # A5*
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    B = np.linalg.cholesky(M.T @ M); B /= abs(np.linalg.det(B)) ** (1.0 / n)
    cell = combigeo.voronoi_cell(B.tolist())
    t = time.time()
    # (1) осуществимость d>=1
    F, diam = forbidden_coords(B, cell, 1.0, n)
    print(f"|F_1| (запрещённых при ℓ=1) = {len(F)}, diam={diam:.4f} [{time.time()-t:.1f}s]")
    c = find_cyclic(F, k, n, ntry=80000)
    print(f"циклическая Γ_c индекса {k} с d>=1: {'НАЙДЕНА c='+str(c) if c else 'НЕ найдена'}")
    if c:
        # точная ширина этой c
        from chromatic_research.campaigns.smart_sub import d_of_sub
        T = np.eye(n); T[:n-1, n-1] = c; T[n-1, n-1] = k
        d = d_of_sub(cell, diam, (T @ B).tolist())
        print(f"   её точная ширина d = {d:.5f}")
    # (2) максимизируем ширину
    t = time.time()
    ell, cc = best_cyclic_csp(B, cell, k, n)
    print(f"макс. ширина циклической при индексе {k}: ℓ≈{ell:.4f} c={cc} "
          f"[{time.time()-t:.0f}s]", flush=True)
    json.dump({"n": n, "k": k, "feasible_d1": c is not None, "max_ell": ell, "c": cc},
              open(results_path(f"csp_a{n}star_k{k}.json"), "w"))
    print("DONE")
