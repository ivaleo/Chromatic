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
from voronoi4d import lattice_points_within, lll_reduce
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


def _candidates(rng, k, length, ntry):
    """Пробы для вектора коэффициентов: сначала структурные (1, t, t², …) mod k,
    затем случайные.  Ленивый генератор: при раннем успехе лишние пробы (их до
    ntry штук) не материализуются, а последовательность остаётся той же."""
    structured = min(k, 400) - 1
    for t in range(1, min(k, 400)):
        yield np.array([pow(t, i + 1, k) for i in range(length)], np.int64)
    for _ in range(ntry - structured):
        yield rng.integers(0, k, size=length).astype(np.int64)


def find_cyclic(F, k, dim, ntry=200000, seed=0):
    """Ищет c∈(Z/k)^{n-1} с Σc_i f_i ≢ f_n (mod k) для всех f∈F. None если не найдено."""
    if not F:
        return [0] * (dim - 1)              # запретов нет — любая c годится
    Farr = np.array(F, dtype=np.int64)       # (|F|, n)
    fhead = Farr[:, :dim - 1] % k            # коэффициенты при c
    ftail = Farr[:, dim - 1] % k
    rng = np.random.default_rng(seed)
    for c in _candidates(rng, k, dim - 1, ntry):
        resid = (fhead @ c - ftail) % k       # для каждого f: Σc_i f_i - f_n mod k
        if np.all(resid != 0):
            return [int(x) for x in c]
    return None


def best_cyclic_csp(B, cell, k, dim, ell_lo=1.0, ell_hi=None, steps=18, ntry=60000):
    """Максимизирует ℓ: бинарный поиск наибольшего ℓ, при котором ∃ циклическая Γ_c
    индекса k с d≥ℓ. Возвращает (best_ell, best_c)."""
    lo = ell_lo
    hi = 3.0 if ell_hi is None else ell_hi   # щедрая верхняя оценка ширины
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
