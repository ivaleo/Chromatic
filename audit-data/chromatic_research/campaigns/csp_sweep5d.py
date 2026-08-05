"""CSP-развёртка по генерическим 5D-формам: для каждой ищем МИНИМАЛЬНЫЙ индекс k,
при котором существует циклическая подрешётка с d>=1 (через сравнения mod k).
Если найдётся k<140 — новая оценка χ(R^5)<140. Дёшево (без перебора подрешёток).
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path
from chromatic_research.forms import norm_gram


def min_cyclic_index(args):
    """Для формы Q: минимальный k in [k_lo,k_hi] с валидной циклической Γ_c (d>=1)."""
    Qlist, n, k_lo, k_hi, seed = args
    import combigeo
    from chromatic_research.core.cyclic_csp import forbidden_coords, find_cyclic
    Q = np.array(Qlist)
    try:
        B = np.linalg.cholesky(Q)
        cell = combigeo.voronoi_cell(B.tolist())
    except Exception:
        return None
    F, diam = forbidden_coords(B, cell, 1.0, n)   # запрещённые при ℓ=1
    if len(F) > 4000:
        return {"nF": len(F), "min_k": None}       # слишком много запретов
    # ищем минимальный k с валидной c
    for k in range(k_lo, k_hi + 1):
        c = find_cyclic(F, k, n, ntry=30000, seed=seed)
        if c is not None:
            return {"nF": len(F), "min_k": k, "c": c}
    return {"nF": len(F), "min_k": None}


if __name__ == "__main__":
    n = 5
    k_lo = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    k_hi = int(sys.argv[2]) if len(sys.argv) > 2 else 140
    nforms = int(sys.argv[3]) if len(sys.argv) > 3 else 24

    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    A5 = norm_gram(np.linalg.cholesky(M.T @ M) @ np.linalg.cholesky(M.T @ M).T * 0 +
                  (np.linalg.cholesky(M.T @ M)), n) if False else \
         (lambda L: norm_gram(L))(np.linalg.cholesky(M.T @ M))
    # базовые решётки
    D5 = norm_gram(np.array([[1,1,0,0,0],[1,-1,0,0,0],[0,1,-1,0,0],[0,0,1,-1,0],
                             [0,0,0,1,-1]], float), n)
    rng = np.random.default_rng(7)
    forms = [("A5*", A5), ("D5", D5)]
    for i in range(nforms - 2):
        base = A5 if i % 2 == 0 else D5
        L = np.linalg.cholesky(base)
        P = L * (1 + rng.normal(scale=0.12, size=L.shape) * np.tri(n))
        forms.append((f"g{i}", norm_gram(P)))

    print(f"CSP-развёртка 5D: {len(forms)} форм, индексы {k_lo}..{k_hi}", flush=True)
    jobs = [(Q.tolist(), n, k_lo, k_hi, s) for s, (_, Q) in enumerate(forms)]
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.map(min_cyclic_index, jobs)
    best = None
    out = {}
    for (name, _), r in zip(forms, res):
        if r is None:
            print(f"  {name}: ошибка построения", flush=True); continue
        mk = r.get("min_k")
        print(f"  {name}: |F_1|={r['nF']}, мин.циклич.индекс с d>=1 = {mk}", flush=True)
        out[name] = r
        if mk is not None and (best is None or mk < best):
            best = mk
    print(f"ИТОГ: минимальный найденный индекс циклич. раскраски R^5 = {best} "
          f"(эталон 140); {'НИЖЕ 140 — НОВЫЙ РЕЗУЛЬТАТ!' if best and best < 140 else 'не ниже 140'}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(results_path("csp_sweep5d.json"), "w"),
              indent=1)
    print("DONE", flush=True)
