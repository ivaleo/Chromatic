"""6D-оценщик БЕЗ перечисления вершин: ячейка задаётся опорными полупространствами
(релевантные векторы через CVP по классам Λ/2Λ), расстояния — квадратичным
программированием. Диаметр — точный для известных решёток, иначе оценка сверху.
Затем min-conflicts ищет валидную подрешётку. Цель: E6*/343 и попытка <343.
"""
import numpy as np
import math
from scipy.optimize import minimize


def gram_schmidt(B):
    n = len(B); Bs = B.astype(float).copy(); mu = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            mu[i, j] = B[i] @ Bs[j] / (Bs[j] @ Bs[j]); Bs[i] -= mu[i, j] * Bs[j]
    return Bs, mu


def lll(B, delta=0.75):
    B = B.astype(float).copy(); n = len(B); k = 1
    while k < n:
        Bs, mu = gram_schmidt(B)
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q: B[k] -= q * B[j]
        Bs, mu = gram_schmidt(B)
        if Bs[k] @ Bs[k] >= (delta - mu[k, k-1]**2) * (Bs[k-1] @ Bs[k-1]): k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]; k = max(k-1, 1)
    return B


def vectors_within(B, bound):
    B = lll(B); Bs, mu = gram_schmidt(B); bn2 = np.array([b @ b for b in Bs])
    n = len(B); out = []; coeffs = [0]*n
    def desc(level, p2):
        if level == 0:
            for c in coeffs:
                if c > 0: break
                if c < 0: return
            else: return
            v = np.array(coeffs, float) @ B
            if v @ v <= bound*bound + 1e-9: out.append(v)
            return
        j = level-1; center = sum(coeffs[i]*mu[i, j] for i in range(j+1, n))
        rem = bound*bound - p2
        if rem < -1e-9: return
        rad = math.sqrt(max(0.0, rem)/bn2[j])
        for c in range(math.ceil(-center-rad-1e-9), math.floor(-center+rad+1e-9)+1):
            coeffs[j] = c; desc(j, p2 + (c+center)**2*bn2[j])
        coeffs[j] = 0
    desc(n, 0.0); return out


def relevant_vectors(B, bound):
    n = len(B); Bl = lll(B); Binv = np.linalg.inv(Bl); coset = {}
    for v in vectors_within(Bl, bound):
        key = tuple(np.rint(v @ Binv).astype(int) % 2)
        if key == (0,)*n: continue
        coset.setdefault(key, []).append(v)
    assert len(coset) == 2**n - 1, f"{len(coset)} of {2**n-1}"
    rel = []
    for vs in coset.values():
        m = min(np.linalg.norm(v) for v in vs)
        ties = [v for v in vs if np.linalg.norm(v) <= m + 1e-9]
        if len(ties) == 1: rel.append(ties[0])
    return rel


def make_cell(B, diam_guess):
    """Возвращает (A, b) опорных полупространств x·A_i<=b_i и функцию dist_to_cell."""
    rel = relevant_vectors(B, diam_guess * 1.05 + 1e-6)
    A = np.array([w for w in rel] + [-w for w in rel])
    b = np.array([w @ w / 2 for w in rel] * 2)
    def dist_to_cell(p):
        if np.all(A @ p <= b + 1e-12): return 0.0
        cons = [{"type": "ineq", "fun": lambda x: b - A @ x, "jac": lambda x: -A}]
        best = None
        for s0 in (np.zeros(len(p)), p*0.5):
            r = minimize(lambda x: (x-p)@(x-p), s0, jac=lambda x: 2*(x-p),
                         constraints=cons, method="SLSQP", options={"maxiter":800,"ftol":1e-12})
            if r.success and np.all(A @ r.x <= b + 1e-9):
                d = float(np.linalg.norm(r.x-p)); best = d if best is None else min(best, d)
        return best if best is not None else float("nan")
    return A, b, dist_to_cell, len(rel)


def forbidden_set_6d(B, diam, dist_to_cell, ell=1.0):
    """Координаты запрещённых векторов (D(v)<ell*diam), D=2·dist(v/2,cell)."""
    Binv = np.linalg.inv(B)
    R = (ell + 1.0) * diam + 1e-6
    F = []
    for v in vectors_within(B, R):
        D = 2.0 * dist_to_cell(0.5 * v)
        if D == D and D < ell * diam - 1e-9:
            F.append(tuple(int(x) for x in np.rint(v @ Binv)))
    return F


if __name__ == "__main__":
    import sys, time, json
    sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")
    from general_csp import invariant_factor_structures, index_and_check
    from minconf_csp import min_conflicts

    n = 6
    # E6* (эйзенштейнова конструкция, как в verify_e6s)
    w = complex(-0.5, math.sqrt(3)/2); th = w - w.conjugate()
    def realify(vs):
        o = []
        for v in vs:
            r = []
            for z in v: r += [z.real, z.imag]
            o.append(r)
        return np.array(o)
    gens = []
    for u in [(th,0,0),(1,1,1),(0,th,0)]:
        gens.append(u); gens.append(tuple(w*z for z in u))
    B = np.linalg.inv(realify(gens)).T
    B = B / abs(np.linalg.det(B))**(1.0/n)
    # λ1: адаптивная граница — начинаем с мин. нормы строки LLL-базиса
    Bl = lll(B); start = min(np.linalg.norm(r) for r in Bl)
    lam1 = min(np.linalg.norm(v) for v in vectors_within(B, start + 1e-9))
    diam = 2 * lam1 / math.sqrt(2)          # ratio sqrt(2) для E6*

    t = time.time()
    A, b, dist_to_cell, nrel = make_cell(B, diam)
    print(f"E6* опорных полупространств: {nrel} (ожид. 126), diam={diam:.4f} [{time.time()-t:.1f}s]",
          flush=True)
    t = time.time()
    F = forbidden_set_6d(B, diam, dist_to_cell, ell=1.0)
    print(f"|F_1| = {len(F)}  [{time.time()-t:.0f}s]", flush=True)

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 343
    print(f"min-conflicts на E6*/{k} (структуры {invariant_factor_structures(k)})...", flush=True)
    found = None
    for e_list in invariant_factor_structures(k):
        if len(e_list) < 2 and k != 343:
            continue
        t = time.time()
        r = min_conflicts(F, e_list, n, max_steps=3000, restarts=20, seed=0)
        if r is not None:
            idx, av = index_and_check(r, e_list, F, n)
            if idx == k and av:
                found = (e_list, r)
                print(f"  НАЙДЕНА структура {e_list} (индекс {idx}) [{time.time()-t:.0f}s]",
                      flush=True); break
        print(f"  структура {e_list}: нет [{time.time()-t:.0f}s]", flush=True)
    print(f"E6*/{k}: {'валидная подрешётка найдена => метод валиден в 6D' if found else 'не найдена'}",
          flush=True)
    json.dump({"k": k, "nrel": nrel, "nF": len(F), "found": found is not None,
               "structure": found[0] if found else None},
              open(f"/Users/mac/Documents/_My_code/Chromatic/audit-data/dim6_k{k}.json", "w"))
    print("DONE", flush=True)
