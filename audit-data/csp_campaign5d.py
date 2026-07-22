"""Сильная параллельная кампания обобщённого CSP-поиска в 5D.

(A) валидация: найти A5*/140 (нециклическая Z/2×Z/70) — метод должен её найти;
(B) атака: для A5* и генерических форм искать подрешётку индекса k<140,
    избегающую F (любая структура факторгруппы). Найдено k<140 => χ(R^5)<140.
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")


def norm_gram(L, n):
    G = L @ L.T
    return G / abs(np.linalg.det(G)) ** (1.0 / n)


def work(args):
    """Для (форма, индекс k): ищет валидную подрешётку любой структуры."""
    Qlist, n, k, budget, seed = args
    import combigeo
    from cyclic_csp import forbidden_coords
    from general_csp import invariant_factor_structures, search_structure
    Q = np.array(Qlist)
    try:
        B = np.linalg.cholesky(Q)
        cell = combigeo.voronoi_cell(B.tolist())
    except Exception:
        return {"k": k, "found": False, "err": True}
    F, diam = forbidden_coords(B, cell, 1.0, n)
    for e_list in invariant_factor_structures(k):
        r = search_structure(F, e_list, n, k, ntry=budget, seed=seed)
        if r is not None:
            return {"k": k, "found": True, "structure": e_list, "phi": r, "nF": len(F)}
    return {"k": k, "found": False, "nF": len(F)}


if __name__ == "__main__":
    n = 5
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    A5 = norm_gram(np.linalg.cholesky(M.T @ M), n)

    if mode == "validate":
        # A5*/140 — восемь параллельных попыток с разными семенами (нециклич. структура)
        print("=== ВАЛИДАЦИЯ: A5*/140 (ожидаем Z/2×Z/70) ===", flush=True)
        jobs = [(A5.tolist(), n, 140, 300000, s) for s in range(8)]
        t0 = time.time()
        with Pool(8) as pool:
            res = pool.map(work, jobs)
        ok = [r for r in res if r.get("found")]
        if ok:
            print(f"НАЙДЕНА A5*/140: структура {ok[0]['structure']} [{time.time()-t0:.0f}s] "
                  f"=> метод валиден в 5D", flush=True)
        else:
            print(f"A5*/140 НЕ найдена за {time.time()-t0:.0f}s "
                  f"(|F|={res[0].get('nF')}) — хит-рейт слишком мал", flush=True)
        json.dump(res, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/csp_validate5d.json", "w"))
    else:
        # АТАКА: k = 139..120 для A5* + генерических форм
        rng = np.random.default_rng(11)
        forms = [("A5*", A5)]
        for i in range(5):
            L = np.linalg.cholesky(A5 @ A5.T)
            P = L * (1 + rng.normal(scale=0.10, size=L.shape) * np.tri(n))
            forms.append((f"g{i}", norm_gram(P, n)))
        best = 140
        out = {}
        with Pool(8) as pool:
            for k in range(139, 119, -1):
                jobs = [(Q.tolist(), n, k, 200000, si) for si, (_, Q) in enumerate(forms)]
                res = pool.map(work, jobs)
                hit = [(nm, r) for (nm, _), r in zip(forms, res) if r.get("found")]
                if hit:
                    nm, r = hit[0]
                    print(f"k={k}: НАЙДЕНА на {nm}, структура {r['structure']} — χ(R^5)<={k}!",
                          flush=True)
                    out[k] = {"form": nm, "structure": r["structure"], "phi": r["phi"]}
                    best = min(best, k)
                else:
                    print(f"k={k}: не найдено", flush=True)
        print(f"ИТОГ атаки: минимальный найденный индекс = {best} "
              f"({'НОВЫЙ РЕЗУЛЬТАТ <140!' if best < 140 else 'не ниже 140'})", flush=True)
        json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/csp_attack5d.json", "w"), indent=1)
    print("DONE", flush=True)
