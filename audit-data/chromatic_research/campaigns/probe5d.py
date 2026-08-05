"""5D-диагностика: даёт ли генерическая (возмущённая) форма БОЛЬШЕЕ d, чем A5*,
на осуществимом индексе? Если да — есть шанс пробить 140 генерическими решётками.

На фиксированном индексе (по умолчанию 96, ~несколько минут исчерпывающе)
сравниваем d(A5*) с d нескольких возмущений A5*. Восходящий тренд => перспективно.
"""
import sys, time, json
import numpy as np
import combigeo
from chromatic_research.paths import results_path


def main():
    n = 5
    M = np.ones((n + 1, n))
    for j in range(n):
        M[j, j] = -n
    A5 = np.linalg.cholesky(M.T @ M)
    A5 /= abs(np.linalg.det(A5)) ** (1.0 / n)

    k = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    nforms = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    print(f"5D индекс k={k}, подрешёток={combigeo.count_sublattices(5,k)}", flush=True)
    rng = np.random.default_rng(5)
    results = []

    t0 = time.time()
    d0 = combigeo.find_optimal(A5.tolist(), index=k, threads=8).normalized
    print(f"A5* (эталон): d={d0:.5f}  [{time.time()-t0:.0f}s]", flush=True)
    results.append(("A5*", d0))
    best = d0

    for i in range(nforms):
        L = np.linalg.cholesky(A5 @ A5.T)
        P = L * (1 + rng.normal(scale=0.08, size=L.shape) * np.tri(n))
        Q = P @ P.T
        Q /= abs(np.linalg.det(Q)) ** (1.0 / n)
        try:
            B = np.linalg.cholesky(Q)
            t = time.time()
            d = combigeo.find_optimal(B.tolist(), index=k, threads=8).normalized
            results.append((f"pert{i}", d))
            mark = " <== выше A5*!" if d > d0 + 1e-4 else ""
            print(f"pert{i}: d={d:.5f}{mark}  [{time.time()-t:.0f}s]", flush=True)
            best = max(best, d)
        except Exception as e:
            print(f"pert{i}: ошибка {type(e).__name__}", flush=True)

    print(f"ИТОГ k={k}: max d = {best:.5f} (A5* давал {d0:.5f}); "
          f"{'генерич. ВЫШЕ — перспективно для <140' if best>d0+1e-3 else 'генерич. не выше A5*'}",
          flush=True)
    json.dump({"k": k, "results": results, "A5star": d0, "best": best},
              open(results_path(f"probe5d_k{k}.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
