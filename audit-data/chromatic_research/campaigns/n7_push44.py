"""N7: параллельный дожим k=44 от победителя-45 (45 пробит с запасом 1.7%, 44=2²·11 может пробиться)."""
import json
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path

def unpack(x):
    L = np.zeros((4, 4)); L[np.tril_indices(4)] = x
    Q = L @ L.T; d = abs(np.linalg.det(Q))
    return Q / d ** 0.25 if d > 1e-12 else None

def pack(Q):
    return np.linalg.cholesky(Q)[np.tril_indices(4)]

def norm_gram(M):
    G = M @ M.T
    return G / abs(np.linalg.det(G)) ** 0.25

def one_start(args):
    x0, k, maxfev = args
    import combigeo
    from scipy.optimize import minimize
    def d_of(Q):
        if Q is None: return 0.0
        try:
            B = np.linalg.cholesky(Q + 1e-12*np.eye(4))
            return combigeo.find_optimal(B.tolist(), index=k, threads=1).normalized
        except Exception:
            return 0.0
    r = minimize(lambda x: -d_of(unpack(x)), np.asarray(x0), method="Nelder-Mead",
                 options={"maxfev": maxfev, "xatol": 1e-9, "fatol": 1e-13})
    return (-r.fun, r.x.tolist())

if __name__ == "__main__":
    W45 = np.array(json.load(open(results_path("n6_push45.json")))["Q"])
    W46 = np.array(json.load(open(results_path("n4_push46.json")))["Q"])
    D4 = norm_gram(np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float))
    rng = np.random.default_rng(44)
    out = {}
    with Pool(8) as pool:
        for k in (44, 43):
            base = pack(W45) if k == 44 else (pack(unpack(np.asarray(out["k44"]["x"])))
                                              if out.get("k44", {}).get("d", 0) >= 1 else pack(W45))
            jobs = [(base.tolist(), k, 1400)]
            for b, reps, scales in [(pack(W45), 10, (0.008, 0.02, 0.05, 0.1)),
                                    (pack(W46), 5, (0.03, 0.08)),
                                    (pack(D4), 3, (0.15,))]:
                for _ in range(reps):
                    s = float(rng.choice(scales))
                    jobs.append(((b * (1 + rng.normal(scale=s, size=10))).tolist(), k, 1000))
            print(f"k={k}: стартов {len(jobs)}", flush=True)
            results = pool.map(one_start, jobs)
            best_d, best_x = max(results, key=lambda t: t[0])
            over = sum(1 for d, _ in results if d >= 1.0)
            print(f"k={k}: max d = {best_d:.7f}  {'>=1 ПРОБОЙ!' if best_d >= 1 else '< 1'}  "
                  f"(пробили {over}/{len(jobs)})", flush=True)
            out[f"k{k}"] = {"d": best_d, "x": best_x,
                            "Q": None if unpack(np.asarray(best_x)) is None
                            else unpack(np.asarray(best_x)).tolist()}
            if best_d < 1.0:
                break
    json.dump(out, open(results_path("n7_push44.json"), "w"),
              indent=1)
    print("DONE", flush=True)
