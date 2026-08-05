"""N5-parallel: каскад k=45 -> 44 -> 43 (+ ретрай 47) с пулом процессов (8 воркеров)."""
import json
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path

def unpack(x):
    L = np.zeros((4, 4))
    L[np.tril_indices(4)] = x
    Q = L @ L.T
    d = abs(np.linalg.det(Q))
    return Q / d ** 0.25 if d > 1e-12 else None

def pack(Q):
    return np.linalg.cholesky(Q)[np.tril_indices(4)]

def one_start(args):
    """Один NM-старт (в отдельном процессе)."""
    k, s, maxfev, seed = args
    import combigeo
    from scipy.optimize import minimize
    def d_of(Q):
        if Q is None: return 0.0
        try:
            B = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
            return combigeo.find_optimal(B.tolist(), index=k).normalized
        except Exception:
            return 0.0
    r = minimize(lambda x: -d_of(unpack(x)), np.asarray(s), method="Nelder-Mead",
                 options={"maxfev": maxfev, "xatol": 1e-8, "fatol": 1e-12})
    return (-r.fun, r.x.tolist())

def push(pool, k, x0, maxfev=650, nstarts=9, scale_list=(0.01, 0.03, 0.06, 0.1)):
    rng = np.random.default_rng(1000 + k)
    starts = [np.asarray(x0)]
    while len(starts) < nstarts:
        s = float(rng.choice(scale_list))
        starts.append(np.asarray(x0) * (1 + rng.normal(scale=s, size=10)))
    jobs = [(k, s.tolist(), maxfev, i) for i, s in enumerate(starts)]
    results = pool.map(one_start, jobs)
    best_d, best_x = max(results, key=lambda t: t[0])
    print(f"k={k}: max d = {best_d:.7f}  {'>=1 ПРОБОЙ' if best_d >= 1 else '< 1'}  "
          f"(из {nstarts} стартов)", flush=True)
    return best_d, best_x

if __name__ == "__main__":
    W46 = np.array(json.load(
        open(results_path("n4_push46.json")))["Q"])
    out = {}
    with Pool(8) as pool:
        prev_x = pack(W46).tolist()
        for k in (45, 44, 43):
            d, xw = push(pool, k, prev_x, maxfev=650, nstarts=10)
            out[f"k{k}"] = {"d": d, "Q": None if unpack(np.asarray(xw)) is None
                            else unpack(np.asarray(xw)).tolist()}
            if d >= 1.0:
                prev_x = xw
            else:
                break
        d47, x47 = push(pool, 47, pack(W46).tolist(), maxfev=550, nstarts=8)
        out["k47"] = {"d": d47, "Q": None if unpack(np.asarray(x47)) is None
                      else unpack(np.asarray(x47)).tolist()}
    json.dump(out, open(results_path("n5_cascade.json"),
                        "w"), indent=1)
    print("DONE", flush=True)
