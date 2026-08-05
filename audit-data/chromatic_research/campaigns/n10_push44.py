"""N10: дожим k=44 от CMA-чемпиона (d=0.98957) — рецепт, пробивший 45 в n6.

Агрессивный NM-мультистарт вокруг лучшей формы CMA (n8) + возмущения разных
масштабов + старты от W45. Работает на 5 ядрах параллельно с лестницей n8.
"""
import json
import time
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path

def unpack(x):
    L = np.zeros((4, 4)); L[np.tril_indices(4)] = x
    Q = L @ L.T; d = abs(np.linalg.det(Q))
    return Q / d ** 0.25 if d > 1e-12 else None


def pack(Q):
    return np.linalg.cholesky(Q)[np.tril_indices(4)]


def one_start(args):
    x0, maxfev = args
    import combigeo
    from scipy.optimize import minimize
    def d_of(Q):
        if Q is None: return 0.0
        try:
            B = np.linalg.cholesky(Q + 1e-12*np.eye(4))
            return combigeo.find_optimal(B.tolist(), index=44, threads=1).normalized
        except Exception:
            return 0.0
    r = minimize(lambda x: -d_of(unpack(x)), np.asarray(x0), method="Nelder-Mead",
                 options={"maxfev": maxfev, "xatol": 1e-9, "fatol": 1e-13})
    return (-r.fun, r.x.tolist())


if __name__ == "__main__":
    C44 = np.array(json.load(open(results_path("n8_cma44_ladder.json")))["k44"]["Q"])
    W45 = np.array(json.load(open(results_path("n6_push45.json")))["Q"])
    rng = np.random.default_rng(4410)
    jobs = [(list(pack(C44)), 1600)]
    for b, reps, scales in [(pack(C44), 10, (0.005, 0.015, 0.04, 0.08)),
                            (pack(W45), 4, (0.02, 0.06))]:
        for _ in range(reps):
            s = float(rng.choice(scales))
            jobs.append(((b * (1 + rng.normal(scale=s, size=10))).tolist(), 1200))
    print(f"стартов: {len(jobs)}", flush=True)
    t0 = time.time()
    with Pool(5) as pool:
        results = pool.map(one_start, jobs)
    best_d, best_x = max(results, key=lambda t: t[0])
    over = sum(1 for d, _ in results if d >= 1.0)
    alld = sorted(round(d, 5) for d, _ in results)
    print(f"k=44 дожим: max d = {best_d:.7f}  {'>=1 ПРОБОЙ!' if best_d >= 1 else '<1'}  "
          f"(пробили {over}/{len(jobs)})  топ5={alld[-5:]}  ({time.time()-t0:.0f}s)", flush=True)
    Q = unpack(np.asarray(best_x))
    json.dump({"d": best_d, "Q": None if Q is None else Q.tolist(),
               "instances": alld},
              open(results_path("n10_push44.json"), "w"), indent=1)
    print("DONE", flush=True)
