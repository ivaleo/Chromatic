"""N6: агрессивный параллельный дожим k=45 (текущий максимум 0.99553 — недобор 0.45%).

24 старта на пуле из 8 процессов: базы — победитель-45, победитель-46, D4, A4*
и их возмущения; высокий бюджет NM. Каждый воркер — однопоточный C++ (параллелизм
на уровне процессов).
"""
import json
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path
from chromatic_research.forms import norm_gram, pack, unpack

def one_start(args):
    x0, seed, maxfev = args
    import combigeo
    from scipy.optimize import minimize
    def d_of(Q):
        if Q is None: return 0.0
        try:
            B = np.linalg.cholesky(Q + 1e-12*np.eye(4))
            return combigeo.find_optimal(B.tolist(), index=45, threads=1).normalized
        except Exception:
            return 0.0
    r = minimize(lambda x: -d_of(unpack(x, 4)), np.asarray(x0), method="Nelder-Mead",
                 options={"maxfev": maxfev, "xatol": 1e-9, "fatol": 1e-13})
    return (-r.fun, r.x.tolist())

if __name__ == "__main__":
    W45 = np.array(json.load(open(results_path("n5_cascade.json")))["k45"]["Q"])
    W46 = np.array(json.load(open(results_path("n4_push46.json")))["Q"])
    D4 = norm_gram(np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float))
    A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
    A4S = norm_gram(np.linalg.inv(np.linalg.cholesky(A4G)).T)

    rng = np.random.default_rng(45)
    bases = [pack(W45), pack(W46), pack(D4), pack(A4S)]
    jobs = []
    # плотные возмущения вокруг двух лучших + редкие вокруг классических
    for b, reps, scales in [(pack(W45), 8, (0.008, 0.02, 0.05)),
                            (pack(W46), 6, (0.02, 0.05, 0.1)),
                            (pack(D4), 3, (0.1, 0.2)),
                            (pack(A4S), 3, (0.1, 0.2))]:
        jobs.append((b.tolist(), 0, 1400))  # чистый старт
        for _ in range(reps):
            s = float(rng.choice(scales))
            jobs.append(((b * (1 + rng.normal(scale=s, size=10))).tolist(), 0, 1000))

    print(f"стартов: {len(jobs)}", flush=True)
    with Pool(8) as pool:
        results = pool.map(one_start, jobs)
    best_d, best_x = max(results, key=lambda t: t[0])
    over = sum(1 for d, _ in results if d >= 1.0)
    print(f"k=45: max d = {best_d:.7f}  {'>=1 ПРОБОЙ!' if best_d >= 1 else '< 1'}  "
          f"(пробили {over}/{len(jobs)})", flush=True)
    json.dump({"d": best_d, "Q": None if unpack(np.asarray(best_x)) is None
               else unpack(np.asarray(best_x)).tolist()},
              open(results_path("n6_push45.json"), "w"),
              indent=1)
    print("DONE", flush=True)
