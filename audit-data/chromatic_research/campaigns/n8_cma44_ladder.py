"""N8: добивочная кампания 4D ниже 45 — CMA-ES по формам Грама (точный оценщик).

Контекст: k=44 атаковали лишь Нелдером-Мидом (n7, 19 стартов, max 0.9656) ДО
появления cma_form.py; систематической лестницы k=31..43 (аналог o2_r3 в 3D)
в 4D не было. Здесь:
  Фаза 1 — тяжёлая атака k=44: 8 CMA-инстансов × 2200 оценок, тёплые старты от
            NM-рекорда-44, W45, W46, D4, A4* + холодные.
  Фаза 2 — лестница k=31..43: по 3 инстанса × 700 оценок (W45, рекорд-44, холодный).
Результаты пишутся инкрементально в n8_cma44_ladder.json.
"""
import json
import time
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path

OUT = results_path("n8_cma44_ladder.json")


def unpack(x):
    L = np.zeros((4, 4)); L[np.tril_indices(4)] = x
    Q = L @ L.T; d = abs(np.linalg.det(Q))
    return Q / d ** 0.25 if d > 1e-12 else None


def pack(Q):
    return np.linalg.cholesky(Q)[np.tril_indices(4)]


def norm_gram(M):
    G = M @ M.T
    return G / abs(np.linalg.det(G)) ** 0.25


def bounds():
    lb, ub = [], []
    for i in range(4):
        for j in range(i + 1):
            if i == j: lb.append(0.15); ub.append(3.0)
            else: lb.append(-3.0); ub.append(3.0)
    return lb, ub


def one_instance(args):
    k, budget, seed, x0, sigma = args
    import combigeo, cma
    def f(x):
        Q = unpack(np.asarray(x))
        if Q is None: return 1.0
        try:
            B = np.linalg.cholesky(Q + 1e-12 * np.eye(4))
            return -combigeo.find_optimal(B.tolist(), index=k, threads=1).normalized
        except Exception:
            return 1.0
    lb, ub = bounds()
    if x0 is None:
        x0 = [1.0 if i == j else 0.0 for i in range(4) for j in range(i + 1)]
    es = cma.CMAEvolutionStrategy(list(x0), sigma, {
        "bounds": [lb, ub], "maxfevals": budget, "seed": seed + 1,
        "verbose": -9, "popsize": 12})
    best_d, best_x = 0.0, None
    while not es.stop():
        X = es.ask()
        vals = [f(x) for x in X]
        es.tell(X, vals)
        i = int(np.argmin(vals))
        if -vals[i] > best_d:
            best_d, best_x = -vals[i], list(X[i])
        if best_d >= 1.0005:  # пробой с запасом — дальше не жжём бюджет
            break
    return (best_d, best_x)


def run_k(pool, k, jobs, out, tag):
    if f"k{k}" in out:  # докатка после паузы: уже посчитано
        print(f"k={k}: пропуск (уже в {OUT})", flush=True)
        return out[f"k{k}"]["d"]
    t0 = time.time()
    results = pool.map(one_instance, jobs)
    best_d, best_x = max(results, key=lambda t: t[0])
    alld = sorted(round(d, 5) for d, _ in results)
    Q = unpack(np.asarray(best_x)) if best_x is not None else None
    out[f"k{k}"] = {"d": best_d, "instances": alld,
                    "Q": None if Q is None else Q.tolist(),
                    "phase": tag, "secs": round(time.time() - t0)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"k={k}: max d = {best_d:.7f}  {'>=1 ПРОБОЙ!' if best_d >= 1 else '<1'}"
          f"  инстансы={alld}  ({time.time()-t0:.0f}s)", flush=True)
    return best_d


if __name__ == "__main__":
    W45 = np.array(json.load(open(results_path("n6_push45.json")))["Q"])
    W46 = np.array(json.load(open(results_path("n4_push46.json")))["Q"])
    X44 = json.load(open(results_path("n7_push44.json")))["k44"]["x"]  # NM-рекорд 0.9656
    D4 = norm_gram(np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float))
    A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
    A4S = norm_gram(np.linalg.inv(np.linalg.cholesky(A4G)).T)

    out = {}
    try:
        out = json.load(open(OUT))  # докатка после паузы
    except Exception:
        pass
    with Pool(8) as pool:
        # Фаза 1: k=44, 8 инстансов
        jobs44 = [
            (44, 2200, 101, X44, 0.12),
            (44, 2200, 102, list(pack(W45)), 0.15),
            (44, 2200, 103, list(pack(W46)), 0.2),
            (44, 2200, 104, list(pack(D4)), 0.35),
            (44, 2200, 105, list(pack(A4S)), 0.35),
            (44, 2200, 106, None, 0.6),
            (44, 2200, 107, None, 0.6),
            (44, 2200, 108, X44, 0.3),
        ]
        d44 = run_k(pool, 44, jobs44, out, "attack")

        # Фаза 2: лестница k=43..31 (сверху вниз — самые перспективные раньше)
        for k in range(43, 30, -1):
            jobs = [
                (k, 700, 200 + k, list(pack(W45)), 0.2),
                (k, 700, 300 + k, X44, 0.2),
                (k, 700, 400 + k, None, 0.6),
            ]
            run_k(pool, k, jobs, out, "ladder")
    print("DONE", flush=True)
