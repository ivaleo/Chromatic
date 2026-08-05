"""N4: интенсивный дожим k=46 (текущий максимум 0.9982958 — до пробоя 0.17%)."""
import json
import numpy as np
from scipy.optimize import minimize
import combigeo

def unpack(x):
    L = np.zeros((4, 4))
    L[np.tril_indices(4)] = x
    Q = L @ L.T
    d = abs(np.linalg.det(Q))
    return Q / d ** 0.25 if d > 1e-12 else None

def pack(Q):
    return np.linalg.cholesky(Q)[np.tril_indices(4)]

def d_of(Q, k=46):
    if Q is None: return 0.0
    try:
        B = np.linalg.cholesky(Q + 1e-12*np.eye(4))
        return combigeo.find_optimal(B.tolist(), index=k).normalized
    except Exception:
        return 0.0


def main():
    prev = json.load(open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n2_4d_frontier.json"))
    Q46 = np.array(prev["k46"]["Q"])
    x0 = pack(Q46)
    rng = np.random.default_rng(46)

    best_d, best_Q = prev["k46"]["d"], Q46
    print(f"старт: d(46) = {best_d:.7f}", flush=True)

    starts = [x0] + [x0 * (1 + rng.normal(scale=s, size=10))
                     for s in (0.01, 0.02, 0.04, 0.07) for _ in range(3)]
    for i, s in enumerate(starts):
        hist = {"b": 0.0}
        def obj(x):
            d = d_of(unpack(x))
            if d > hist["b"] + 2e-4:
                hist["b"] = d
                if d > best_d:
                    print(f"  [start{i}] d(46) = {d:.7f}{'  >= 1 !!!' if d >= 1 else ''}", flush=True)
            return -d
        r = minimize(obj, s, method="Nelder-Mead",
                     options={"maxfev": 700, "xatol": 1e-8, "fatol": 1e-12})
        if -r.fun > best_d:
            best_d, best_Q = -r.fun, unpack(r.x)
            print(f"start{i}: улучшение до {best_d:.7f}", flush=True)
            if best_d >= 1.0:
                break

    print(f"\nИТОГ k=46: max d = {best_d:.7f}  {'ПРОБОЙ!' if best_d >= 1 else '< 1'}", flush=True)
    json.dump({"d": best_d, "Q": best_Q.tolist()},
              open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n4_push46.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
