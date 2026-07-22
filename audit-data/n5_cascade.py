"""N5: каскад вниз — k=45, 44, 43 (и контрольный ретрай 47) от победителя-46."""
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

def d_of(Q, k):
    if Q is None: return 0.0
    try:
        B = np.linalg.cholesky(Q + 1e-12*np.eye(4))
        return combigeo.find_optimal(B.tolist(), index=k).normalized
    except Exception:
        return 0.0

W46 = np.array(json.load(open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n4_push46.json"))["Q"])
x46 = pack(W46)
rng = np.random.default_rng(45)
out = {}

def push(k, starts, maxfev):
    best_d, best_Q = 0.0, None
    for i, s in enumerate(starts):
        hist = {"b": 0.0}
        def obj(x):
            d = d_of(unpack(x), k)
            if d > hist["b"] + 3e-4:
                hist["b"] = d
                print(f"  [k={k} s{i}] d={d:.7f}{'  ПРОБОЙ' if d >= 1 else ''}", flush=True)
            return -d
        r = minimize(obj, s, method="Nelder-Mead",
                     options={"maxfev": maxfev, "xatol": 1e-8, "fatol": 1e-12})
        if -r.fun > best_d:
            best_d, best_Q = -r.fun, unpack(r.x)
        if best_d >= 1.0 and i >= 1:
            break
    print(f"k={k}: max d = {best_d:.7f}  {'>=1 ПРОБОЙ' if best_d >= 1 else '< 1'}", flush=True)
    out[f"k{k}"] = {"d": best_d, "Q": None if best_Q is None else best_Q.tolist()}
    return best_d, best_Q

# каскад: 45 от победителя-46, потом ниже от каждого нового победителя
prev_x = x46
for k in (45, 44, 43):
    starts = [prev_x] + [prev_x * (1 + rng.normal(scale=s, size=10))
                         for s in (0.01, 0.03, 0.06, 0.1) for _ in range(2)]
    d, Qw = push(k, starts, 650)
    if d >= 1.0 and Qw is not None:
        prev_x = pack(Qw)
    else:
        break

# ретрай 47 от победителя-46 (правдоподобно, что и 47 пробивается из этой зоны)
push(47, [x46] + [x46 * (1 + rng.normal(scale=s, size=10)) for s in (0.02, 0.05)],
     500)

json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n5_cascade.json", "w"),
          indent=1)
print("DONE", flush=True)
