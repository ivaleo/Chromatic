"""N2: форм-оптимальная граница в R^4.

(a) усиленные мультистарты k=47 и k=46 (пробить ниже 48?);
(b) максимальная ширина d(k) по всем формам для k = 49..56 (новая лестница).
"""
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

def norm_gram(M):
    G = M @ M.T
    return G / abs(np.linalg.det(G)) ** 0.25

D4 = norm_gram(np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float))
A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
A4S = norm_gram(np.linalg.inv(np.linalg.cholesky(A4G)).T)
W48 = np.array(json.load(open("/Users/mac/Documents/_My_code/Chromatic/audit-data/r5_push48.json"))["k48"]["Q"])
W48 = W48 / abs(np.linalg.det(W48)) ** 0.25

rng = np.random.default_rng(11)
out = {}

def campaign(k, starts, maxfev, tag):
    best_d, best_Q = 0.0, None
    for i, s in enumerate(starts):
        hist = {"b": 0.0}
        def obj(x):
            d = d_of(unpack(x), k)
            if d > hist["b"] + 5e-4:
                hist["b"] = d
                print(f"  [{tag} start{i}] d({k})={d:.6f}", flush=True)
            return -d
        r = minimize(obj, s, method="Nelder-Mead",
                     options={"maxfev": maxfev, "xatol": 1e-6, "fatol": 1e-10})
        if -r.fun > best_d:
            best_d, best_Q = -r.fun, unpack(r.x)
    print(f"{tag}: k={k} max d = {best_d:.7f}", flush=True)
    return {"d": best_d, "Q": None if best_Q is None else best_Q.tolist()}

# (a) k=47, 46 — жёсткие мультистарты
for k in (47, 46):
    starts = [pack(W48)] + [pack(W48) * (1 + rng.normal(scale=s, size=10))
                            for s in (0.04, 0.08, 0.12, 0.2) for _ in range(2)] + \
             [pack(D4), pack(A4S)]
    out[f"k{k}"] = campaign(k, starts, 320, f"break{k}")

# (b) ширины k=49..56
for k in range(49, 57):
    starts = [pack(D4), pack(A4S), pack(W48),
              pack(W48) * (1 + rng.normal(scale=0.1, size=10))]
    out[f"width{k}"] = campaign(k, starts, 240, f"width{k}")

json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n2_4d_frontier.json", "w"),
          indent=1)
print("DONE", flush=True)
