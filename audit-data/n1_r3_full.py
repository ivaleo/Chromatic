"""N1: полная оптимизация форм в R^3 (6-параметрическое пространство Холецкого,
не только диагональное семейство) — Нелдер-Мид per k=14..32."""
import json
import numpy as np
from scipy.optimize import minimize
import combigeo

def unpack(x):
    L = np.zeros((3, 3))
    L[np.tril_indices(3)] = x
    Q = L @ L.T
    d = abs(np.linalg.det(Q))
    return Q / d ** (1/3) if d > 1e-10 else None

def pack(Q):
    return np.linalg.cholesky(Q)[np.tril_indices(3)]

def d_of(Q, k):
    if Q is None: return 0.0
    try:
        B = np.linalg.cholesky(Q + 1e-12*np.eye(3))
        return combigeo.find_optimal(B.tolist(), index=k).normalized
    except Exception:
        return 0.0

def diag_ab(a, b):
    return np.array([[a,b,b],[b,a,b],[b,b,a]], float) @ np.array([[a,b,b],[b,a,b],[b,b,a]], float).T


def main():
    prev = json.load(open("/Users/mac/Documents/_My_code/Chromatic/audit-data/r1_refined.json"))
    BCC = np.array([[2,0,0],[0,2,0],[1,1,1]], float)
    rng = np.random.default_rng(5)
    out = {}
    for k in range(14, 33):
        starts = [pack(BCC @ BCC.T / abs(np.linalg.det(BCC @ BCC.T))**(1/3))]
        if str(k) in prev:
            a, b = prev[str(k)]["ab"]
            M = np.array([[a,b,b],[b,a,b],[b,b,a]], float)
            G = M @ M.T
            starts.insert(0, pack(G / abs(np.linalg.det(G))**(1/3)))
        starts += [starts[0] * (1 + rng.normal(scale=0.15, size=6)) for _ in range(3)]
        best_d, best_Q = prev.get(str(k), {}).get("d", 0.0), None
        for s in starts:
            r = minimize(lambda x: -d_of(unpack(x), k), s, method="Nelder-Mead",
                         options={"maxfev": 500, "xatol": 1e-7, "fatol": 1e-10})
            if -r.fun > best_d:
                best_d, best_Q = -r.fun, unpack(r.x)
        base = prev.get(str(k), {}).get("d", 0.0)
        mark = f"+{best_d-base:.5f} vs семейство" if best_d > base + 1e-5 else "= семейство"
        print(f"k={k:2d}  d={best_d:.7f}  {mark}", flush=True)
        out[k] = {"d": best_d, "Q": None if best_Q is None else best_Q.tolist(), "family_d": base}
    json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n1_r3_full.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
