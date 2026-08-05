"""R1: непрерывная оптимизация решётки в R^3 (2-параметрическое семейство
rows (a,b,b),(b,a,b),(b,b,a)) — уточнение рекордов campaign_b по Нелдеру—Миду."""
import json
import numpy as np
from scipy.optimize import minimize
import combigeo
from chromatic_research.paths import results_path

def basis(a, b):
    return [[a, b, b], [b, a, b], [b, b, a]]

def d_of(k, a, b):
    try:
        M = np.array(basis(a, b), float)
        if abs(np.linalg.det(M)) < 1e-6:
            return 0.0
        return combigeo.find_optimal(basis(a, b), index=k).normalized
    except Exception:
        return 0.0


def main():
    prev = json.load(open(results_path("campaign_b.json")))
    out = {}
    for k in range(16, 33):
        fam, p = prev[str(k)]["family"]
        starts = [(1.0, p) if fam == "F1" else (p, -1.0), (1.0, -1.0)]
        best_d, best_ab = prev[str(k)]["d"], starts[0]
        for s in starts:
            r = minimize(lambda x: -d_of(k, x[0], x[1]), np.array(s, float),
                         method="Nelder-Mead",
                         options={"maxfev": 400, "xatol": 1e-6, "fatol": 1e-9})
            if -r.fun > best_d:
                best_d, best_ab = -r.fun, (float(r.x[0]), float(r.x[1]))
        grid_d = prev[str(k)]["d"]
        gain = best_d - grid_d
        print(f"k={k:2d}  d={best_d:.7f}  (a,b)=({best_ab[0]:.5f},{best_ab[1]:.5f})  "
              f"{'+%.5f vs сетка' % gain if gain > 1e-6 else '= сетка'}", flush=True)
        out[k] = {"d": best_d, "ab": best_ab, "grid_d": grid_d}
    json.dump(out, open(results_path("r1_refined.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
