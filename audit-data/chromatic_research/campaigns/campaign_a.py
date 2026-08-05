"""Campaign A: regression vs audit sweep + extended 2D/3D tables.

1) Re-run the audit lattices with the FIXED engine; any d mismatch vs
   audit-data/sweep_results.json is reported loudly.
2) Extend: Z2/A2 to k=40, Z3/FCC/BCC to k=64.
"""
import json, math, time
from fractions import Fraction
import numpy as np
import combigeo
from chromatic_research.paths import results_path


def main():
    AUDIT = json.load(open(results_path("sweep_results.json")))
    SQ3 = math.sqrt(3.0)
    A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
    A4B = np.linalg.cholesky(A4G); A4S = np.linalg.inv(A4B).T

    LATTICES = {
        "Z2":  ([[1,0],[0,1]], 40),
        "A2":  ([[1,0],[0.5,SQ3/2]], 40),
        "Z3":  ([[1,0,0],[0,1,0],[0,0,1]], 64),
        "FCC": ([[1,1,0],[1,0,1],[0,1,1]], 64),
        "BCC": ([[2,0,0],[0,2,0],[1,1,1]], 64),
        "Z4":  (np.eye(4).tolist(), 82),
        "D4":  ([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], 60),
        "A4":  (A4B.tolist(), 65),
        "A4s": (A4S.tolist(), 65),
    }

    out, mismatches = {}, []
    for name, (basis, kmax) in LATTICES.items():
        t0 = time.time()
        res = combigeo.find_optimal_range(basis, 2, kmax)
        rows = []
        for k in sorted(res):
            d = res[k].normalized
            f = Fraction(d*d).limit_denominator(200000)
            rows.append({"k": k, "d": d, "d2": [f.numerator, f.denominator],
                         "D": res[k].best.min_distance, "examined": res[k].examined})
            old = next((r for r in AUDIT.get(name, {}).get("rows", []) if r["k"] == k), None)
            if old is not None and abs(old["d"] - d) > 1e-9:
                mismatches.append((name, k, old["d"], d))
        out[name] = rows
        print(f"{name}: k<=%d done in %.1fs" % (kmax, time.time()-t0), flush=True)

    print("MISMATCHES vs audit:", mismatches if mismatches else "none", flush=True)
    json.dump(out, open(results_path("campaign_a.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
