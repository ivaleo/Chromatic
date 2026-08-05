"""Campaign B: Ivanov's deformed-BCC families in R^3 — search for NEW interval bounds.

Two 1-parameter families of lattices (rows):
  F1(b) = [[1,b,b],[b,1,b],[b,b,1]],  b in (-1, 0)   (b=-1 -> BCC)
  F2(a) = [[a,-1,-1],[-1,a,-1],[-1,-1,a]], a in (0.55, 1.45)  (a=1 -> BCC)
For each k = 13..32: max over the family grids of the optimal d(k);
compare with pure BCC/FCC and Ivanov's published rows (18: 1.115838,
21: 1.133698, 23: 1.137320).
"""
import json, time
from fractions import Fraction
import numpy as np
import combigeo

def diag_ab(a, b):
    return [[a, b, b], [b, a, b], [b, b, a]]


def main():
    KS = range(13, 33)
    best = {k: (0.0, None) for k in KS}

    t0 = time.time()
    grids = [("F1", b) for b in np.arange(-0.995, -0.0049, 0.005)] + \
            [("F2", a) for a in np.arange(0.55, 1.4501, 0.005)]
    for fam, p in grids:
        basis = diag_ab(1.0, float(p)) if fam == "F1" else diag_ab(float(p), -1.0)
        try:
            res = combigeo.find_optimal_range(basis, 13, 32)
        except Exception as e:
            print(f"skip {fam} {p:.3f}: {e}", flush=True)
            continue
        for k in KS:
            d = res[k].normalized
            if d > best[k][0]:
                best[k] = (d, (fam, round(float(p), 4)))
    print(f"[grid of {len(grids)} lattices done in {time.time()-t0:.0f}s]", flush=True)

    # local refinement around each k-winner (step 0.0005)
    for k in KS:
        d0, tag = best[k]
        if tag is None:
            continue
        fam, p0 = tag
        for p in np.arange(p0 - 0.006, p0 + 0.0061, 0.0005):
            basis = diag_ab(1.0, float(p)) if fam == "F1" else diag_ab(float(p), -1.0)
            try:
                r = combigeo.find_optimal(basis, index=k)
            except Exception:
                continue
            if r.normalized > best[k][0]:
                best[k] = (r.normalized, (fam, round(float(p), 5)))

    IVANOV = {18: 1.115838, 21: 1.133698, 23: 1.137320, 24: 1.303840, 27: 1.549193}
    out = {}
    for k in KS:
        d, tag = best[k]
        ref = IVANOV.get(k)
        mark = ""
        if ref is not None:
            mark = "= Ivanov" if abs(d - ref) < 5e-4 else ("> Ivanov!" if d > ref else "< Ivanov")
        print(f"k={k:2d}  best_d={d:.6f}  family={tag}  {mark}", flush=True)
        out[k] = {"d": d, "family": tag, "ivanov": ref}
    json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/campaign_b.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
