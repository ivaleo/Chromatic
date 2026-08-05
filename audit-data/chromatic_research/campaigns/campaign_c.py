"""Campaign C: extended 4D staircases to k = 100 (D4, A4*, K3,3-rep, 111--rep)."""
import json, time
from fractions import Fraction
import numpy as np
import combigeo
from chromatic_research.paths import results_path

R = {1:[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],2:[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],
     3:[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]],4:[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],
     6:[[1,0,-1,0],[0,0,0,0],[-1,0,1,0],[0,0,0,0]],7:[[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]],
     8:[[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]],9:[[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]],
     10:[[0,0,0,0],[0,0,0,0],[0,0,1,-1],[0,0,-1,1]],
     11:[[4,2,-2,-2],[2,4,-2,-2],[-2,-2,4,0],[-2,-2,0,4]],
     12:[[1,1,-1,-1],[1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,1]]}

def gram(rays):
    Q = np.zeros((4, 4))
    for i in rays:
        Q += np.array(R[i], float)
    return np.linalg.cholesky(Q).tolist()


def main():
    A4G = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], float)
    A4S = np.linalg.inv(np.linalg.cholesky(A4G)).T.tolist()

    LATTICES = {
        "D4":   [[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],
        "A4s":  A4S,
        "K3,3": gram([1,2,3,4,6,7,8,9,12]),
        "111-": gram([1,2,3,4,6,7,8,9,10,11]),
    }

    out = {}
    for name, basis in LATTICES.items():
        t0 = time.time()
        res = combigeo.find_optimal_range(basis, 2, 100)
        rows = []
        for k in sorted(res):
            d = res[k].normalized
            f = Fraction(d*d).limit_denominator(500000)
            rows.append({"k": k, "d": d, "d2": [f.numerator, f.denominator]})
        feas = [r["k"] for r in rows if r["d"] >= 1.0 - 1e-12]
        print(f"{name}: feasible k = {feas}  [{time.time()-t0:.0f}s]", flush=True)
        out[name] = rows
    json.dump(out, open(results_path("campaign_c.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
