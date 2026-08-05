"""Systematic sweep: max normalized forbidden-interval d(k) for classic lattices.

For each lattice and each number of colors k, find the sublattice of index k
maximizing d = D/diam(V0). Feasible (usable as a space coloring) iff d >= 1.
Exact rational reconstruction of d^2 is attempted (it must be rational for
rational Gram matrices).
"""
import json
import math
import time
from fractions import Fraction

import combigeo
import numpy as np


def main():
    SQ3 = math.sqrt(3.0)

    A4_GRAM = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]], float)
    A4_BASIS = np.linalg.cholesky(A4_GRAM)          # rows = lattice vectors
    A4S_BASIS = np.linalg.inv(A4_BASIS).T           # dual lattice, rows

    LATTICES = {
        "Z2":  ([[1, 0], [0, 1]], range(2, 31)),
        "A2":  ([[1, 0], [0.5, SQ3 / 2]], range(2, 31)),
        "Z3":  ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], range(2, 41)),
        "FCC": ([[1, 1, 0], [1, 0, 1], [0, 1, 1]], range(2, 41)),
        "BCC": ([[2, 0, 0], [0, 2, 0], [1, 1, 1]], range(2, 41)),
        "Z4":  ([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], range(2, 83)),
        "D4":  ([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], range(2, 61)),
        "A4":  (A4_BASIS.tolist(), range(2, 66)),
        "A4s": (A4S_BASIS.tolist(), range(2, 66)),
    }

    out = {}
    for name, (basis, ks) in LATTICES.items():
        cell = combigeo.voronoi_cell(basis)
        print(f"=== {name}: f={cell.f_vector} diam={cell.diameter:.6f} "
              f"diam^2~{Fraction(cell.diameter**2).limit_denominator(100000)}", flush=True)
        rows = []
        t0 = time.time()
        for k in ks:
            res = combigeo.find_optimal(basis, index=k)
            d = res.normalized
            d2 = Fraction(d * d).limit_denominator(100000)
            rows.append({
                "k": k, "d": d, "d2_frac": [d2.numerator, d2.denominator],
                "D": res.best.min_distance, "feasible": bool(d >= 1.0),
                "examined": res.examined,
                "transition": res.best.transition,
            })
            mark = "  <== FEASIBLE" if d >= 1.0 else ""
            print(f"  k={k:3d}  d={d:.6f}  d^2~{str(d2):>9s}  D={res.best.min_distance:.6f}  "
                  f"nsub={res.examined}{mark}", flush=True)
        print(f"  [{name}: {time.time()-t0:.1f}s]", flush=True)
        out[name] = {"f_vector": list(cell.f_vector), "diameter": cell.diameter, "rows": rows}

    with open("/private/tmp/claude-501/-Users-mac-Documents--My-code-Chromatic/a660b7db-31c5-4e8c-a567-e60eed295063/scratchpad/sweep_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
