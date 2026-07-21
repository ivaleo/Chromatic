"""Verification batch 2.

Part A: Ivanov's R^3 theorem rows (deformed BCC lattices) and sparseness tables —
        check each printed (lattice, sublattice, bound) triple, and where the printed
        sublattice is corrupt, find the true optimum by full search.
Part B: Ivanov's special 6-coloring lattice in R^2 (ratio 0.99144).
Part C: All 16 representative 4-dim lattices from Vallentin-Weissbach-Zimmermann 2024
        (Gram = sums of Voronoi rays R1..R12), scan k=2..54 for any d >= 1 below 49.
"""
import json
import math
import time
from fractions import Fraction

import numpy as np
import combigeo

OUT = {}

def frac(x):
    f = Fraction(x).limit_denominator(200000)
    return f"{f.numerator}/{f.denominator}"

def diag_ab(a, b):
    return [[a, b, b], [b, a, b], [b, b, a]]

def check_pair(name, lat, C, printed_bound, run_full=None):
    """Verify d for sublattice given by integer matrix C (rows, in lattice coords)."""
    lat = np.array(lat, float)
    C = np.array(C, float)
    sub = C @ lat
    k = round(abs(np.linalg.det(C)))
    cell = combigeo.voronoi_cell(lat.tolist())
    D = combigeo.min_color_distance(lat.tolist(), sub.tolist())
    d = D / cell.diameter
    line = (f"{name}: k={k} printed_bound={printed_bound} computed d={d:.6f} "
            f"(d^2~{frac(d*d)}) D={D:.6f} diam={cell.diameter:.6f} "
            f"{'OK >= printed' if d >= printed_bound - 5e-4 else '!!! BELOW printed'}")
    print(line, flush=True)
    rec = {"k": k, "printed": printed_bound, "d": d, "D": D, "diam": cell.diameter}
    if run_full:
        res = combigeo.find_optimal(lat.tolist(), index=run_full)
        rec["opt_d"] = res.normalized
        rec["opt_transition"] = res.best.transition
        print(f"    full search k={run_full}: optimal d={res.normalized:.6f} "
              f"(d^2~{frac(res.normalized**2)}) transition={res.best.transition}", flush=True)
    OUT[name] = rec

print("=== Part A: Ivanov R^3 Theorem 3 rows ===", flush=True)
# row 1: k=18, lattice diag(0.849009; -1), Gamma rows (1 -2 1; -1 -1 2; 2 3 1), bound 1.115
check_pair("R3-k18", diag_ab(0.849009, -1), [[1, -2, 1], [-1, -1, 2], [2, 3, 1]], 1.115,
           run_full=18)
# row 2: k=21, lattice diag(1; -0.246656); printed Gamma corrupt (det 12) -> full search only
lat21 = diag_ab(1, -0.246656)
res21 = combigeo.find_optimal(lat21, index=21)
print(f"R3-k21 (printed matrix corrupt, det 12): full search optimal d={res21.normalized:.6f} "
      f"(printed bound 1.133) transition={res21.best.transition}", flush=True)
OUT["R3-k21"] = {"printed": 1.133, "opt_d": res21.normalized,
                 "opt_transition": res21.best.transition}
# also what does the printed (wrong-det) matrix give?
check_pair("R3-k21-printed-matrix", lat21, [[0, -1, 1], [0, 1, 3], [3, 0, 1]], 0.0)
# row 3: k=23, lattice diag(1; -0.750981), Gamma (0 -2 1; -3 -2 -4; 2 -1 0), bound 1.137
check_pair("R3-k23", diag_ab(1, -0.750981), [[0, -2, 1], [-3, -2, -4], [2, -1, 0]], 1.137,
           run_full=23)
# row 4: k=24, BCC diag(1; -1), Gamma (-2 1 1; 2 3 3; 2 3 0), bound 1.303
check_pair("R3-k24", diag_ab(1, -1), [[-2, 1, 1], [2, 3, 3], [2, 3, 0]], 1.303)
# row 5: k=27, BCC, 3I, bound 1.549
check_pair("R3-k27", diag_ab(1, -1), [[3, 0, 0], [0, 3, 0], [0, 0, 3]], 1.549)

print("=== Part A2: Ivanov sparseness tables (d < 1) ===", flush=True)
SQ3 = math.sqrt(3.0)
HEX2 = [[2, 0], [1, SQ3]]
check_pair("R2-sparse-k3", HEX2, [[2, -1], [1, 1]], 0.5)
check_pair("R2-sparse-k4", HEX2, [[2, 0], [0, 2]], 0.866)
LAT6 = [[2 + math.sqrt(2), -math.sqrt(2)], [1, SQ3]]
check_pair("R2-sparse-k6", LAT6, [[2, 1], [0, 3]], 0.991, run_full=6)
# R^3 sparseness rows
check_pair("R3-sparse-k4", diag_ab(1, -1), [[1, 0, 3], [0, 1, 3], [0, 0, 4]], 0.316)
check_pair("R3-sparse-k5", diag_ab(1, -0.816322), [[1, 0, 4], [0, 1, 4], [0, 0, 5]], 0.404)
check_pair("R3-sparse-k6", diag_ab(1, -0.718788), [[1, 0, 2], [0, 1, 2], [0, 0, 6]], 0.447)
check_pair("R3-sparse-k7", diag_ab(1, -0.366288), [[1, 0, 6], [0, 1, 6], [0, 0, 7]], 0.468)
check_pair("R3-sparse-k8", diag_ab(1, -1), [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 0.774)
check_pair("R3-sparse-k12", diag_ab(1, -0.703622), [[1, 1, 3], [0, 2, 0], [0, 0, 6]], 0.844)
check_pair("R3-sparse-k14", diag_ab(1, -0.147065), [[1, 0, 3], [0, 1, 7], [0, 0, 14]], 0.876)
# Ivanov aux: Z^2 8-coloring and Z^3 26-coloring
check_pair("Z2-8colors", [[1, 0], [0, 1]], [[3, -1], [2, 2]], 1.0)
check_pair("Z3-26colors", [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
           [[3, -1, 0], [2, 2, -2], [1, 0, 3]], 1.0)

print("=== Part C: 16 representative 4-dim lattices, k=2..54 ===", flush=True)
R = {
 1: [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
 2: [[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],
 3: [[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]],
 4: [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],
 5: [[1,-1,0,0],[-1,1,0,0],[0,0,0,0],[0,0,0,0]],
 6: [[1,0,-1,0],[0,0,0,0],[-1,0,1,0],[0,0,0,0]],
 7: [[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]],
 8: [[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]],
 9: [[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]],
 10:[[0,0,0,0],[0,0,0,0],[0,0,1,-1],[0,0,-1,1]],
 11:[[4,2,-2,-2],[2,4,-2,-2],[-2,-2,4,0],[-2,-2,0,4]],
 12:[[1,1,-1,-1],[1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,1]],
}
REPS = {
 "K5(=A4*)":      [1,2,3,4,5,6,7,8,9,10],
 "K3,3":          [1,2,3,4,6,7,8,9,12],
 "111-":          [1,2,3,4,6,7,8,9,10,11],
 "K5-1":          [1,2,3,4,5,7,8,9,10],
 "211+":          [1,2,3,4,6,8,9,11,12],
 "K5-1-1":        [1,2,3,4,5,7,8,10],
 "221+":          [1,2,3,4,8,9,11,12],
 "K5-2-1":        [1,2,4,5,7,8,10],
 "222+":          [1,3,4,8,9,11,12],
 "C2221":         [1,2,3,4,7,9,10],
 "K5-3(cls A4)":  [1,2,4,7,8,9,10],
 "K4+1":          [1,2,3,4,8,9,10],
 "C221+1":        [1,2,3,4,8,10],
 "C3+C3":         [1,4,7,8,9,10],
 "C3+1+1":        [1,2,3,4,8],
 "1+1+1+1(=Z4)":  [1,2,3,4],
}
scan = {}
for name, rays in REPS.items():
    Q = np.zeros((4, 4))
    for i in rays:
        Q += np.array(R[i], float)
    B = np.linalg.cholesky(Q)  # rows = basis vectors
    cell = combigeo.voronoi_cell(B.tolist())
    t0 = time.time()
    rows = []
    first_feasible = None
    for k in range(2, 55):
        res = combigeo.find_optimal(B.tolist(), index=k)
        d = res.normalized
        rows.append({"k": k, "d": d})
        if d >= 1.0 and first_feasible is None:
            first_feasible = k
            print(f"  {name}: FIRST FEASIBLE k={k} d={d:.6f} (d^2~{frac(d*d)})", flush=True)
    best = max(rows, key=lambda r: r["d"])
    print(f"{name}: |Vor|={cell.f_vector[-1]} f={cell.f_vector} "
          f"first_feasible={first_feasible} best_d={best['d']:.6f}@k={best['k']} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    scan[name] = {"f_vector": list(cell.f_vector), "first_feasible": first_feasible,
                  "rows": rows}
OUT["scan4d"] = scan

with open("/private/tmp/claude-501/-Users-mac-Documents--My-code-Chromatic/a660b7db-31c5-4e8c-a567-e60eed295063/scratchpad/verify2_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
print("DONE", flush=True)
