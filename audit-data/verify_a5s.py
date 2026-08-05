"""Verify Arman-Bondarenko-Prymak-Radchenko: chi(E^5) <= 140 via A5* and explicit C5.

Lattice Lambda = M Z^5 (columns), M = 6x5: rows (-5,1,1,1,1),(1,-5,1,1,1),...,(1,1,1,1,-5),(1,1,1,1,1).
Sublattice Lambda' = M C5 Z^5, |det C5| = 140.
Intrinsic 5x5 row-basis B = chol(M^T M); sublattice rows = C5^T B.
Expect: permutohedron cell f = [720, 1800, 1560, 540, 62]; d = D/diam >= 1.
"""
import math
import time
from fractions import Fraction

import numpy as np
import combigeo


def main():
    n = 5
    M = np.ones((n + 1, n), float)
    for j in range(n):
        M[j, j] = -n
    G = M.T @ M
    B = np.linalg.cholesky(G)          # rows = generators of A5* (scaled by 6)

    C5 = np.array([
        [-2, 1, -2, -1, 0],
        [-3, 1, 0, -1, -2],
        [0, 1, 1, -1, -3],
        [-2, 0, -2, 2, -2],
        [-2, -2, 0, 0, -2],
    ], float)
    k = round(abs(np.linalg.det(C5)))
    print(f"|det C5| = {k} (expect 140)", flush=True)

    t0 = time.time()
    cell = combigeo.voronoi_cell(B.tolist())
    print(f"cell built in {time.time()-t0:.1f}s: f={cell.f_vector} (expect [720,1800,1560,540,62]) "
          f"diam={cell.diameter:.6f}", flush=True)

    sub = C5.T @ B
    det_ratio = abs(np.linalg.det(sub)) / abs(np.linalg.det(B))
    print(f"index check: det(sub)/det(B) = {det_ratio:.3f}", flush=True)

    t0 = time.time()
    D = combigeo.min_color_distance(B.tolist(), sub.tolist())
    d = D / cell.diameter
    f = Fraction(d * d).limit_denominator(200000)
    print(f"D={D:.9f} diam={cell.diameter:.9f} d={d:.9f} d^2~{f} "
          f"feasible={d >= 1.0} [{time.time()-t0:.1f}s]", flush=True)

    lam1 = np.linalg.norm(combigeo.shortest_vector(B.tolist()))
    print(f"lambda1={lam1:.6f}, covering/packing ratio = diam/lambda1 = {cell.diameter/lam1:.6f}",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
