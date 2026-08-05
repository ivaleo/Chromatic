"""Exhaustive search over ALL sublattices of index 140 in A5* (independent check of
Arman et al. Prop 12 optimality and of our witness d = sqrt(39/35))."""
import math, time
import numpy as np
import combigeo
from fractions import Fraction


def main():
    n = 5
    M = np.ones((n + 1, n), float)
    for j in range(n):
        M[j, j] = -n
    B = np.linalg.cholesky(M.T @ M)
    total = combigeo.count_sublattices(5, 140)
    print(f"sublattices of index 140 in dim 5: {total}", flush=True)
    t0 = time.time()
    res = combigeo.find_optimal(B.tolist(), index=140)
    d = res.normalized
    print(f"A5* k=140 exhaustive: d={d:.9f} d^2~{Fraction(d*d).limit_denominator(200000)} "
          f"examined={res.examined} time={time.time()-t0:.0f}s", flush=True)
    print("transition:", res.best.transition, flush=True)
    print("sub_basis:", np.round(np.array(res.best.sub_basis), 6).tolist(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
