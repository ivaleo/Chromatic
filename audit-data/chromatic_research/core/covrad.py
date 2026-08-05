"""Vertex-free covering radius R_cov (=> diam=2*R_cov) via random-direction LPs
over the Voronoi cell's supporting halfspaces (relevant_facets). Any dimension.

R_cov = max_{x in V0} ||x||. Each random objective u gives a cell VERTEX
argmax u.x (an LP over facet inequalities normal.x <= offset). Over many
directions we recover the farthest vertex = deepest hole = R_cov.
Converges from BELOW; use many directions + local polish for accuracy."""
import numpy as np
from scipy.optimize import linprog
import combigeo
from chromatic_research.paths import results_path

def covering_radius(B, n_dirs=4000, seed=0, polish=True):
    facets = combigeo.relevant_facets(B.tolist())   # list of (lattice_vector v, offset=|v|/2)
    n = len(B)
    A = np.array([f[0] for f in facets], float)      # rows: lattice vectors v (normal dir)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    A_unit = A / nrm                                  # unit normals
    b = np.array([f[1] for f in facets], float)       # offsets |v|/2 (already for unit normal)
    rng = np.random.default_rng(seed)
    best = 0.0; best_x = None
    # seed directions: facet normals themselves + random
    seeds = list(A_unit) + [rng.standard_normal(n) for _ in range(n_dirs)]
    for u in seeds:
        u = u / (np.linalg.norm(u) + 1e-15)
        # maximize u.x  <=> minimize -u.x  s.t. A_unit x <= b
        res = linprog(-u, A_ub=A_unit, b_ub=b, bounds=[(None, None)]*n, method='highs')
        if res.success:
            x = res.x; r = np.linalg.norm(x)
            if r > best: best = r; best_x = x
    return best, best_x

if __name__ == "__main__":
    import sys, math
    from chromatic_research.core.lattices import CATALOG
    # known-exact references (diam/lam1): validate the estimator
    KNOWN = {'A5*': math.sqrt(7/3), 'D5': math.sqrt(5/2), 'D6': math.sqrt(3),
             'E6*': math.sqrt(2), 'E8': math.sqrt(2), 'D5*': 1.5, 'D6*': math.sqrt(3),
             'E6': 1.6330, 'A6*': math.sqrt(8/3), 'D7': math.sqrt(3.5), 'D8': 2.0,
             'D9': math.sqrt(4.5), 'A7*': math.sqrt(3), 'A8*': math.sqrt(10/3),
             'A9*': math.sqrt(11/3), 'E7': math.sqrt(3)}
    names = sys.argv[1:] if len(sys.argv) > 1 else list(KNOWN)
    import time
    for name in names:
        B = CATALOG[name](); n = len(B)
        lam1 = float(np.linalg.norm(combigeo.shortest_vector(B.tolist())))
        t = time.time()
        R, _ = covering_radius(B, n_dirs=3000, seed=1)
        diam = 2*R; ratio = diam/lam1
        k = KNOWN.get(name)
        tag = f"known={k:.5f} err={abs(ratio-k):.5f}" if k else ""
        print(f"{name:5s} n={n} lam1={lam1:.4f} diam={diam:.5f} diam/lam1={ratio:.5f} {tag} [{time.time()-t:.0f}s]", flush=True)
