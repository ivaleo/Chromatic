"""Cyclic-automorphism-invariant coloring search for A_n*.

Idea: restrict phi to have g-INVARIANT kernel, g = (n+1)-cycle automorphism.
Then phi is a left eigenvector of the coord-action A_g = M_g^T over Z/k (eigenvalue
a unit 8th root of 1), and F collapses to <g,+-1>-orbits (check one rep per orbit).
This is the 'ideal sublattice' family that the ABPR C_n constructions live in.
Multiple eigenvectors (possibly different eigenvalues) => product group G."""
import sys, numpy as np, itertools
from sympy import Matrix, factor_list, symbols, Poly, ZZ
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
import combigeo
from symlat import An_star_ambient, gram, perm_matrix
from covrad import covering_radius

def setup(n):
    W = An_star_ambient(n); G = gram(W)
    B = np.linalg.cholesky(G)                       # combigeo basis, same lattice/coords
    R,_ = covering_radius(B, n_dirs=500); diam = 2*R
    Fc = np.array(combigeo.forbidden_coords(B.tolist(), diam, 1.0), dtype=np.int64)
    cyc = list(range(1, n+1)) + [0]
    Mg = perm_matrix(W, cyc)
    Ag = Mg.T                                        # action on coord-vectors c
    return dict(n=n, W=W, B=B, diam=diam, Fc=Fc, Ag=Ag)

def orbit_reps(Fc, Ag, order):
    """canonical reps of F under <A_g, +-1>."""
    seen = set(); reps = []
    Fset = set(map(tuple, Fc.tolist()))
    for f in Fc:
        key = tuple(int(x) for x in f)
        if key in seen: continue
        orb = []
        c = f.copy()
        for _ in range(order):
            orb.append(tuple(int(x) for x in c))
            orb.append(tuple(int(-x) for x in c))
            c = Ag @ c
        canon = min(orb)
        if canon in seen: continue
        for o in orb: seen.add(o)
        reps.append(np.array(f, dtype=np.int64))
    return reps

if __name__ == "__main__":
    x = symbols('x')
    for n in [5, 7]:
        S = setup(n); Fc, Ag = S['Fc'], S['Ag']
        reps = orbit_reps(Fc, Ag, n+1)
        # rational spectrum of A_g (which cyclotomic factors -> which k admit eigenvectors)
        cp = Matrix(Ag.tolist()).charpoly(x)
        facs = factor_list(cp.as_expr())
        fac_str = " * ".join(f"({f})^{m}" if m>1 else f"({f})" for f,m in facs[1])
        print(f"n={n}: |F|={len(Fc)}  ->  <g,+-1>-orbits = {len(reps)}  (reduction {len(Fc)/len(reps):.0f}x)")
        print(f"   charpoly(A_g) = {fac_str}")
