"""Beyond ABPR: color the CONFLICT GRAPH on Lambda/Lambda' instead of giving each
coset a distinct color. Vertices = cosets Lambda/Lambda'; edge [a]~[b] iff (a-b)
mod Lambda' is hit by the forbidden set F (i.e. cells a,b can realize distance 2R).
ABPR use |Lambda/Lambda'| colors; chi(H) can be strictly smaller. First check:
does F mod Lambda' cover ALL nonzero cosets? If not, cosets can be merged."""
import numpy as np, sys
import combigeo
from chromatic_research.core.covrad import covering_radius
from chromatic_research.paths import results_path

def M_Anstar(n):
    M = np.ones((n+1, n))
    for i in range(n): M[i, i] = -n
    return M

C5 = np.array([
    [-2, 1,-2,-1, 0],
    [-3, 1, 0,-1,-2],
    [ 0, 1, 1,-1,-3],
    [-2, 0,-2, 2,-2],
    [-2,-2, 0, 0,-2]], dtype=np.int64)

def adjugate_int(C):
    det = int(round(np.linalg.det(C.astype(float))))
    Cinv = np.linalg.inv(C.astype(float))
    A = np.round(Cinv * det).astype(np.int64)
    return A, det

def coset_keys(F, C):
    """exact coset key of each f in F: A f mod det (A=adjugate(C)). 0-vector -> excluded."""
    A, det = adjugate_int(C)
    K = (F @ A.T) % det                       # |F| x n, each row a coset key in (Z/det)^n
    keys = set()
    for row in K:
        t = tuple(int(x) for x in row)
        if any(t): keys.add(t)                # skip the zero coset (shouldn't appear: F avoids Lambda')
    return keys, det

def analyze(name, M, C):
    n = M.shape[1]; G = M.T @ M
    B = np.linalg.cholesky(G)
    R,_ = covering_radius(B, n_dirs=600); diam = 2*R
    F = np.array(combigeo.forbidden_coords(B.tolist(), diam, 1.0), dtype=np.int64)
    det = int(round(abs(np.linalg.det(C.astype(float)))))
    # validity: C avoids F
    A, _ = adjugate_int(C)
    inLp = np.all((F @ A.T) % det == 0, axis=1)
    S, det2 = coset_keys(F, C)
    print(f"{name}: n={n} index={det} |F|={len(F)} valid(F∩Λ'=∅)={not inLp.any()}")
    print(f"   distinct nonzero cosets hit by F: |S̄| = {len(S)}  of {det-1} nonzero cosets")
    print(f"   -> {'ALL covered: conflict graph complete, no merging' if len(S)>=det-1 else 'MISSED %d cosets -> mergeable, chi(H) may be < %d'%(det-1-len(S), det)}")
    return F, C, S, det

if __name__ == "__main__":
    from chromatic_research.core import e7_abpr
    print("=== A_5* / 140 (validate) ===")
    analyze("A5*", M_Anstar(5), C5)
    print("=== E_7* / 1372 ===")
    analyze("E7*", e7_abpr.M_E7(), e7_abpr.C7)
