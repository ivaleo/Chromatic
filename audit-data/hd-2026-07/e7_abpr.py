"""Reconstruct ABPR's exact E_7* construction (arXiv:2112.13438v2, Sec 3) and
validate in our framework, then search for a sublattice of index < 1372.

Lambda = M Z^7 in R^8 (M = 8x7).  Sublattice Lambda' = M C_7 Z^7, index |det C_7|.
In M-basis coords (a point M c has coords c), the forbidden set F = {v!=0: D(v)<2R}
matches ABPR's F exactly (D(v)=dist(V,v+V)). A sublattice with integer basis matrix
C (columns) is VALID iff no f in F lies in C Z^7, i.e. C^{-1} f not in Z^7 for all f."""
import numpy as np, sys
from fractions import Fraction
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
import combigeo
from covrad import covering_radius

def M_E7():
    q = 0.25
    M = np.array([
        [-1, 0, 0, 0, 0, 0, -3*q],
        [ 1,-1, 0, 0, 0, 0, -3*q],
        [ 0, 1,-1, 0, 0, 0,    q],
        [ 0, 0, 1,-1, 0, 0,    q],
        [ 0, 0, 0, 1,-1, 0,    q],
        [ 0, 0, 0, 0, 1,-1,    q],
        [ 0, 0, 0, 0, 0, 1,    q],
        [ 0, 0, 0, 0, 0, 0,    q],
    ], dtype=float)          # 8 x 7 ; columns = basis vectors
    return M

C7 = np.array([
    [ 0,-4, -5,-3,-4,-4,-1],
    [-1,-5,-10,-7,-5,-5,-4],
    [-2,-2, -9,-4,-5,-4,-4],
    [-3,-2, -5,-4,-4,-1,-3],
    [-1,-1, -4,-1,-3, 0,-3],
    [-2, 0, -1, 0, 0, 0, 0],
    [ 0, 4,  6, 4, 4, 4, 4],
], dtype=np.int64)

if __name__ == "__main__":
    M = M_E7()
    G = M.T @ M                                   # 7x7 Gram
    B = np.linalg.cholesky(G)                     # combigeo basis (same lattice/coords as M)
    lam1 = np.linalg.norm(combigeo.shortest_vector(B.tolist()))
    R,_ = covering_radius(B, n_dirs=600); diam = 2*R
    print(f"E7* built: lam1={lam1:.4f}  R(cover)={R:.4f}  diam=2R={diam:.4f}  cov/pack ratio={diam/lam1:.4f}")
    print(f"|det C7| = {round(abs(np.linalg.det(C7)))} (expect 1372)")

    F = np.array(combigeo.forbidden_coords(B.tolist(), diam, 1.0), dtype=np.int64)
    print(f"|F_1| = {len(F)}  (forbidden lattice vectors, M-basis integer coords)")

    # validity of ABPR sublattice: C7 columns are the sublattice basis in coords.
    # f in Lambda' iff C7^{-1} f in Z^7. Valid iff NO f in F satisfies that.
    C7inv = np.linalg.inv(C7.astype(float))
    Y = C7inv @ F.T                               # 7 x |F|
    is_int = np.all(np.abs(Y - np.round(Y)) < 1e-6, axis=0)
    nviol = int(is_int.sum())
    print(f"ABPR C7 (index 1372): forbidden vectors inside Lambda' = {nviol}  -> "
          f"{'VALID (avoids F)' if nviol==0 else 'INVALID'}")
