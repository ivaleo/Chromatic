"""Search for an E_7* sublattice of index < 1372 avoiding the forbidden set F.
Works directly on the sublattice basis (fast algebraic validity check), bypassing
min-conflicts. Methods: (1) gradient descent from ABPR's C_7 -- replace a
generator by a shorter non-forbidden vector reducing |det| while staying valid;
(2) randomized short-vector bases. Validity: no f in F lies in C Z^7."""
import numpy as np, sys, itertools, time
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
import combigeo
from covrad import covering_radius
from e7_abpr import M_E7, C7

def build():
    M = M_E7(); G = M.T @ M
    B = np.linalg.cholesky(G)
    R,_ = covering_radius(B, n_dirs=600); diam = 2*R
    F = np.array(combigeo.forbidden_coords(B.tolist(), diam, 1.0), dtype=np.int64)
    return M, G, B, diam, F

def short_pool(B, F, radius):
    """short lattice vectors (integer M-coords) NOT in F, sorted by norm."""
    n = B.shape[0]
    vs = combigeo._vectors_near(B.tolist(), [0.0]*n, radius)
    Binv = np.linalg.inv(np.array(B))
    Fset = set(map(tuple, F.tolist()))
    G = np.array(B) @ np.array(B).T
    out = []
    for v in vs:
        c = np.round(np.array(v) @ Binv).astype(np.int64)
        if not c.any(): continue
        t = tuple(int(x) for x in c); nt = tuple(-int(x) for x in c)
        if t in Fset or nt in Fset: continue      # F stores one of +-v: check BOTH
        nrm = float(c @ G @ c)
        out.append((nrm, c))
    out.sort(key=lambda z: z[0])
    # dedup +-
    seen=set(); pool=[]
    for nrm,c in out:
        key=tuple(int(x) for x in c); nkey=tuple(-int(x) for x in c)
        if key in seen or nkey in seen: continue
        seen.add(key); pool.append(c)
    return np.array(pool)

class Validator:
    def __init__(self, F):
        self.F = F.astype(float).T          # 7 x |F|
    def valid(self, C):
        """no f in F inside C Z^7 (C: 7x7 int columns=generators)."""
        try: Ci = np.linalg.inv(C.astype(float))
        except np.linalg.LinAlgError: return False
        Y = Ci @ self.F
        return not np.any(np.all(np.abs(Y - np.round(Y)) < 1e-6, axis=0))

def index_of(C):
    return int(round(abs(np.linalg.det(C.astype(float)))))

def gradient_descent(C0, pool, val, max_sweeps=40):
    C = C0.copy(); best = index_of(C)
    for sweep in range(max_sweeps):
        improved = False
        for j in range(C.shape[1]):
            for g in pool:                        # pool sorted short->long
                Cn = C.copy(); Cn[:, j] = g
                d = index_of(Cn)
                if 0 < d < best and val.valid(Cn):
                    C = Cn; best = d; improved = True
                    break
        if not improved: break
    return C, best

if __name__ == "__main__":
    radius = float(sys.argv[1]) if len(sys.argv)>1 else 2.6
    M,G,B,diam,F = build()
    val = Validator(F)
    pool = short_pool(B, F, radius)
    print(f"E7*: |F|={len(F)} pool(non-forbidden short, |v|<{radius})={len(pool)}  "
          f"shortest-nonforbidden norm={float(pool[0] @ G @ pool[0]):.3f}", flush=True)
    assert val.valid(C7), "ABPR C7 must be valid"
    print(f"start: ABPR C7 index={index_of(C7)} valid=True", flush=True)
    t=time.time()
    C, idx = gradient_descent(C7, pool, val)
    print(f"gradient descent from C7 -> index={idx}  [{time.time()-t:.1f}s]  "
          f"{'*** BEATS 1372! ***' if idx<1372 else ''}", flush=True)
    if idx < 1372:
        np.save("/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07/e7_best_C.npy", C)
        print("  saved improved sublattice basis")
