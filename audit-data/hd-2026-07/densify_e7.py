"""Densification search: given a valid sublattice C (index k), find a valid
SUPERLATTICE of index k/p (p | k) by adjoining an order-p glue vector x=C a/p,
where a in null(C mod p). This reduces the color count by a prime factor. ABPR's
gradient descent (short-vector replacement) does not perform these moves, so a
valid densification of C_7 would beat 1372. BFS over the densening tree."""
import numpy as np, sys, itertools, time
from sympy import factorint
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
from beat_e7 import build, short_pool, Validator, index_of
from e7_abpr import C7
from pool_cover import nullspace_modp

def modinv(a, p): return pow(int(a) % p, p-2, p)

def densifications(C, val):
    """all valid superlattices of C of index k/p (p prime factor of k)."""
    k = index_of(C)
    out = []
    for p in list(factorint(k).keys()):
        ns = nullspace_modp((C % p).tolist(), p)
        if not ns: continue
        for coeffs in itertools.product(range(p), repeat=len(ns)):
            if not any(coeffs): continue
            a = np.zeros(7, dtype=np.int64)
            for i,cf in enumerate(coeffs):
                if cf: a = (a + cf*np.array(ns[i], dtype=np.int64)) % p
            piv = next((i for i in range(7) if a[i] % p != 0), None)
            if piv is None: continue
            a = (a * modinv(a[piv], p)) % p            # a[piv]=1 mod p
            Ca = C @ a
            if np.any(Ca % p != 0): continue
            x = (Ca // p).astype(np.int64)
            Cn = C.copy(); Cn[:, piv] = x
            if index_of(Cn) == k // p and val.valid(Cn):
                out.append(Cn)
    return out

if __name__ == "__main__":
    M,G,B,diam,F = build()
    val = Validator(F)
    assert val.valid(C7)
    print(f"E7*: |F|={len(F)}  start C7 index={index_of(C7)}", flush=True)
    # BFS over densening tree, dedup by index (keep one witness per index)
    seen_idx = {1372}; frontier = [C7.copy()]; best = 1372; bestC = C7.copy()
    t0=time.time(); explored=0
    while frontier:
        C = frontier.pop()
        for Cn in densifications(C, val):
            explored += 1
            k = index_of(Cn)
            if k < best: best, bestC = k, Cn.copy(); print(f"  *** densified to index {k} (valid!) [{time.time()-t0:.0f}s] ***", flush=True)
            if k not in seen_idx:
                seen_idx.add(k); frontier.append(Cn)
    print(f"explored {explored} valid densifications; indices seen: {sorted(seen_idx)}", flush=True)
    print(f"=== best valid index = {best} {'(NEW chi(E^7)<=%d)'%best if best<1372 else '(no improvement; 1372 densening-minimal)'} ===", flush=True)
    if best < 1372:
        np.save("/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07/e7_best_C.npy", bestC)
