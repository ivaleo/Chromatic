"""Massively randomized short-vector sublattice search for E_7* (ABPR's own
method, run hard) + densification of every valid find. Pick 7 short non-forbidden
vectors, check full-rank + validity, then densify to a local index-minimum.
Any valid index < 1372 is a new upper bound on chi(E^7)."""
import numpy as np, sys, time
from chromatic_research.campaigns.beat_e7 import build, short_pool, Validator, index_of
from chromatic_research.campaigns.densify_e7 import densifications
import multiprocessing as mp
from chromatic_research.paths import results_path

_G = {}
def _init(d): _G.update(d)

def full_densify(C, val):
    best = C; bidx = index_of(C); changed = True
    while changed:
        changed = False
        for Cn in densifications(best, val):
            k = index_of(Cn)
            if k < bidx: best, bidx = Cn, k; changed = True; break
    return best, bidx

def worker(args):
    seed, trials, nshort = args
    pool = _G['pool']; F = _G['F']; val = Validator(F)
    rng = np.random.default_rng(seed)
    npool = len(pool)
    best = 10**9; bestC = None; nvalid = 0
    lim = min(npool, nshort)
    for _ in range(trials):
        idx = rng.choice(lim, size=7, replace=False)
        C = pool[idx].T.copy()                      # 7x7, columns = chosen vectors
        d = index_of(C)
        if d == 0 or d >= best:
            if d == 0: continue
        if not val.valid(C): continue
        nvalid += 1
        Cf, k = full_densify(C, val)
        if k < best: best, bestC = k, Cf
    return best, (bestC.tolist() if bestC is not None else None), nvalid

if __name__ == "__main__":
    nshort = int(sys.argv[1]) if len(sys.argv)>1 else 120
    trials = int(sys.argv[2]) if len(sys.argv)>2 else 300000
    chains = int(sys.argv[3]) if len(sys.argv)>3 else 8
    radius = float(sys.argv[4]) if len(sys.argv)>4 else 2.8
    M,G,B,diam,F = build()
    pool = short_pool(B, F, radius)
    print(f"E7*: |F|={len(F)} pool={len(pool)}; sampling 7 of shortest {nshort}; "
          f"{chains}x{trials} trials", flush=True)
    store = dict(pool=pool, F=F)
    args = [(s, trials, nshort) for s in range(chains)]
    t0=time.time(); results=[]
    with mp.Pool(min(chains, mp.cpu_count()-1), initializer=_init, initargs=(store,)) as p:
        for best, C, nv in p.imap_unordered(worker, args):
            results.append((best, C))
            tag = " *** BEATS 1372! ***" if best < 1372 else ""
            print(f"  chain: {nv} valid sublattices, best index={best}{tag} [{time.time()-t0:.0f}s]", flush=True)
    gb = min(results, key=lambda r: r[0])
    print(f"=== global best valid index = {gb[0]} {'(NEW chi(E^7)<=%d!)'%gb[0] if gb[0]<1372 else '(no improvement over 1372)'} ===", flush=True)
    if gb[0] < 1372 and gb[1]:
        np.save(results_path("e7_best_C.npy"), np.array(gb[1]))
