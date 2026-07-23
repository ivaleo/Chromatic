"""Simulated annealing over VALID E_7* sublattices to escape ABPR's local optimum
(index 1372) and find a smaller index. Move = replace one generator by a short
non-forbidden vector; accept if valid (rank 7, avoids F) per annealing schedule.
Parallel chains; any index < 1372 is a new upper bound on chi(E^7)."""
import numpy as np, sys, time
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
from beat_e7 import build, short_pool, Validator, index_of, gradient_descent
from e7_abpr import C7
import multiprocessing as mp

_G = {}
def _init(d): _G.update(d)

def sa_chain(args):
    seed, steps, radius = args
    B = _G['B']; F = _G['F']; pool = _G['pool']; val = Validator(F)
    rng = np.random.default_rng(seed)
    npool = len(pool)
    C = C7.copy().astype(np.int64); cur = index_of(C)
    best = cur; bestC = C.copy()
    T0, T1 = 60.0, 0.5
    for t in range(steps):
        T = T0 * (T1/T0) ** (t/steps)
        j = int(rng.integers(7))
        # bias toward short pool vectors (index ~ geometric volume)
        gi = int(rng.integers(min(npool, 400 + t//50)))
        Cn = C.copy(); Cn[:, j] = pool[gi]
        d = index_of(Cn)
        if d == 0: continue
        if not val.valid(Cn): continue
        dd = d - cur
        if dd <= 0 or rng.random() < np.exp(-dd / T):
            C = Cn; cur = d
            if cur < best:
                best = cur; bestC = C.copy()
    # polish with gradient descent
    Cg, bg = gradient_descent(bestC, pool, val, max_sweeps=20)
    if bg < best: best, bestC = bg, Cg
    return best, bestC.tolist()

if __name__ == "__main__":
    radius = float(sys.argv[1]) if len(sys.argv)>1 else 2.8
    steps = int(sys.argv[2]) if len(sys.argv)>2 else 4000
    chains = int(sys.argv[3]) if len(sys.argv)>3 else 8
    M,G,B,diam,F = build()
    pool = short_pool(B, F, radius)
    print(f"E7*: |F|={len(F)} pool={len(pool)} (|v|<{radius}); {chains} SA chains x {steps} steps", flush=True)
    store = dict(B=B, F=F, pool=pool)
    args = [(s, steps, radius) for s in range(chains)]
    t0=time.time(); results=[]
    with mp.Pool(min(chains, mp.cpu_count()-1), initializer=_init, initargs=(store,)) as p:
        for best, C in p.imap_unordered(sa_chain, args):
            results.append((best, C))
            tag = " *** BEATS 1372! ***" if best < 1372 else ""
            print(f"  chain best index={best}{tag} [{time.time()-t0:.0f}s]", flush=True)
    gb = min(results, key=lambda r: r[0])
    print(f"=== global best index = {gb[0]} {'(NEW: chi(E^7) <= %d)'%gb[0] if gb[0]<1372 else '(no improvement)'} ===", flush=True)
    if gb[0] < 1372:
        np.save("/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07/e7_best_C.npy", np.array(gb[1]))
        print("  saved improved basis to e7_best_C.npy", flush=True)
