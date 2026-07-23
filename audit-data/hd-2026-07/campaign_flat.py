"""Flat parallel-pool attack: map feasibility over all (lattice,k) for the
promising lattices, dims 5-6. Perfect core utilization; yields the full set of
feasible indices per lattice (min = classical chi upper bound; also feeds the
interval phase). Winner neighborhood confirmed at full budget."""
import sys, time, json
sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")
import combigeo
from campaign_hd import prep, structures_rich_first, find_at_k, confirm
import multiprocessing as mp

OUT = "/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07/campaign_flat_results.json"
_G = {}
def _init(fd):
    global _G; _G = fd
def screen_task(args):
    name, k = args
    n, F = _G[name]
    # calibrated on the HARD known case A5*/140: 2000x15 -> ~100% find; 2 seeds for margin
    return (name, k, find_at_k(F, n, k, 2000, 15, (0, 1)) is not None)

# (lattice, k_lo, k_hi) — promising lattices only; ranges cover below+at champion
RANGES = [
    ('A5*', 63, 141), ('D5*', 63, 180),
    ('E6*', 127, 343), ('A6*', 127, 343), ('D6*', 127, 343),
]

if __name__ == "__main__":
    fd = {}; meta = {}
    for name, lo, hi in RANGES:
        t=time.time(); B,diam,F = prep(name); n=len(B)
        fd[name] = (n, F); meta[name] = dict(n=n, diam=diam, nF=len(F), lo=lo, hi=hi)
        print(f"prep {name}: n={n} |F|={len(F)} diam={diam:.4f} [{time.time()-t:.1f}s]", flush=True)
    tasks = [(name, k) for name,lo,hi in RANGES for k in range(lo, hi+1)]
    print(f"total tasks: {len(tasks)}", flush=True)
    feas = {name: [] for name,_,_ in RANGES}
    NCORE = max(2, min(9, mp.cpu_count()-1))
    t0=time.time(); done=0
    with mp.Pool(NCORE, initializer=_init, initargs=(fd,)) as pool:
        for name,k,ok in pool.imap_unordered(screen_task, tasks, chunksize=3):
            done+=1
            if ok: feas[name].append(k)
            if done % 150 == 0: print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)
    # confirm min per lattice
    results=[]
    for name,lo,hi in RANGES:
        ks=sorted(feas[name]); n=meta[name]['n']; rec=dict(name=name, **meta[name], feasible=ks, min_k=None)
        for k in ks:
            c=confirm(fd[name][1], n, k)
            if c is not None:
                rec['min_k']=k; rec['e_list']=c[1]; rec['phi']=c[0]; break
        results.append(rec)
        print(f"[{name}] n={n} min_k={rec['min_k']} feasible={ks}", flush=True)
    with open(OUT,"w") as f: json.dump(results,f,indent=1)
    print(f"\n=== DONE [{time.time()-t0:.0f}s] ===", flush=True)
    best={}
    for r in results:
        if r['min_k'] and (r['n'] not in best or r['min_k']<best[r['n']][1]): best[r['n']]=(r['name'],r['min_k'])
    for nn in sorted(best): print(f"  chi(R^{nn}) <= {best[nn][1]} via {best[nn][0]}")
