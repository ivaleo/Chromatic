"""Fast forbidden-distance-interval frontier via RATCHET.

Per lattice compute (int coords, D(v)/diam) once. For index k: find a valid
coloring at ell=1, take its EXACT width d=min D/diam over ker phi; then target
ell just above d and search again -- each success raises d with a fast find
(valid exists => found quickly), only the final refusal costs a full budget.
Avoids the binary-search's repeated full-budget boundary failures.

Reports the exact normalized width d(k) = max over found index-k colorings of
min_{v in Gamma\\0} D(v)/diam, i.e. chi(R^n,[1,d(k)]) <= k. Parallel across k."""
import sys, time, json
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
import numpy as np, combigeo
from lattices import CATALOG
from covrad import covering_radius
from campaign_hd import structures_rich_first
import multiprocessing as mp

_S = {}   # per-worker: name -> (n, diam, coords, drat)

def prep_vD(name, ell_max=1.75):
    B = CATALOG[name](); n = len(B)
    R,_ = covering_radius(B, n_dirs=2500, seed=1); diam = 2*R
    facets = combigeo.relevant_facets(B.tolist())
    Rmax = (ell_max+1.0)*diam + 1e-6
    vs = combigeo._vectors_near(B.tolist(), [0.0]*n, Rmax)
    Binv = np.linalg.inv(np.array(B))
    coords=[]; drat=[]
    for v in vs:
        if all(abs(x)<1e-9 for x in v): continue
        d = 2.0*combigeo.dist_to_halfspaces([0.5*x for x in v], facets)
        c = np.round(np.array(v) @ Binv).astype(int)
        coords.append([int(x) for x in c]); drat.append(d/diam)
    return n, diam, np.array(coords), np.array(drat)

def d_of_phi(phi, e_list, coords, drat):
    killed = np.ones(len(coords), dtype=bool)
    for j,e in enumerate(e_list):
        killed &= ((coords @ np.array(phi[j])) % e == 0)
    return float(drat[killed].min()) if killed.any() else float('inf')

def find_at(coords, drat, n, k, ell, ms, rs, seeds, cap=5):
    F = coords[drat < ell - 1e-12].tolist()
    for e_list in structures_rich_first(k, cap):
        for s in seeds:
            found, phi, idx = combigeo.min_conflicts(F, e_list, n, ms, rs, s)
            if found and idx == k:
                return phi, e_list
    return None

def ratchet_k(name, k):
    n, diam, coords, drat = _S[name]
    best = None
    # many cheap seeds at ell=1 (F_1 small): collect the widest coloring found
    for s in range(10):
        r = find_at(coords, drat, n, k, 1.0, 2000, 12, (s,), cap=4)
        if r is None: continue
        d = d_of_phi(*r, coords, drat)
        if best is None or d > best[0]: best = (d, r[0], r[1])
    if best is None:
        return {'name':name,'n':n,'k':k,'d':None}
    # lean ratchet: raise target just above current width; stop on refusal
    for _ in range(8):
        target = best[0] + 0.02
        r = find_at(coords, drat, n, k, target, 1800, 12, (0,1,2), cap=3)
        if r is None: break
        d = d_of_phi(*r, coords, drat)
        if d <= best[0] + 1e-9: break
        best = (d, r[0], r[1])
    return {'name':name,'n':n,'k':k,'diam':diam,'d':round(best[0],4),
            'e_list':best[2],'phi':best[1]}

def _init(store):
    global _S; _S = store
def _job(args):
    name,k = args
    t=time.time(); r=ratchet_k(name,k); r['secs']=round(time.time()-t,1); return r

if __name__ == "__main__":
    spec = eval(sys.argv[1])   # {name: [k,...]}
    store = {}
    for name in spec:
        t=time.time(); store[name] = prep_vD(name)
        print(f"prep {name}: n={store[name][0]} #shortvec={len(store[name][2])} diam={store[name][1]:.4f} [{time.time()-t:.1f}s]", flush=True)
    tasks = [(name,k) for name,ks in spec.items() for k in ks]
    results=[]
    with mp.Pool(max(2,min(8,mp.cpu_count()-2)), initializer=_init, initargs=(store,)) as pool:
        for r in pool.imap_unordered(_job, tasks):
            results.append(r)
            print(f"[{r['name']}/{r['k']}] d={r['d']} ({r.get('e_list')}) [{r.get('secs')}s]", flush=True)
    results.sort(key=lambda r:(r['n'],r['k']))
    OUT="/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07/interval_fast_results.json"
    with open(OUT,"w") as f: json.dump(results,f,indent=1)
    print("\n=== STAIRCASES (chi(R^n,[1,d]) <= k) ===")
    for r in results:
        if r['d']: print(f"  n={r['n']} k={r['k']:4d}: d={r['d']}  ({r['name']}, {r['e_list']})")
