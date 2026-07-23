"""High-dimension campaign: minimal index k for a valid l=1 lattice coloring
(=> classical chi(R^n) <= k), across candidate lattices, via C++ min-conflicts.

Feasible indices are arithmetically special -> sweep the FULL [lb,kmax] range
bottom-up, stop at the first feasible k (that IS the minimum). Each screen call
is lean (1 seed, top-5 richest structures; valid found <0.3s, fails ~0.5s).
Lattices run in parallel; the winner is CONFIRMED at full budget/many seeds and
its phi recorded for exact certification."""
import sys, time, json, math
sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data")
import numpy as np, combigeo
from lattices import CATALOG
from covrad import covering_radius
from general_csp import invariant_factor_structures
import multiprocessing as mp

OUT = "/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07/campaign_hd_results.json"

def prep(name):
    B = CATALOG[name]()
    R, _ = covering_radius(B, n_dirs=2500, seed=1)
    diam = 2*R
    F = combigeo.forbidden_coords(B.tolist(), diam, 1.0)
    return B, diam, F

def structures_rich_first(k, cap=5):
    S = invariant_factor_structures(k)
    S.sort(key=lambda e: (-len(e), e))
    return S[:cap]

def find_at_k(F, n, k, max_steps, restarts, seeds, cap=5):
    for e_list in structures_rich_first(k, cap):
        for s in seeds:
            found, phi, idx = combigeo.min_conflicts(F, e_list, n, max_steps, restarts, s)
            if found and idx == k:
                return phi, e_list
    return None

def confirm(F, n, k):
    return find_at_k(F, n, k, 6000, 20, tuple(range(10)), cap=99)

def sweep_lattice(name, kmax, out_q):
    t0 = time.time()
    B, diam, F = prep(name); n = len(B)
    lb = 2**(n+1) - 1
    rec = {'name': name, 'n': n, 'diam': diam, 'nF': len(F), 'min_k': None}
    kfound = None; hit = None
    for k in range(lb, kmax+1):
        r = find_at_k(F, n, k, 1000, 4, (0,))
        if r is not None:
            kfound, hit = k, r; break
    if kfound is not None:
        # confirm at full budget (guards against a rare screen false-negative just below)
        lo = max(lb, kfound-3)
        for kk in range(lo, kfound+1):
            c = confirm(F, n, kk)
            if c is not None:
                kfound, hit = kk, c; break
        phi, e_list = hit
        rec.update({'min_k': kfound, 'e_list': e_list, 'phi': phi})
    rec['secs'] = round(time.time()-t0, 1)
    out_q.put(rec)
    print(f"[{name}] n={n} |F|={len(F)} diam={diam:.4f} -> min_k={rec['min_k']} "
          f"({rec.get('e_list')}) [{rec['secs']}s]", flush=True)

if __name__ == "__main__":
    JOBS = [
        ('A5*', 145), ('D5', 175), ('D5*', 175),
        ('E6*', 360), ('E6', 240), ('D6', 240), ('D6*', 240), ('A6*', 280),
        ('E7', 440), ('E7*', 540), ('A7*', 440), ('D7', 380), ('D7*', 540),
        ('E8', 740), ('D8', 660), ('A8*', 840), ('D8*', 970),
        ('A9*', 1740), ('D9', 1340), ('D9*', 1940),
    ]
    sel = sys.argv[1:]
    if sel: JOBS = [j for j in JOBS if j[0] in sel]
    out_q = mp.Queue(); results = []; active = []; i = 0
    NCORE = max(2, min(8, mp.cpu_count()-2))
    while i < len(JOBS) or active:
        while i < len(JOBS) and len(active) < NCORE:
            name, kmax = JOBS[i]; i += 1
            p = mp.Process(target=sweep_lattice, args=(name, kmax, out_q)); p.start()
            active.append(p)
        while not out_q.empty(): results.append(out_q.get())
        active = [p for p in active if p.is_alive()]
        time.sleep(0.5)
    while not out_q.empty(): results.append(out_q.get())
    results.sort(key=lambda r: (r['n'], r['min_k'] or 10**9))
    with open(OUT, "w") as f: json.dump(results, f, indent=1)
    print("\n=== SUMMARY (min k per lattice, l=1) ===", flush=True)
    best_by_n = {}
    for r in results:
        print(f"  n={r['n']} {r['name']:5s} min_k={r['min_k']} {r.get('e_list')}")
        if r['min_k'] and (r['n'] not in best_by_n or r['min_k'] < best_by_n[r['n']][1]):
            best_by_n[r['n']] = (r['name'], r['min_k'])
    print("=== BEST per dimension (classical chi upper bound) ===")
    for nn in sorted(best_by_n): print(f"  chi(R^{nn}) <= {best_by_n[nn][1]}  via {best_by_n[nn][0]}")
