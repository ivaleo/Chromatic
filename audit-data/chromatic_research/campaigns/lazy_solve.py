"""Lazy constraint generation (constraint-core / CEGAR) for the coloring CSP.

Find phi: Z^n -> G (|G|=k) with phi(f)!=0 for ALL f in F, where |F| is huge (dim>=7).
Instead of running min-conflicts on all of F, keep a small ACTIVE core S:
  1. solve phi on S (fast, |S| << |F|)
  2. check full F: violated = {f: phi(f)=0}
  3. if none -> phi is valid on all F  (DONE, found)
     if S itself unsolvable -> F unsolvable (subset infeasible => full infeasible)
     else add worst violators to S, repeat.
Sound: subset of F, so 'unsolvable on S' => unsolvable on F; 'solves S and no
violation' => solves F. Converges because F finite; the core |S| at convergence
is usually tiny. This turns |F|~2e4 min-conflicts into |S|~few-hundred."""
import sys, time
import numpy as np, combigeo
from chromatic_research.core.lattices import CATALOG
from chromatic_research.core.covrad import covering_radius
from chromatic_research.core.campaign_hd import structures_rich_first
from chromatic_research.paths import results_path

def prep_F(name, ell=1.0):
    B = CATALOG[name](); n = len(B)
    R,_ = covering_radius(B, n_dirs=400 if n<=6 else 500); diam = 2*R
    F = np.array(combigeo.forbidden_coords(B.tolist(), diam, ell), dtype=np.int64)
    # sort by lattice-norm (shortest first = fundamental core)
    G = np.array(B) @ np.array(B).T
    norms = np.einsum('ij,jk,ik->i', F, G, F)
    order = np.argsort(norms)
    return B, n, diam, F[order]

def violations(phi, e_list, F):
    killed = np.ones(len(F), dtype=bool)
    for j,e in enumerate(e_list):
        killed &= ((F @ np.array(phi[j])) % e == 0)
    return np.nonzero(killed)[0]

def lazy_solve(F, e_list, n, k, init=250, batch=150, ms=2500, rs=16, seeds=(0,1,2),
               max_rounds=60, verbose=False):
    """Returns (phi, |core|, rounds) if valid coloring found; else (None, |core|, rounds)."""
    N = len(F)
    core = list(range(min(init, N)))
    for rnd in range(max_rounds):
        S = F[core]
        sol = None
        for s in seeds:
            found, phi, idx = combigeo.min_conflicts(S.tolist(), e_list, n, ms, rs, s)
            if found and idx == k:
                sol = phi; break
        if sol is None:
            return (None, len(core), rnd)          # core unsolvable => F unsolvable (subset)
        vio = violations(sol, e_list, F)
        if len(vio) == 0:
            return (sol, len(core), rnd)            # solves full F  -> VALID
        if verbose:
            print(f"    round {rnd}: |core|={len(core)} violations={len(vio)}", flush=True)
        # add worst violators (already norm-sorted: take smallest-norm violators first)
        add = [i for i in vio.tolist() if i not in set(core)][:batch]
        core.extend(add)
    return (None, len(core), max_rounds)

def solve_index(F, n, k, cap=5, **kw):
    for e_list in structures_rich_first(k, cap):
        phi, core, rnds = lazy_solve(F, e_list, n, k, **kw)
        if phi is not None:
            return {'ok':True,'k':k,'e_list':e_list,'core':core,'rounds':rnds}
    return {'ok':False,'k':k,'core':core,'rounds':rnds}

if __name__ == "__main__":
    name = sys.argv[1]; ks = eval(sys.argv[2])
    B,n,diam,F = prep_F(name)
    print(f"{name}: n={n} |F|={len(F)} diam={diam:.4f}", flush=True)
    for k in ks:
        t=time.time(); r = solve_index(F, n, k, verbose=True)
        tag = f"VALID core={r['core']} rounds={r['rounds']} ({r['e_list']})" if r['ok'] else f"none (core={r['core']})"
        print(f"  k={k}: {tag} [{time.time()-t:.1f}s]", flush=True)
