"""Optimization over Gram forms: search generic (non-symmetric) lattices for a
valid coloring at an index BELOW the symmetric threshold (chi(R^5)<=140 via A5*,
chi(R^6)<=343 via E6*), exactly as the 4D breakthrough 49->45 used generic
lattices.

Lattice basis B = lower-triangular (exp-diagonal), normalized |det B|=1.
Objective (minimize): best_killed(B, k) = min over invariant-factor structures of
the fewest forbidden vectors min-conflicts leaves unseparated at index k, l=1.
best_killed==0  <=>  a valid k-coloring exists  <=>  chi(R^n) <= actual index.

CMA-ES (gradient-free, noise-tolerant) seeded from the symmetric champion +
random restarts; population evaluated in parallel."""
import sys, time, json, math
import numpy as np, combigeo, cma
from chromatic_research.core.lattices import CATALOG
from chromatic_research.core.covrad import covering_radius
from chromatic_research.core.campaign_hd import structures_rich_first
import multiprocessing as mp
from chromatic_research.paths import results_path

# ---------- lattice parametrization ----------
def tri_index(n): return [(i,j) for i in range(n) for j in range(i+1)]

def build_B(params, n):
    """lower-triangular basis, exp on diagonal (pos-def), normalized |det|=1."""
    L = np.zeros((n,n)); idx = tri_index(n)
    for p,(i,j) in zip(params, idx):
        L[i,j] = math.exp(p) if i==j else p
    d = abs(np.prod(np.diag(L)))
    if d < 1e-12: return None
    L = L / d**(1.0/n)
    return L

def params_from_B(B, n):
    Q = np.array(B) @ np.array(B).T
    Q = Q / abs(np.linalg.det(Q))**(1.0/n)
    L = np.linalg.cholesky(Q)                    # lower-triangular
    out=[]
    for (i,j) in tri_index(n):
        out.append(math.log(L[i,i]) if i==j else L[i,j])
    return out

# ---------- objective ----------
_CFG = {}
def _init(cfg): _CFG.update(cfg)

def evaluate(params):
    n = _CFG['n']; k = _CFG['k']; ndir = _CFG['ndir']
    ms, rs = _CFG['ms'], _CFG['rs']; cap = _CFG['cap']
    B = build_B(params, n)
    if B is None: return (1e9, None)
    try:
        R,_ = covering_radius(B, n_dirs=ndir, seed=7); diam = 2*R
        F = combigeo.forbidden_coords(B.tolist(), diam, 1.0)
    except Exception:
        return (1e9, None)
    if len(F) == 0: return (1e9, None)
    best = len(F)+1
    for e_list in structures_rich_first(k, cap):
        found, bk, idx = combigeo.min_conflicts_cost(F, e_list, n, ms, rs, 0)
        if bk < best: best = bk
        if found and idx <= k:
            return (0.0, {'e_list':e_list,'index':int(idx),'nF':len(F),'params':list(params)})
    # cost = best_killed (dominant, feasibility) + gradient toward SMALL forbidden
    # sets (generic-lattice insight: fewer constraints => colorable at lower index)
    nref = _CFG['nref']
    return (best + 0.3*len(F)/nref, {'best_killed':int(best),'nF':len(F),'params':list(params)})

def run(name, targets, gens=70, popsize=None, sigma0=0.22, n_restarts=3, seed=0):
    n = len(CATALOG[name]()); results={}
    x0 = params_from_B(CATALOG[name](), n)
    nref = len(combigeo.forbidden_coords(CATALOG[name]().tolist(),
                2*covering_radius(CATALOG[name](), n_dirs=400)[0], 1.0))
    ndir = 260 if n<=5 else 360
    rng = np.random.default_rng(seed)
    for k in targets:
        cfg = dict(n=n, k=k, ndir=ndir, ms=1800, rs=14, cap=4, nref=nref)  # rs>=14: reliable oracle
        ps = popsize or (12 + 2*n)
        best_cost=1e18; best_info=None; t0=time.time()
        with mp.Pool(max(2,min(8,mp.cpu_count()-2)), initializer=_init, initargs=(cfg,)) as pool:
            for rst in range(n_restarts):
                # restart 0: seed at champion; others: perturbed / random starts
                start = list(x0) if rst==0 else [xi + rng.normal(0,0.5) for xi in x0]
                es = cma.CMAEvolutionStrategy(start, sigma0,
                      {'popsize':ps,'maxiter':gens,'seed':seed+k+100*rst,'verbose':-9,'tolfun':1e-11})
                gen=0
                while not es.stop():
                    X = es.ask()
                    res = pool.map(evaluate, X)
                    costs = [c for c,_ in res]
                    es.tell(X, costs)
                    gi = int(np.argmin(costs))
                    if costs[gi] < best_cost:
                        best_cost = costs[gi]; best_info = res[gi][1] or {'params':list(X[gi])}
                    gen+=1
                    if gen%10==0:
                        print(f"  [{name} k={k} r{rst}] gen {gen}: best_cost={best_cost:.3f} "
                              f"(best_killed={best_info.get('best_killed','?') if best_info else '?'}) [{time.time()-t0:.0f}s]", flush=True)
                    if best_cost==0.0:
                        print(f"  *** {name} k={k}: VALID COLORING, index={best_info.get('index')} "
                              f"struct={best_info.get('e_list')} -> chi(R^{n}) <= {best_info.get('index')} ***", flush=True)
                        break
                if best_cost==0.0: break
        # HIGH-BUDGET confirm of the best shape (guards against oracle false-negatives)
        confirmed = None
        if best_info and best_cost > 0 and best_info.get('params'):
            B = build_B(best_info['params'], n)
            R,_ = covering_radius(B, n_dirs=600); diam=2*R
            F = combigeo.forbidden_coords(B.tolist(), diam, 1.0)
            for e_list in structures_rich_first(k, 8):
                for s in range(6):
                    found, bk, idx = combigeo.min_conflicts_cost(F, e_list, n, 5000, 40, s)
                    if found and idx <= k:
                        confirmed = {'index':int(idx),'e_list':e_list}; break
                if confirmed: break
            if confirmed:
                best_cost = 0.0
                print(f"  *** {name} k={k}: CONFIRMED valid coloring index={confirmed['index']} "
                      f"struct={confirmed['e_list']} -> chi(R^{n}) <= {confirmed['index']} ***", flush=True)
        results[k] = {'best_cost':best_cost, 'best_killed':(best_info or {}).get('best_killed'),
                      'confirmed':confirmed, 'info':best_info, 'secs':round(time.time()-t0,1)}
        print(f"[{name} k={k}] DONE best_cost={best_cost:.3f} confirmed={bool(confirmed)} [{results[k]['secs']}s]", flush=True)
    return results

if __name__ == "__main__":
    spec = eval(sys.argv[1])            # {'A5*':[138,135,132,130,128,126,120], ...}
    gens = int(sys.argv[2]) if len(sys.argv)>2 else 60
    allres={}
    for name,ks in spec.items():
        allres[name] = run(name, ks, gens=gens)
    OUT=results_path("gram_opt_results.json")
    json.dump(allres, open(OUT,"w"), indent=1, default=float)
    print("\n=== SUMMARY ===")
    for name,ks in allres.items():
        for k,r in ks.items():
            tag = f"BEAT! index={r['info'].get('index')}" if r['best_cost']==0 else f"cost={r['best_cost']:.2f}"
            print(f"  {name} target k={k}: {tag}")
