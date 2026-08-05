"""Exact CP-SAT solver for the coloring CSP (OR-Tools).

Find phi=(phi_1..phi_m), phi_j: Z^n -> Z/e_j, phi_j(x)=sum a_ji x_i, such that for
every forbidden f: NOT all phi_j(f)=0  (i.e. OR_j phi_j(f) != 0 mod e_j).

Unlike min-conflicts (local search: finds needles unreliably, never proves
infeasibility), CP-SAT (a) systematically FINDS rigid/ideal solutions, and (b)
returns UNSAT = a PROOF no coloring of this structure exists -> certifies a
minimal index. Encoding: reduce f mod e (=> nonneg sums), r=s mod e via
AddModuloEquality, bool b=[r!=0], AddBoolOr over forms."""
import sys, time
import numpy as np, combigeo
from ortools.sat.python import cp_model
from chromatic_research.core.lattices import CATALOG
from chromatic_research.core.covrad import covering_radius
from chromatic_research.core.campaign_hd import structures_rich_first
from chromatic_research.core.general_csp import index_and_check
from chromatic_research.paths import results_path

def solve_structure(F, e_list, n, time_limit=60, workers=8):
    F = np.asarray(F, dtype=np.int64)
    model = cp_model.CpModel()
    a = [[model.NewIntVar(0, e-1, f'a{j}_{i}') for i in range(n)] for j,e in enumerate(e_list)]
    for fi in range(len(F)):
        bs = []
        for j,e in enumerate(e_list):
            fmod = [int(x) % e for x in F[fi]]
            smax = sum((e-1)*x for x in fmod) or 1
            sp = model.NewIntVar(0, smax, f's{j}_{fi}')
            model.Add(sp == sum(int(fmod[i])*a[j][i] for i in range(n)))
            r = model.NewIntVar(0, e-1, f'r{j}_{fi}')
            model.AddModuloEquality(r, sp, e)
            b = model.NewBoolVar(f'b{j}_{fi}')
            model.Add(r >= 1).OnlyEnforceIf(b)
            model.Add(r == 0).OnlyEnforceIf(b.Not())
            bs.append(b)
        model.AddBoolOr(bs)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    st = solver.Solve(model)
    if st in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        phi = [[int(solver.Value(a[j][i])) for i in range(n)] for j in range(len(e_list))]
        return 'SAT', phi, solver.WallTime()
    if st == cp_model.INFEASIBLE:
        return 'UNSAT', None, solver.WallTime()
    return 'UNKNOWN', None, solver.WallTime()

def solve_index(F, n, k, time_limit=60, cap=6):
    for e_list in structures_rich_first(k, cap):
        st, phi, wt = solve_structure(F, e_list, n, time_limit)
        if st == 'SAT':
            idx, avoids = index_and_check(phi, e_list, F.tolist() if hasattr(F,'tolist') else F, n)
            return {'status':'SAT','k':k,'e_list':e_list,'index':idx,'avoids':bool(avoids),'wall':wt}
        if st == 'UNKNOWN':
            return {'status':'UNKNOWN','k':k,'e_list':e_list,'wall':wt}   # timed out
    return {'status':'UNSAT','k':k,'wall':wt}   # every structure proven infeasible

if __name__ == "__main__":
    name=sys.argv[1]; ks=eval(sys.argv[2]); tl=int(sys.argv[3]) if len(sys.argv)>3 else 60
    B=CATALOG[name](); n=len(B); R,_=covering_radius(B,n_dirs=500); diam=2*R
    F=np.array(combigeo.forbidden_coords(B.tolist(),diam,1.0),dtype=np.int64)
    print(f"{name}: n={n} |F_1|={len(F)} diam={diam:.4f} (CP-SAT, {tl}s/structure)",flush=True)
    for k in ks:
        t=time.time(); r=solve_index(F,n,k,time_limit=tl)
        if r['status']=='SAT':
            print(f"  k={k}: SAT index={r['index']} avoids={r['avoids']} struct={r['e_list']} [{time.time()-t:.1f}s wall]",flush=True)
        elif r['status']=='UNSAT':
            print(f"  k={k}: UNSAT (proven: no coloring) [{time.time()-t:.1f}s]",flush=True)
        else:
            print(f"  k={k}: timeout/unknown [{time.time()-t:.1f}s]",flush=True)
