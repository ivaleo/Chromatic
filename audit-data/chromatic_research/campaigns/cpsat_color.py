"""CP-SAT graph coloring of the cell-conflict graph on a period Gamma.
If chi(H_Gamma) < ABPR index, that is a NEW chi(R^n) bound outside the ABPR scheme.
Graph coloring is CP-SAT's strength (unlike the modular-arithmetic CSP)."""
import numpy as np, sys, time
from ortools.sat.python import cp_model
from chromatic_research.campaigns.conflict_color import get_F, build_graph
from chromatic_research.core import e7_abpr
from chromatic_research.campaigns.conflict_graph import M_Anstar, C5
from chromatic_research.paths import results_path

def color_with(adj, V, ncolors, tl=60, workers=8):
    m = cp_model.CpModel()
    col = [m.NewIntVar(0, ncolors-1, f'c{v}') for v in range(V)]
    # symmetry break: fix a clique's colors 0..q-1 if easy; at least fix vertex 0
    m.Add(col[0] == 0)
    seen=set()
    for u in range(V):
        for w in adj[u]:
            if u<w and (u,w) not in seen:
                m.Add(col[u] != col[w]); seen.add((u,w))
    s=cp_model.CpSolver(); s.parameters.max_time_in_seconds=tl; s.parameters.num_search_workers=workers
    st=s.Solve(m)
    if st in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        cols=set(s.Value(c) for c in col); return 'SAT', len(cols)
    if st==cp_model.INFEASIBLE: return 'UNSAT', None
    return 'UNKNOWN', None

if __name__=="__main__":
    name=sys.argv[1] if len(sys.argv)>1 else "A5*"
    diag=eval(sys.argv[2]) if len(sys.argv)>2 else [2,1,1,1,1]
    targets=eval(sys.argv[3]) if len(sys.argv)>3 else None
    tl=int(sys.argv[4]) if len(sys.argv)>4 else 60
    if name=="A5*": M,C,idx=M_Anstar(5),C5,140
    else: M,C,idx=e7_abpr.M_E7(),e7_abpr.C7,1372
    F=get_F(M); Gamma=C@np.diag(diag)
    adj,V=build_graph(F,Gamma)
    E=sum(len(a) for a in adj)//2
    print(f"{name}: period diag{diag} |V|={V} |E|={E} (ABPR idx {idx})", flush=True)
    if targets is None: targets=[idx-1, idx-2, idx-5]
    for t in targets:
        t0=time.time(); st,used=color_with(adj,V,t,tl=tl)
        print(f"  {t} colors: {st}"+(f" (used {used})" if used else "")
              +(f"  *** chi(R)<= {used} — BEATS {idx}! ***" if st=='SAT' and used<idx else "")
              +f" [{time.time()-t0:.0f}s]", flush=True)
        if st=='UNSAT':
            print(f"  -> chi(H) = {t+1} for this period (proven)", flush=True); break
