"""Color the cell-conflict graph on a LARGER period Gamma (a sublattice of the
ABPR Lambda', hence still a valid period: Gamma∩F=∅). H_Gamma = Cayley(Lambda/Gamma,
F mod Gamma). chi(H_Gamma) <= ABPR index, and may be strictly less (the clique is
well below the index). Any proper coloring with c < index colors => chi(R^n) <= c,
a NEW bound OUTSIDE the ABPR sublattice scheme (non-subgroup graph coloring)."""
import numpy as np, sys, time, itertools
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
import combigeo
from covrad import covering_radius

def get_F(M):
    n=M.shape[1]; G=M.T@M; B=np.linalg.cholesky(G)
    R,_=covering_radius(B,n_dirs=600); diam=2*R
    return np.array(combigeo.forbidden_coords(B.tolist(),diam,1.0),dtype=np.int64)

def adjugate_int(C):
    det=int(round(abs(np.linalg.det(C.astype(float)))))
    return np.round(np.linalg.inv(C.astype(float))*det).astype(object), det

def build_graph(F, Gamma):
    """vertices = Lambda/Gamma; Cayley graph with connection set (F ∪ -F) mod Gamma."""
    A, det = adjugate_int(Gamma)
    detG = int(round(abs(np.linalg.det(Gamma.astype(float)))))
    n = Gamma.shape[0]
    def key(x):
        return tuple(int(v) % detG for v in (A @ np.array(x, dtype=object)))
    # enumerate group by BFS over standard generators
    e = [np.eye(n, dtype=np.int64)[i] for i in range(n)]
    gens = [key(ei) for ei in e]
    zero = tuple([0]*n)
    idx = {zero:0}; order=[zero]; qi=0
    # precompute key addition table via generator steps (group is abelian)
    # represent each element by its key; add by adding a lattice vector's key
    def keyadd(k, kv):  # add two keys (both are A x mod detG) - linear, so componentwise mod
        return tuple((k[i]+kv[i]) % detG for i in range(n))
    while qi < len(order):
        k = order[qi]; qi+=1
        for g in gens:
            nk = keyadd(k,g)
            if nk not in idx: idx[nk]=len(order); order.append(nk)
    V = len(order)                       # should equal detG
    # connection set keys
    conn = set()
    for f in F:
        conn.add(key(f)); conn.add(key(-f))
    conn.discard(zero)
    # adjacency: g ~ g + s
    adj = [set() for _ in range(V)]
    for k,i in idx.items():
        for s in conn:
            j = idx.get(keyadd(k,s))
            if j is not None and j!=i:
                adj[i].add(j)
    return adj, V

def dsatur(adj, V):
    color=[-1]*V; sat=[set() for _ in range(V)]; deg=[len(a) for a in adj]
    for _ in range(V):
        # pick uncolored vertex with max saturation, tie by degree
        u=max((i for i in range(V) if color[i]<0),
              key=lambda i:(len(sat[i]), deg[i]))
        used=set(color[j] for j in adj[u] if color[j]>=0)
        c=0
        while c in used: c+=1
        color[u]=c
        for j in adj[u]: sat[j].add(c)
    return max(color)+1

if __name__=="__main__":
    import e7_abpr
    from conflict_graph import M_Anstar, C5
    name=sys.argv[1] if len(sys.argv)>1 else "A5*"
    refis = eval(sys.argv[2]) if len(sys.argv)>2 else [[1,1,1,1,1],[2,1,1,1,1],[2,2,1,1,1],[3,1,1,1,1]]
    if name=="A5*": M,C,idx=M_Anstar(5),C5,140
    else: M,C,idx=e7_abpr.M_E7(),e7_abpr.C7,1372
    F=get_F(M)
    print(f"{name}: ABPR index={idx} |F|={len(F)}", flush=True)
    best=idx
    for r in refis:
        Gamma = C @ np.diag(r)
        t=time.time(); adj,V=build_graph(F,Gamma)
        ncol=dsatur(adj,V)
        if ncol<best: best=ncol
        print(f"  period refine diag{r}: |Lambda/Gamma|={V}  DSATUR colors={ncol}"
              f"{'  *** < %d : chi(R)<=%d ***'%(idx,ncol) if ncol<idx else ''} [{time.time()-t:.0f}s]", flush=True)
    print(f"=== best cell-coloring for {name}: {best} vs ABPR {idx} ===")
