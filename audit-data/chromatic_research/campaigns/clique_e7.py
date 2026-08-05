"""Max clique of the cell-conflict graph (connection set S = F ∪ -F): a set of
lattice cells pairwise realizing the forbidden distance. clique <= chi_infinity <=
ABPR index. If max clique == 1372, ABPR's E_7*/1372 is optimal even among general
(non-sublattice) cell colorings. If clique << 1372, a better cell coloring MAY exist."""
import numpy as np, sys, time
import combigeo
from chromatic_research.core.covrad import covering_radius
from chromatic_research.paths import results_path

def get_F(M):
    n = M.shape[1]; G = M.T @ M; B = np.linalg.cholesky(G)
    R,_ = covering_radius(B, n_dirs=600); diam = 2*R
    F = np.array(combigeo.forbidden_coords(B.tolist(), diam, 1.0), dtype=np.int64)
    return F

def greedy_clique(S_set, S_list, seed, rounds=1):
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(rounds):
        order = S_list[rng.permutation(len(S_list))]
        clique = [np.zeros(order.shape[1], dtype=np.int64)]
        cand = order
        while len(cand):
            v = cand[0]
            clique.append(v)
            # keep w in cand with (w - v) in S
            keep = [w for w in cand if tuple(int(x) for x in (w - v)) in S_set]
            cand = np.array(keep) if keep else np.empty((0, order.shape[1]), dtype=np.int64)
        if len(clique) > len(best): best = clique
    return best

if __name__ == "__main__":
    from chromatic_research.core import e7_abpr
    from chromatic_research.campaigns.conflict_graph import M_Anstar, C5
    cases = [("A5*", M_Anstar(5), 140), ("E7*", e7_abpr.M_E7(), 1372)]
    for name, M, idx in cases:
        F = get_F(M)
        S = np.vstack([F, -F])
        S_set = set(tuple(int(x) for x in r) for r in S)
        t=time.time()
        cl = greedy_clique(S_set, S, seed=0, rounds=6)
        print(f"{name}: index={idx} |F|={len(F)}  greedy max clique = {len(cl)} "
              f"({'== index -> ABPR optimal for cell colorings' if len(cl)>=idx else 'GAP: clique %d < index %d -> possible room'%(len(cl),idx)}) [{time.time()-t:.0f}s]", flush=True)
