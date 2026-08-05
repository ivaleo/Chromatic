"""Flexible g-invariant colorings: for an arbitrary functional a (mod m), use the
whole g-ORBIT of forms {a, a A_g, ..., a A_g^n}. Combined kernel is g-invariant
(no eigenvector restriction), so a ranges over ALL of (Z/m)^n -> much broader than
the eigenvector family. Valid iff a is nonzero (mod m) on some member of every
g-orbit of F. Index = |image of x -> (a·g^j x)_j|. Search small m, min index."""
import sys, numpy as np, itertools
from chromatic_research.campaigns.invariant_search import setup, orbit_reps
from chromatic_research.paths import results_path

def orbit_members(reps, Ag, order):
    """for each rep, its (order) g-orbit coord-vectors, stacked: (nreps, order, n)."""
    out = []
    for f in reps:
        orb = []; c = np.array(f, dtype=np.int64)
        for _ in range(order):
            orb.append(c.copy()); c = Ag @ c
        out.append(np.array(orb))
    return np.array(out)                 # (nreps, order, n)

def companion_index(a, Ag, m, n, order):
    """index = |image of x -> (a·A_g^j x)_j mod m|; via subgroup gen by columns."""
    b = []; cur = np.array(a, dtype=np.int64)
    for _ in range(order):
        b.append(cur % m); cur = cur @ Ag        # b_j = a A_g^j  (row action)
    B = np.array(b) % m                            # (order, n)
    gens = [tuple(int(B[j,i]) % m for j in range(order)) for i in range(n)]
    seen = {tuple([0]*order)}; fr=[tuple([0]*order)]
    while fr:
        x = fr.pop()
        for g in gens:
            y = tuple((x[j]+g[j])%m for j in range(order))
            if y not in seen: seen.add(y); fr.append(y)
    return len(seen)

def search(n, ms, cap=None, sample=None):
    S = setup(n); Ag, Fc = S['Ag'], S['Fc']
    reps = orbit_reps(Fc, Ag, n+1)
    order = n+1
    OM = orbit_members(reps, Ag, order)            # (nreps, order, n)
    print(f"n={n}: |F|={len(Fc)} reps={len(reps)}", flush=True)
    best = None
    rng = np.random.default_rng(0)
    for m in ms:
        # candidate functionals a in (Z/m)^n (exhaustive if small, else sampled)
        total = m**n
        if sample and total > sample:
            cand = rng.integers(0, m, size=(sample, n))
        else:
            cand = np.array(list(itertools.product(range(m), repeat=n)), dtype=np.int64)
        found_m = 0
        for a in cand:
            if not a.any(): continue
            # valid iff every rep-orbit has a nonzero dot mod m
            dots = (OM @ a) % m                      # (nreps, order)
            alive = (dots != 0).any(axis=1)          # rep separated?
            if alive.all():
                idx = companion_index(a, Ag, m, n, order)
                if best is None or idx < best[0]:
                    # full-F verify: build companion forms, check all Fc
                    b=[]; cur=a.copy()
                    for _ in range(order): b.append(cur % m); cur = cur @ Ag
                    fullkill = np.ones(len(Fc), bool)
                    for bj in b: fullkill &= ((Fc @ (np.array(bj)%m)) % m == 0)
                    if not fullkill.any():
                        best = (idx, m, [int(x) for x in a])
                        print(f"  m={m}: VALID index={idx}  a={best[2]}", flush=True)
                found_m += 1
        print(f"  m={m}: {found_m} valid functionals among {len(cand)} tried", flush=True)
    if best: print(f"=== best g-orbit coloring: chi(R^{n}) <= {best[0]} (m={best[1]}) ===", flush=True)
    else: print("  none valid", flush=True)
    return best

if __name__ == "__main__":
    n=int(sys.argv[1]); ms=eval(sys.argv[2]); samp=int(sys.argv[3]) if len(sys.argv)>3 else 200000
    search(n, ms, sample=samp)
