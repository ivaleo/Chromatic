"""C++ vs Python min-conflicts benchmark on E6*/343 (n=6, |F|=3861)."""
import numpy as np, math, time, combigeo


def main():
    w = complex(-0.5, math.sqrt(3)/2); th = w - w.conjugate()
    def realify(vs):
        o=[]
        for v in vs:
            r=[]
            for z in v: r += [z.real, z.imag]
            o.append(r)
        return np.array(o)
    gens=[]
    for u in [(th,0,0),(1,1,1),(0,th,0)]:
        gens.append(u); gens.append(tuple(w*z for z in u))
    E6 = np.linalg.inv(realify(gens)).T; E6 = E6/abs(np.linalg.det(E6))**(1/6)
    lam1 = np.linalg.norm(combigeo.shortest_vector(E6.tolist())); diam = 2*lam1/math.sqrt(2)
    F = combigeo.forbidden_coords(E6.tolist(), diam, 1.0)
    Fnp = np.array(F, dtype=np.int64)
    n = 6; e_list = [7,7,7]
    print(f"n={n} |F|={len(F)} e_list={e_list}")

    # ---- pure-numpy Python min-conflicts (same algorithm) ----
    def py_min_conflicts(F, e_list, n, max_steps=3000, restarts=15, seed=0):
        rng = np.random.default_rng(seed)
        m = len(e_list); nf = len(F); e = np.array(e_list)
        for _ in range(restarts):
            phi = np.array([rng.integers(0, e_list[j], n) for j in range(m)])  # m x n
            res = np.array([(phi[j] @ F.T) % e_list[j] for j in range(m)])     # m x nf
            killed = (res == 0).all(axis=0)                                    # nf bool
            for step in range(max_steps):
                nk = int(killed.sum())
                if nk == 0:
                    return True, phi
                fsel = int(rng.choice(np.nonzero(killed)[0]))
                best = None; best_after = nk+1
                for _t in range(40):
                    j = int(rng.integers(m)); i = int(rng.integers(n))
                    if F[fsel][i] % e_list[j] == 0: continue
                    val = int(rng.integers(e_list[j])); delta = val - phi[j][i]
                    if delta == 0: continue
                    if (res[j][fsel] + delta*F[fsel][i]) % e_list[j] == 0: continue
                    newresj = (res[j] + delta*Fnp[:,i]) % e_list[j]
                    kill2 = killed.copy(); kill2 = (res==0).all(axis=0)
                    # recompute killed if row j replaced
                    after = int(np.logical_and(newresj==0, np.delete(res,j,0).__eq__(0).all(axis=0) if m>1 else (newresj==0)).sum())
                    if after < best_after: best_after=after; best=(j,i,val)
                    if after == 0: break
                if best is None or rng.integers(5)==0:
                    j=int(rng.integers(m)); i=int(rng.integers(n)); val=int(rng.integers(e_list[j])); best=(j,i,val)
                j,i,val = best; delta = val - phi[j][i]
                if delta != 0:
                    phi[j][i] = val
                    res[j] = (res[j] + delta*Fnp[:,i]) % e_list[j]
                    killed = (res==0).all(axis=0)
        return False, None

    # C++ timing
    tc=[]
    for s in range(8):
        t=time.time(); f,phi,idx=combigeo.min_conflicts(F,e_list,n,3000,15,s); tc.append(time.time()-t)
        if f: break
    print(f"C++    : found idx={idx} in {tc[-1]*1000:.0f}ms (median {sorted(tc)[len(tc)//2]*1000:.0f}ms over {len(tc)} seeds)")

    # Python timing
    tp=[]
    for s in range(8):
        t=time.time(); f,phi=py_min_conflicts(Fnp,e_list,n,3000,15,s); tp.append(time.time()-t)
        if f: break
    print(f"Python : found={f} in {tp[-1]*1000:.0f}ms over {len(tp)} seeds")
    print(f"speedup ~ {tp[-1]/tc[-1]:.0f}x (last-seed wall)")


if __name__ == "__main__":
    main()
