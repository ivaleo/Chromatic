"""O2: R^3 — (A) эмпирическая проверка предела Кулсона (можно ли d>=1 при k<15?);
(B) расширение таблицы интервальных ширин до k=50 оптимизацией по всем формам."""
import json
import numpy as np
from multiprocessing import Pool
from chromatic_research.paths import results_path

def unpack(x):
    L=np.zeros((3,3)); L[np.tril_indices(3)]=x
    Q=L@L.T; d=abs(np.linalg.det(Q))
    return Q/d**(1/3) if d>1e-10 else None
def pack(Q): return np.linalg.cholesky(Q)[np.tril_indices(3)]
def norm_gram(M):
    G=M@M.T; return G/abs(np.linalg.det(G))**(1/3)

def one(args):
    x0,k,mf=args
    import combigeo
    from scipy.optimize import minimize
    def d_of(Q):
        if Q is None: return 0.0
        try: return combigeo.find_optimal(np.linalg.cholesky(Q+1e-12*np.eye(3)).tolist(),index=k,threads=1).normalized
        except Exception: return 0.0
    r=minimize(lambda x:-d_of(unpack(x)),np.asarray(x0),method="Nelder-Mead",
               options={"maxfev":mf,"xatol":1e-7,"fatol":1e-11})
    return (-r.fun,r.x.tolist())

if __name__=="__main__":
    BCC=norm_gram(np.array([[2,0,0],[0,2,0],[1,1,1]],float))
    FCC=norm_gram(np.array([[1,1,0],[1,0,1],[0,1,1]],float))
    Z3=norm_gram(np.eye(3))
    rng=np.random.default_rng(3)
    out={}
    with Pool(8) as pool:
        # (A) предел: k = 8..15 (можно ли пробить порог раньше 15?)
        print("=== (A) проверка предела Кулсона (порог d>=1) ===",flush=True)
        for k in range(8,16):
            starts=[pack(BCC),pack(FCC),pack(Z3)]
            while len(starts)<12:
                b=starts[rng.integers(0,3)]
                starts.append(b*(1+rng.normal(scale=float(rng.choice([0.05,0.12,0.25])),size=6)))
            res=pool.map(one,[(s.tolist(),k,700) for s in starts])
            bd,_=max(res,key=lambda t:t[0])
            print(f"k={k}: max d = {bd:.6f}  {'>=1 !!!' if bd>=1 else '< 1'}",flush=True)
            out[f"floor_k{k}"]=bd
        # (B) ширины k=15..50
        print("=== (B) интервальные ширины k=15..50 ===",flush=True)
        for k in range(15,51):
            starts=[pack(BCC),pack(FCC),pack(Z3)]
            while len(starts)<10:
                b=starts[rng.integers(0,3)]
                starts.append(b*(1+rng.normal(scale=float(rng.choice([0.05,0.15])),size=6)))
            res=pool.map(one,[(s.tolist(),k,500) for s in starts])
            bd,bx=max(res,key=lambda t:t[0])
            print(f"k={k}: max d = {bd:.6f}",flush=True)
            out[f"width_k{k}"]={"d":bd,"Q":unpack(np.asarray(bx)).tolist()}
    json.dump(out,open(results_path("o2_r3.json"),"w"),indent=1)
    print("DONE",flush=True)
