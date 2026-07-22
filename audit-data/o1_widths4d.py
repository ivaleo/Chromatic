"""O1: максимальная ширина запрещённого интервала d(k) в R^4 при k=49..64
через оптимизацию метрики по ВСЕМ формам (генерические решётки, не только
классические D4/A4*). Проверяем гипотезу пользователя: при k>49 классические
решётки могут быть неоптимальны по ширине."""
import json
import numpy as np
from multiprocessing import Pool

def unpack(x):
    L=np.zeros((4,4)); L[np.tril_indices(4)]=x
    Q=L@L.T; d=abs(np.linalg.det(Q))
    return Q/d**0.25 if d>1e-12 else None
def pack(Q): return np.linalg.cholesky(Q)[np.tril_indices(4)]
def norm_gram(M):
    G=M@M.T; return G/abs(np.linalg.det(G))**0.25

def one(args):
    x0,k,mf=args
    import combigeo
    from scipy.optimize import minimize
    def d_of(Q):
        if Q is None: return 0.0
        try: return combigeo.find_optimal(np.linalg.cholesky(Q+1e-12*np.eye(4)).tolist(),index=k,threads=1).normalized
        except Exception: return 0.0
    r=minimize(lambda x:-d_of(unpack(x)),np.asarray(x0),method="Nelder-Mead",
               options={"maxfev":mf,"xatol":1e-7,"fatol":1e-11})
    return (-r.fun,r.x.tolist())

if __name__=="__main__":
    D4=norm_gram(np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]],float))
    A4G=np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]],float)
    A4S=norm_gram(np.linalg.inv(np.linalg.cholesky(A4G)).T)
    # предыдущие победители-ширины как старты
    try: prev=json.load(open("/Users/mac/Documents/_My_code/Chromatic/audit-data/n2_4d_frontier.json"))
    except Exception: prev={}
    rng=np.random.default_rng(4964)
    out={}
    classical={49:1.080123,54:1.072381,56:1.183216}  # √(7/6),√(23/20),√(7/5)
    with Pool(8) as pool:
        for k in range(49,65):
            starts=[pack(D4),pack(A4S)]
            pw=prev.get(f"width{k}",{}).get("Q")
            if pw: starts.append(pack(np.array(pw)))
            while len(starts)<14:
                b=starts[rng.integers(0,min(3,len(starts)))]
                starts.append(b*(1+rng.normal(scale=float(rng.choice([0.03,0.08,0.15])),size=10)))
            res=pool.map(one,[(s.tolist(),k,600) for s in starts])
            bd,bx=max(res,key=lambda t:t[0])
            cl=classical.get(k)
            mark=f" (классич. {cl:.4f}, +{bd-cl:.4f})" if cl else ""
            print(f"k={k}: max d = {bd:.6f}{mark}",flush=True)
            out[k]={"d":bd,"Q":unpack(np.asarray(bx)).tolist()}
    json.dump(out,open("/Users/mac/Documents/_My_code/Chromatic/audit-data/o1_widths4d.json","w"),indent=1)
    print("DONE",flush=True)
