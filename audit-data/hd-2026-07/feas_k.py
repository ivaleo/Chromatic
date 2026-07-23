"""Map feasible indices k (at l=1) for one lattice over [lo,hi], reliable budget."""
import sys, time
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
sys.path.insert(0,"/Users/mac/Documents/_My_code/Chromatic/audit-data")
import combigeo
from campaign_hd import prep, find_at_k
import multiprocessing as mp

_G = {}
def _init(nF): _G['nF'] = nF
def task(k):
    n, F = _G['nF']
    return (k, find_at_k(F, n, k, 2000, 15, (0, 1)) is not None)

if __name__ == "__main__":
    name=sys.argv[1]; lo=int(sys.argv[2]); hi=int(sys.argv[3])
    B,diam,F=prep(name); n=len(B)
    print(f"{name} |F|={len(F)} diam={diam:.4f} scan k in [{lo},{hi}] at l=1", flush=True)
    with mp.Pool(max(2,min(8,mp.cpu_count()-2)), initializer=_init, initargs=((n,F),)) as pool:
        feas=[k for k,ok in pool.map(task, range(lo,hi+1), chunksize=2) if ok]
    print(f"FEASIBLE k for {name}: {feas}", flush=True)
