"""R2: попытка пробить 49 в R^4 — оптимизация метрики по вторичным конусам.

У генерической решётки 111- при k=48 d = 0.99941 (представитель конуса).
Максимизируем d(k) по ВСЕМ формам Q = sum t_i R_i внутри конуса (t_i > 0,
масштаб фиксирован) Нелдером-Мидом в лог-параметризации. Если d(48) >= 1 —
это chi(R^4) <= 48 (сенсация, перепроверять всеми средствами); ожидаемый
исход по гипотезе Arman et al. — потолок ниже 1. Заодно: максимальная ширина
на k=49 (лучше ли sqrt(7/6)?).
"""
import json
import numpy as np
from scipy.optimize import minimize
import combigeo

R = {1:[[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],2:[[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]],
     3:[[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]],4:[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],
     5:[[1,-1,0,0],[-1,1,0,0],[0,0,0,0],[0,0,0,0]],
     6:[[1,0,-1,0],[0,0,0,0],[-1,0,1,0],[0,0,0,0]],7:[[1,0,0,-1],[0,0,0,0],[0,0,0,0],[-1,0,0,1]],
     8:[[0,0,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,0]],9:[[0,0,0,0],[0,1,0,-1],[0,0,0,0],[0,-1,0,1]],
     10:[[0,0,0,0],[0,0,0,0],[0,0,1,-1],[0,0,-1,1]],
     11:[[4,2,-2,-2],[2,4,-2,-2],[-2,-2,4,0],[-2,-2,0,4]],
     12:[[1,1,-1,-1],[1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,1]]}

CONES = {
    "111-": [1,2,3,4,6,7,8,9,10,11],
    "K3,3": [1,2,3,4,6,7,8,9,12],
    "K5":   [1,2,3,4,5,6,7,8,9,10],
}

def d_of_form(rays, t, k):
    Q = np.zeros((4,4))
    for ti, ri in zip(t, rays):
        Q += ti * np.array(R[ri], float)
    try:
        B = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        return 0.0
    try:
        return combigeo.find_optimal(B.tolist(), index=k).normalized
    except Exception:
        return 0.0

def optimize(cone, k, maxfev, seeds):
    rays = CONES[cone]
    m = len(rays)
    best_d, best_t = 0.0, None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        s0 = np.zeros(m) if seed == 0 else rng.normal(scale=0.35, size=m)
        history = {"best": 0.0}
        def obj(s):
            t = np.exp(s - s.mean())          # масштаб-инвариантность
            d = d_of_form(rays, t, k)
            if d > history["best"] + 1e-7:
                history["best"] = d
                print(f"  [{cone} k={k} seed={seed}] d={d:.7f}", flush=True)
            return -d
        r = minimize(obj, s0, method="Nelder-Mead",
                     options={"maxfev": maxfev, "xatol": 1e-5, "fatol": 1e-10})
        if -r.fun > best_d:
            best_d, best_t = -r.fun, np.exp(r.x - r.x.mean()).tolist()
    print(f"{cone} k={k}: max d = {best_d:.7f}", flush=True)
    return {"d": best_d, "t": best_t}


def main():
    out = {}
    # главная попытка: k=48 на генерическом конусе (старт с представителя, d=0.99941)
    out["111-_k48"] = optimize("111-", 48, maxfev=260, seeds=[0, 1, 2])
    # k=47 и 46 — на всякий случай (дешевле по числу подрешёток)
    out["111-_k47"] = optimize("111-", 47, maxfev=200, seeds=[0, 1])
    out["K3,3_k48"] = optimize("K3,3", 48, maxfev=200, seeds=[0, 1])
    # ширина интервала на 49: можно ли лучше sqrt(7/6) = 1.0801235?
    out["111-_k49"] = optimize("111-", 49, maxfev=200, seeds=[0, 1])
    out["K3,3_k49"] = optimize("K3,3", 49, maxfev=200, seeds=[0, 1])
    out["K5_k54"]   = optimize("K5", 54, maxfev=150, seeds=[0])
    json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/r2_cone.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
