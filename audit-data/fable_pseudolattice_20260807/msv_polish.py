"""Дополировка топ-конфигураций кампаний МСВ увеличенным бюджетом NM.

Однопоточно (чтобы не мешать параллельным пулам). Берёт топ-M стартов из
results/msv_{exp}.json, дожимает NM (maxfev 1500, xatol 1e-8) + 3 джиттера,
пересдаёт финальным оценщиком. Гард: стеновой лимит на конфигурацию.
Запуск: python msv_polish.py msv_E2_k15.json [M]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msv_campaign as mcamp
import msv_core as mc

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
WALL_PER_CONFIG = 420.0          # сек на конфигурацию (NM + джиттеры)
MAXFEV = 1500


def polish(fname: str, top_m: int = 6, seed: int = 7):
    data = json.load(open(os.path.join(RESULTS, fname)))
    starts = sorted(data["all_starts"], key=lambda t: -t["d"])[:top_m]
    rng = np.random.default_rng(seed)
    out = []
    t00 = time.time()
    for r in starts:
        t0 = time.time()
        jl = r["j_list"]
        f = lambda x: -mcamp.objective(x, jl, mcamp.D_CAP_SEARCH,
                                       mcamp.FW_ITERS_SEARCH)
        best_d, best_x = r["d"], np.array(r["x"])
        xs = [best_x] + [best_x * (1 + rng.normal(scale=s, size=len(best_x)))
                         for s in (0.02, 0.06, 0.15)]
        for xi, x0 in enumerate(xs):
            if time.time() - t0 > WALL_PER_CONFIG:
                print(f"  {r['tag']}: стеновой лимит, джиттеры прерваны", flush=True)
                break
            res = minimize(f, x0, method="Nelder-Mead",
                           options={"maxfev": MAXFEV, "xatol": 1e-8,
                                    "fatol": 1e-12, "adaptive": True})
            if -res.fun > best_d:
                best_d, best_x = -float(res.fun), res.x.copy()
        B, frac = mcamp.unpack_x(best_x, len(jl))
        rr = mc.evaluate(B, frac, jl,
                         mcamp.hnfs_for(tuple(sorted(set(jl)))),
                         d_cap=mcamp.D_CAP_FINAL, fw_iters=mcamp.FW_ITERS_FINAL)
        out.append({"tag": r["tag"], "j_list": jl, "d_before": r["d"],
                    "d_polished": best_d, "d_final": rr.get("d"),
                    "x": [float(v) for v in best_x],
                    "wall_s": round(time.time() - t0, 1)})
        print(f"  {r['tag']} {jl}: {r['d']:.6f} → {best_d:.6f} "
              f"(финал {rr.get('d'):.6f}, {out[-1]['wall_s']}s)", flush=True)
    res_path = os.path.join(RESULTS, fname.replace(".json", "_polished.json"))
    json.dump({"source": fname, "maxfev": MAXFEV, "seed": seed,
               "wall_s": round(time.time() - t00, 1), "polished": out},
              open(res_path, "w"), indent=1, ensure_ascii=False)
    print(f"→ {res_path}", flush=True)


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "msv_E2_k15.json"
    top_m = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    polish(fname, top_m)
