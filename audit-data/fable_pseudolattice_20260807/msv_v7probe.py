"""Проба V₇: значение симметричного разбиения (7,7) — потолок s=2-класса при
k = 14 (лемма чётной доминации) — на лучших найденных структурах (P, t).

Берёт чемпионов из msv_E2_k15_polished.json (7+8) и msv_E3_k16(_polished).json
(8+8), оценивает на их (P,t) разбиение (7,7) и дожимает NM.
Итог — потолок МСВ s=2 при k=14 против решёточного 0.9012.
Запуск: python msv_v7probe.py  → results/msv_v7_probe.json
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


def collect_sources():
    out = []
    p = os.path.join(RESULTS, "msv_E2_k15_polished.json")
    if os.path.exists(p):
        for r in json.load(open(p))["polished"]:
            if len(r["j_list"]) == 2:
                out.append(("E2pol:" + r["tag"], r["x"]))
    p = os.path.join(RESULTS, "msv_E3_k16_polished.json")
    if os.path.exists(p):
        for r in json.load(open(p))["polished"]:
            out.append(("E3pol:" + r["tag"], r["x"]))
    else:
        for r in json.load(open(os.path.join(RESULTS, "msv_E3_k16.json")))["top"]:
            out.append(("E3:" + r["tag"], r["x"]))
    return out


def run():
    t0 = time.time()
    rows = []
    f = lambda x: -mcamp.objective(x, [7, 7], mcamp.D_CAP_SEARCH,
                                   mcamp.FW_ITERS_SEARCH)
    for tag, x in collect_sources():
        x = np.array(x, float)
        d0 = -f(x)
        res = minimize(f, x, method="Nelder-Mead",
                       options={"maxfev": 1200, "xatol": 1e-8,
                                "fatol": 1e-12, "adaptive": True})
        d1 = -float(res.fun)
        B, frac = mcamp.unpack_x(res.x, 2)
        rr = mc.evaluate(B, frac, [7, 7], {7: mc.prepare_hnfs(7)},
                         d_cap=1.6, fw_iters=800)
        rows.append({"source": tag, "d_at_source": d0, "d_polished": d1,
                     "d_final": rr.get("d"),
                     "x": [float(v) for v in res.x]})
        print(f"  {tag}: (7,7) на структуре = {d0:.6f} → дожим {d1:.6f} "
              f"(финал {rr.get('d'):.6f})", flush=True)
    best = max(rows, key=lambda r: r["d_final"] or 0.0)
    out = {"lemma": "even-domination: любое s=2 разбиение (j1,j2) ≤ (min,min); "
                    "V7 — потолок s=2 при k=14",
           "lattice_frontier_k14": 0.901027,
           "V7_best": best["d_final"], "rows": rows,
           "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(RESULTS, "msv_v7_probe.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"V7 = {best['d_final']:.6f} (решётка k=14: 0.901027) → "
          f"{'ПРОБОЙ' if best['d_final'] > 0.9012 else 'ниже решётки'}",
          flush=True)


if __name__ == "__main__":
    run()
