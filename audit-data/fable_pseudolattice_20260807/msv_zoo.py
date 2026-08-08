"""Детерминированный зоопарк: именованные кристаллографические структуры
при канонической геометрии, лучшие Γ-разбиения по HNF, без оптимизации.

Быстрый интерпретируемый срез: где мульти-узловые структуры стоят
относительно решёточного фронтира ДО всякой подгонки параметров.
Запуск: python msv_zoo.py  → results/msv_zoo.json  (однопоточно, ~минуты)
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msv_core as mc

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LATTICE_FRONTIER = {14: 0.900965, 15: 1.026593, 16: 1.029728, 17: 1.064844,
                    18: 1.115838, 19: 1.139320, 20: 1.171811, 21: 1.262783}


def hex_basis(c_over_a: float) -> np.ndarray:
    B = np.array([[1.0, 0, 0],
                  [0.5, math.sqrt(3) / 2, 0],
                  [0, 0, c_over_a]])
    return B / abs(np.linalg.det(B)) ** (1 / 3)


def fcc_basis() -> np.ndarray:
    B = np.array([[1., 1, 0], [1, 0, 1], [0, 1, 1]])
    return B / abs(np.linalg.det(B)) ** (1 / 3)


STRUCTURES = {
    "HCP":     (hex_basis(math.sqrt(8 / 3)), [[0., 0, 0], [1 / 3, 1 / 3, 0.5]]),
    "diamond": (fcc_basis(), [[0., 0, 0], [0.25, 0.25, 0.25]]),
    "NaCl":    (fcc_basis(), [[0., 0, 0], [0.5, 0.5, 0.5]]),
    "AAB_hex": (hex_basis(1.8), [[0., 0, 0], [1 / 3, 2 / 3, 0.5]]),
}


def splits_for(k: int, s: int) -> list[tuple[int, ...]]:
    """Невозрастающие разбиения k на s частей ≥ 2."""
    out = []
    def rec(rem, parts, cap):
        if len(parts) == s:
            if rem == 0:
                out.append(tuple(parts))
            return
        lo = 2
        for v in range(min(cap, rem - lo * (s - len(parts) - 1)), lo - 1, -1):
            rec(rem - v, parts + [v], v)
    rec(k, [], k)
    return out


def run():
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    out = {"frontier_lattice": LATTICE_FRONTIER, "structures": {}}
    for name, (B, frac) in STRUCTURES.items():
        frac = np.array(frac, float)
        s = len(frac)
        rows = {}
        for k in range(14, 22):
            best = None
            for spl in splits_for(k, s):
                hn = {j: mc.prepare_hnfs(j) for j in set(spl)}
                r = mc.evaluate(B, frac, list(spl), hn, d_cap=1.45, fw_iters=200)
                if r.get("ok") and (best is None or r["d"] > best["d"]):
                    best = {"d": r["d"], "split": list(spl),
                            "gaps": r["gaps"], "diams": r["diams"]}
            rows[k] = best
            ref = LATTICE_FRONTIER[k]
            print(f"[{time.time()-t0:6.1f}s] {name:8s} k={k}: "
                  f"d={best['d'] if best else float('nan'):.6f} "
                  f"(решётка {ref:.6f}) split={best['split'] if best else None}",
                  flush=True)
        out["structures"][name] = rows
    json.dump(out, open(os.path.join(RESULTS, "msv_zoo.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"готово за {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run()
