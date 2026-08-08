"""Кампания 4D: кластерный зонд вокруг чемпиона 45.

Цели:
  k=45 — расширить интервал 1.016339 кластерными классами (N=90);
  k=44 — пробить решёточную стену 45 (решёточный фронтир 44 -> 0.990).

Запуск: .venv/bin/python audit-data/fable_pseudolattice_20260807/campaign4d.py [--budget 900]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import combigeo
import cluster_coloring as cc
from campaign3d import RESULTS, run_config, runtime_info

HERE = Path(__file__).parent


def champion45():
    d = json.load(open(HERE.parent / "results" / "n6_k45_rational.json"))
    Q = np.array([[float(Fraction(x)) for x in row] for row in d["Q_fractions"]])
    B = np.linalg.cholesky(Q)
    r = combigeo.find_optimal(B.tolist(), index=45, threads=1)
    assert abs(r.normalized - 1.0163393146674755) < 1e-9
    return B, np.array(r.best.transition, dtype=np.int64)


def best_period_refinement(B: np.ndarray, T_gamma: np.ndarray, m: int) -> np.ndarray:
    """G = S @ Gamma с S — HNF индекса m, максимизирующая lambda_1(G)."""
    gamma_basis = T_gamma.astype(float) @ B
    best_l1, best_T = -1.0, None
    for S in combigeo.sublattices(B.shape[0], m):
        S = np.array(S, dtype=np.int64)
        sv = combigeo.shortest_vector((S.astype(float) @ gamma_basis).tolist())
        l1 = float(np.linalg.norm(sv))
        if l1 > best_l1:
            best_l1, best_T = l1, S @ T_gamma
    return best_T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--cpsat-timeout", type=float, default=20.0)
    ap.add_argument("--cutoff-mult", type=float, default=2.6)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    B, T45 = champion45()
    out = {"runtime": runtime_info(),
           "champion": {"d": 1.0163393146674755, "transition": T45.tolist()},
           "configs": []}
    t0 = time.time()
    H90 = best_period_refinement(B, T45, 2)
    for k in (45, 44):
        out["configs"].append(run_config(
            f"champ45x2/k{k}", B, H90, k,
            budget_s=args.budget, cpsat_timeout=args.cpsat_timeout,
            cutoff_mult=args.cutoff_mult, a_cap_mult=1.8))
        json.dump(out, open(RESULTS / "probe4d.json", "w"), indent=1)
    print(f"-> {RESULTS / 'probe4d.json'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
