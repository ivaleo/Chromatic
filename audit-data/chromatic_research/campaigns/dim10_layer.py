"""Dimension 10 by laminating ``E8/2401`` with one Eisenstein layer lattice.

The rigorous product ``E8/2401 (+) A2/19`` gives ``45619``
(:mod:`chromatic_research.campaigns.dim10_product`).  Replacing the orthogonal
sum by a *laminated* one -- the same two blocks, but with the ``A2`` layers
offset inside the ``E8`` cell -- lets the layer separation borrow from the
offsets instead of from the layer scale, and the index drops.

What the offsets cannot change is the (P1) diameter bound, which depends only on
the layer scale ``a``; so maximising ``d`` at fixed ``a`` is the same as
maximising ``D_min``, and the search is CMA over the eight real coordinates of
one complex offset vector (the construction is ``Z[omega]``-equivariant, so that
vector determines both layer generators).  This is exactly the shape of search
that produced ``7203`` in dimension 9.

Status of the output is deliberately two-tiered, as in the dimension-9 campaign:

  * ``d_p1``      -- rigorous, using ``diam <= 2 sqrt(3/2 + a^2/3)``;
  * ``d_measured``-- candidate, using the covering radius from vertex ascent,
    which approaches the truth from below and therefore *over*states ``d``.

Beyond ``a = sqrt3/2`` the (P1) bound exceeds ``sqrt 7`` and only the measured
column is meaningful -- which is where the interesting indices live.

Usage::

    python -m chromatic_research.campaigns.dim10_layer --alpha 4+2w --scales 0.95 1.05
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import combigeo
from chromatic_research.core.layer_lamination import (
    eisenstein_layer, eisenstein_map, layer_budget,
)
from chromatic_research.campaigns.planar_theorem_check import e8_theta_basis
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)
SQRT7 = math.sqrt(7.0)
R8SQ = 1.5                              # R(E8)^2 at lambda1^2 = 3

ALPHAS = {"3": 3 + 0j, "4": 4 + 0j, "4+1w": 4 + OMEGA, "4+2w": 4 + 2 * OMEGA,
          "5": 5 + 0j, "5+2w": 5 + 2 * OMEGA, "5+3w": 5 + 3 * OMEGA}


def base_pair() -> tuple[np.ndarray, np.ndarray]:
    e8 = e8_theta_basis()
    kernel = np.rint(np.linalg.solve(
        e8.T, (e8 @ eisenstein_map(3 + OMEGA, 4).T).T).T).astype(int)
    return e8, kernel


def optimise(scale: float, alpha: complex, budget: int, seeds: int) -> dict:
    import cma
    e8, kernel = base_pair()
    p1 = 2.0 * math.sqrt(R8SQ + scale * scale / 3.0)
    yard = max(p1, SQRT7)               # fixed normaliser so seeds are comparable

    def negative(x):
        lam = eisenstein_layer(e8, kernel, np.asarray(x), scale, alpha, R8SQ)
        return -lam.separation(yard, facet_cap=400) / yard

    best = (-1.0, None)
    for seed in range(seeds):
        rng = np.random.default_rng(4242 + seed + int(scale * 997))
        strategy = cma.CMAEvolutionStrategy(
            list(rng.uniform(-0.6, 0.6, 8)), 0.35,
            {"seed": 4242 + seed, "maxfevals": budget, "verbose": -9, "popsize": 12})
        strategy.optimize(negative)
        if -strategy.result.fbest > best[0]:
            best = (-strategy.result.fbest, np.asarray(strategy.result.xbest))

    lam = eisenstein_layer(e8, kernel, best[1], scale, alpha, R8SQ)
    separation = lam.separation(yard)
    measured = lam.measured_diameter(n_dirs=400)
    whole, _ = lam.lattices()
    return dict(
        scale=scale, index=lam.index, N=lam.index // 2401,
        D_min=separation, diam_p1=p1, diam_measured=measured,
        d_p1=separation / p1 if p1 <= SQRT7 else None,
        d_measured=min(separation, SQRT7) / measured,
        lambda1=float(np.linalg.norm(combigeo.shortest_vector(whole.tolist()))),
        offset=[float(v) for v in best[1]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", default="4+2w", choices=sorted(ALPHAS))
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.86, 0.95, 1.05, 1.15])
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    alpha = ALPHAS[args.alpha]
    print(f"base E8/2401: diam = sqrt6, d = sqrt(7/6); layer budget R_L <= "
          f"{layer_budget(math.sqrt(6), math.sqrt(7 / 6)):.6f}")
    print(f"layer multiplier {args.alpha}, N = {round(abs(alpha) ** 2)}, "
          f"index = {2401 * round(abs(alpha) ** 2)}   (3^10 = {3 ** 10})")
    print(f"\n{'a':>6} {'D_min':>12} {'diam(P1)':>10} {'d(P1)':>9} "
          f"{'diam meas':>10} {'d meas':>9} {'lam1':>7}")

    records = []
    for scale in args.scales:
        start = time.time()
        rec = optimise(scale, alpha, args.budget, args.seeds)
        records.append(rec)
        p1 = f"{rec['d_p1']:9.6f}" if rec["d_p1"] else "    (>s7)"
        print(f"{scale:6.3f} {rec['D_min']:12.7f} {rec['diam_p1']:10.6f} {p1} "
              f"{rec['diam_measured']:10.6f} {rec['d_measured']:9.6f} "
              f"{rec['lambda1']:7.4f}  [{time.time()-start:.0f}s]", flush=True)

    path = results_path(f"dim10_layer_{args.alpha}.json")
    path.write_text(json.dumps(dict(alpha=args.alpha, records=records), indent=2))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
