"""Exhaustive width ladder in the plane: ``max d`` over all lattices and sublattices.

Two of the three ingredients are complete enumerations, which is what makes the
answer usable as a floor argument:

  * every plane lattice is, up to similarity, ``rows (1,0), (x,y)`` with ``y > 0``
    (the scale and rotation cancel in ``d = D_min/diam``);
  * every sublattice of index ``k`` is ``H`` in Hermite normal form,
    ``[[c,0],[b,a]]`` with ``a c = k``, ``0 <= b < c`` -- there are ``sigma(k)``
    of them and all are tried;
  * only the maximisation over ``(x, y)`` is numeric (multistart Nelder-Mead).

The rung that matters for :mod:`chromatic_research.campaigns.dim10_product` is
``d >= sqrt 7``, the width a plane spacer must reach to sit next to ``E8/2401``
in a product.  The search finds it first at ``k = 19`` (the Eisenstein
``5 + 2 omega``); ``k = 16, 17, 18`` fall short, and the volumetric Minkowski
screen already forbids ``k <= 15``.  So ``2401 * 19`` cannot be improved by
choosing a different plane block.

Usage::

    python -m chromatic_research.campaigns.ladder2d --max-index 24
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.optimize import minimize

import combigeo
from chromatic_research.core.lamination import enumerate_upto, unit_facets
from chromatic_research.campaigns.dim10_product import min_separation_wide
from chromatic_research.paths import results_path

SQRT7 = math.sqrt(7.0)


def hermite_forms(k: int) -> list[np.ndarray]:
    """All sublattices of index ``k`` of a rank-2 lattice, as row transforms."""
    out = []
    for c in range(1, k + 1):
        if k % c:
            continue
        a = k // c
        out.extend(np.array([[c, 0], [b, a]], dtype=float) for b in range(c))
    return out


def circumradius(facets) -> float:
    """Half the cell diameter: the farthest vertex of a plane Voronoi cell."""
    normals = np.array([f[0] for f in facets], dtype=float)
    offsets = np.array([f[1] for f in facets], dtype=float)
    best = 0.0
    for i in range(len(facets)):
        for j in range(i + 1, len(facets)):
            pair = np.array([normals[i], normals[j]])
            if abs(np.linalg.det(pair)) < 1e-9:
                continue
            vertex = np.linalg.solve(pair, np.array([offsets[i], offsets[j]]))
            if np.all(normals @ vertex <= offsets + 1e-9):
                best = max(best, float(np.linalg.norm(vertex)))
    return best


def width(x: float, y: float, form: np.ndarray) -> float:
    if y <= 1e-6:
        return -1.0
    basis = np.array([[1.0, 0.0], [x, y]])
    facets = unit_facets(basis)
    diam = 2.0 * circumradius(facets)
    if diam <= 0.0:
        return -1.0
    return min_separation_wide(form @ basis, diam, facets) / diam


def best_for_index(k: int, starts: int = 24, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed + k)
    best, argbest, form_best = -1.0, None, None
    for form in hermite_forms(k):
        for _ in range(starts):
            x0 = rng.uniform(-0.5, 0.5)
            y0 = rng.uniform(max(0.4, math.sqrt(max(1.0 - x0 * x0, 0.0))), 2.0)
            res = minimize(lambda p: -width(p[0], p[1], form), [x0, y0],
                           method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 2000})
            if -res.fun > best:
                best, argbest, form_best = -res.fun, tuple(res.x), form.copy()
    return dict(index=k, d=best, shape=list(argbest),
                hermite=form_best.astype(int).tolist(),
                reaches_sqrt7=bool(best >= SQRT7 - 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-index", type=int, default=3)
    parser.add_argument("--max-index", type=int, default=24)
    parser.add_argument("--starts", type=int, default=24)
    args = parser.parse_args()

    records, first = [], None
    print(f"{'k':>4} {'d_max':>12} {'1/d^2':>10}  {'>=sqrt7':>8}   shape / Hermite")
    for k in range(args.min_index, args.max_index + 1):
        rec = best_for_index(k, args.starts)
        records.append(rec)
        if rec["reaches_sqrt7"] and first is None:
            first = k
        print(f"{k:4d} {rec['d']:12.7f} {1/rec['d']**2:10.6f}  "
              f"{'yes' if rec['reaches_sqrt7'] else '':>8}   "
              f"({rec['shape'][0]:+.5f},{rec['shape'][1]:.5f}) {rec['hermite']}")

    print(f"\nfirst index reaching d >= sqrt7 = {SQRT7:.9f}:  k = {first}")
    path = results_path("ladder2d.json")
    path.write_text(json.dumps(dict(first_sqrt7=first, records=records), indent=2))
    print(f"written {path}")


if __name__ == "__main__":
    main()
