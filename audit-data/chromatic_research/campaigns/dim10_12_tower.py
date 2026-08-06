"""Dimensions 10-12: the ``E8`` lamination tower with the colouring ``Gamma = 3*Lambda``.

Two facts drive the construction.

**The real-multiplier rule.**  For ``Gamma = m*Lambda`` (index ``m^n``) the minimum
of ``D`` is attained at ``m*u`` with ``u`` a minimal vector: ``u`` is always
Voronoi-relevant, so the cell reaches exactly ``lambda1/2`` in that direction and

    dist(m u / 2, V0) = m lambda1 / 2 - lambda1 / 2,     D_min = (m - 1) lambda1.

For ``m = 3`` this is ``D_min = 2 lambda1`` and hence

    d = 2 / rho,        rho = diam / lambda1,

so the construction is admissible exactly when ``rho <= 2``.  (``m = 2`` would need
``rho <= 1``, impossible above dimension 1.)  The rule was checked against direct
measurement on twelve lattices and agrees to six decimals; note it is specific to
*real* multipliers -- for ``3+omega`` the minimum moves elsewhere and the
Eisenstein identity ``D^2 = (7/3) lambda1^2`` applies instead.

**The tower.**  Laminating ``E8`` keeps ``rho`` below 2 in every dimension.  Adding a
layer of height ``t`` over a deep hole ``c``,

    (P1)  R_up(n+1)^2 = R_up(n)^2 + t^2 / 4        (nearest layer, nearest point)
          lambda1 stays 1 as long as   R^2 + t^2 >= 1.

The two roles of the covering radius pull in opposite directions and must be kept
apart: the height has to come from a **lower** bound on ``R`` (otherwise the layer
vector is too short and ``lambda1`` silently drops below 1), while the diameter has
to be propagated from an **upper** bound.  Feeding the same estimate to both is
what makes an apparently rigorous recursion wrong.

With ``t^2 = 1 - R_low^2`` the recursion stays below the fixed point ``diam^2 = 4``
for every ``n``, so ``rho < 2`` and ``d > 1`` in all dimensions ``n >= 8``.

Only ``R_up`` is an inequality: ``lambda1`` is verified directly by
``shortest_vector`` and ``D_min`` by exhaustive enumeration of every ``3*Lambda``
vector of norm ``<= 2 diam`` (anything longer clears because ``D(v) >= |v| - diam``),
each one projected by two independent routines.

Usage::

    python -m chromatic_research.campaigns.dim10_12_tower --nmax 12
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import combigeo
from chromatic_research.core.lattices import E8
from chromatic_research.core.lamination import (
    deep_hole, enumerate_upto, unit_facets,
)
from chromatic_research.paths import results_path


def slsqp_distance(point: np.ndarray, normals: np.ndarray, offsets: np.ndarray) -> float:
    """Projection onto the cell by SLSQP -- independent of combigeo's Dykstra."""
    point = np.asarray(point, dtype=float)
    if np.all(normals @ point <= offsets + 1e-12):
        return 0.0
    constraint = {"type": "ineq", "fun": lambda y: offsets - normals @ y,
                  "jac": lambda y: -normals}
    result = minimize(lambda y: float((y - point) @ (y - point)), np.zeros_like(point),
                      jac=lambda y: 2 * (y - point), constraints=[constraint],
                      method="SLSQP", options={"maxiter": 500, "ftol": 1e-15})
    return float(np.linalg.norm(result.x - point))


def separation_of_triple_lattice(basis: np.ndarray, diameter: float) -> tuple[float, int, int]:
    """``min D(v)`` over ``3*Lambda``, every candidate checked by two projections."""
    facets = unit_facets(basis)
    normals = np.array([f[0] for f in facets], dtype=float)
    offsets = np.array([f[1] for f in facets], dtype=float)
    vectors = enumerate_upto(3.0 * np.asarray(basis, float), 2 * diameter)
    norms = np.linalg.norm(vectors, axis=1)
    worst, checked = math.inf, 0
    for index in np.argsort(norms):
        if norms[index] - diameter >= worst:
            break
        dykstra = 2 * float(
            combigeo.dist_to_halfspaces((vectors[index] / 2).tolist(), facets))
        slsqp = 2 * slsqp_distance(vectors[index] / 2, normals, offsets)
        checked += 1
        worst = min(worst, dykstra, slsqp)
    return worst, checked, int(len(vectors))


def build_tower(nmax: int = 12, *, n_dirs: int = 900, budget: float = 1200.0) -> list[dict]:
    start = time.time()
    basis = np.asarray(E8(), dtype=float)
    basis = basis / float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    radius_upper = math.sqrt(0.5)          # exact for E8 at lambda1 = 1
    radius_lower = radius_upper            # exact, so both roles coincide here
    records: list[dict] = []
    for n in range(9, nmax + 1):
        if time.time() - start > budget:
            print("  [budget] stopping the tower early", flush=True)
            break
        height = math.sqrt(max(1.0 - radius_lower**2, 1e-12))
        _, hole = deep_hole(basis, n_dirs=n_dirs // 2, seed=3)
        lifted = np.zeros((n, n))
        lifted[:n - 1, :n - 1] = basis
        lifted[n - 1, :n - 1] = hole
        lifted[n - 1, n - 1] = height
        basis = lifted
        radius_upper = math.sqrt(radius_upper**2 + height**2 / 4.0)     # (P1)
        diameter = 2 * radius_upper
        lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
        measured, _ = deep_hole(basis, n_dirs=n_dirs, seed=5)
        radius_lower = measured
        separation, checked, total = separation_of_triple_lattice(basis, diameter)
        record = {
            "n": n, "index": 3**n, "layer_height": height, "lambda1": lam1,
            "covering_radius_upper": radius_upper, "covering_radius_measured": measured,
            "diam_rigorous": diameter, "diam_measured": 2 * measured,
            "D_min": separation,
            "d_rigorous": separation / diameter,
            "d_measured": separation / (2 * measured),
            "n_vectors": total, "n_checked": checked,
            "valid_rigorous": bool(separation / diameter >= 1.0),
        }
        records.append(record)
        print(f"  n={n:2d} lam1={lam1:.9f} diam<={diameter:.9f} D_min={separation:.9f} "
              f"d>={record['d_rigorous']:.9f} index 3^{n}={3**n} "
              f"{'RIGOROUS VALID' if record['valid_rigorous'] else 'NOT VALID'} "
              f"[{checked}/{total} vectors, {time.time() - start:.0f}s]", flush=True)
    return records


def scan_intermediate_sublattices(
    basis: np.ndarray, diameter_lower: float, n: int, *, budget: float = 900.0
) -> dict:
    """Can any sublattice strictly containing ``3*Lambda`` beat index ``3^n``?

    Such sublattices are the subgroups ``H`` of ``Lambda/3Lambda = (Z/3)^n``, of index
    ``3^n/|H|``, and ``H`` is admissible only if every nonzero class ``c`` in it has
    ``min_{0 != v in c} D(v) >= diam``.  Every class has a representative of norm
    ``<= R(3 Lambda) = 3R``, so enumerating that ball meets all of them.

    ``diameter_lower`` must be a **lower** bound on the diameter: a class is then
    declared bad only when it holds a vector that is forbidden for the true
    diameter as well, so a zero survivor count is a sound negative result.
    """
    start = time.time()
    facets = unit_facets(basis)
    radius = 1.5 * diameter_lower          # = 3R, the covering radius of 3*Lambda
    vectors = enumerate_upto(np.asarray(basis, float), radius)
    coordinates = np.linalg.solve(np.asarray(basis, float).T, vectors.T).T
    keys = (np.mod(np.rint(coordinates).astype(np.int64), 3)
            @ (3 ** np.arange(n, dtype=np.int64)))
    total = 3**n
    bad = np.zeros(total, dtype=bool)
    bad[0] = True                                  # the zero class is admissible
    remaining, projections = total - 1, 0
    for index in np.argsort(np.linalg.norm(vectors, axis=1)):
        if time.time() - start > budget:
            print("  [budget] class scan stopped early", flush=True)
            break
        key = keys[index]
        if bad[key]:
            continue
        separation = 2 * float(
            combigeo.dist_to_halfspaces((vectors[index] / 2).tolist(), facets))
        projections += 1
        if separation < diameter_lower:
            bad[key] = True
            remaining -= 1
            if remaining == 0:
                break
        if projections % 10000 == 0:
            print(f"    [heartbeat] {projections} projections, {remaining} classes "
                  f"left, {time.time() - start:.0f}s", flush=True)
    survivors = int(total - bad.sum())
    print(f"  n={n}: {projections} projections, {survivors} of {total - 1} classes "
          f"could still be admissible", flush=True)
    return {"n": n, "classes": total - 1, "survivors": survivors,
            "projections": projections, "enumerated": int(len(vectors)),
            "diameter_lower": diameter_lower,
            "conclusion": ("3^n is optimal among sublattices containing 3*Lambda"
                           if survivors == 0 else "some class survives")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmax", type=int, default=12)
    parser.add_argument("--budget", type=float, default=1200.0)
    parser.add_argument("--scan-classes", type=int, default=0,
                        help="also test whether index 3^n can be beaten in this dimension")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    records = build_tower(args.nmax, budget=args.budget)
    out = args.output or results_path("dim10_12_tower.json")
    payload: dict | list = records
    if args.scan_classes:
        basis = np.asarray(E8(), dtype=float)
        basis = basis / float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
        radius_lower = math.sqrt(0.5)
        for n in range(9, args.scan_classes + 1):
            height = math.sqrt(max(1.0 - radius_lower**2, 1e-12))
            _, hole = deep_hole(basis, n_dirs=450, seed=3)
            lifted = np.zeros((n, n))
            lifted[:n - 1, :n - 1] = basis
            lifted[n - 1, :n - 1] = hole
            lifted[n - 1, n - 1] = height
            basis = lifted
            radius_lower, _ = deep_hole(basis, n_dirs=900, seed=5)
        payload = {"tower": records,
                   "class_scan": scan_intermediate_sublattices(
                       basis, 2 * radius_lower, args.scan_classes)}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"saved {out}", flush=True)
    return 0 if all(r["valid_rigorous"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
