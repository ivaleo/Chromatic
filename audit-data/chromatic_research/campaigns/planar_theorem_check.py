"""Numeric verification of two new one-line theorems found in the 2026-08-07 audit.

**Planar theorem.**  For every Eisenstein lattice ``Lambda`` (a ``Z[omega]``-module)
and ``alpha = 3 + omega``:

    D(alpha w) >= sqrt(7/3) |w|   for every nonzero w in Lambda,   hence
    D(alpha Lambda) >= sqrt(7/3) lambda1(Lambda).

Proof sketch: the six equal-norm vectors ``+-w, +-omega w, +-(1+omega) w`` cut,
inside the plane ``span(w, omega w)``, the hexagonal ``A2`` Voronoi cell of the
sublattice ``Z[omega] w``; their halfspace normals lie *in* that plane, so the
orthogonal projection of ``V0`` stays inside the hexagon, and the planar
distance ``dist(alpha w / 2, hexagon) = sqrt(7/12) |w|`` is exact 2-dimensional
geometry.  This closes the ``>=`` half of the Eisenstein identity
(open question 7 of the paper) and upgrades ``D_min = sqrt(7)`` of
``E8/2401``, ``E6*/343`` and the lamination floors from a numeric check to a
theorem.  Here we verify the *per-vector* inequality on ``A2``, ``E6*`` and
``E8`` (theta-construction over the tetracode), including the equality case at
minimal vectors.

**Inradius lemma.**  ``B(0, lambda1/2) ⊆ V0`` gives ``D(v) <= |v| - lambda1``.
Consequence: for ``Gamma = 3 Lambda`` with ``rho < 2``, no sublattice strictly
between ``3 Lambda`` and ``Lambda`` is admissible -- every nonzero class of
``Lambda / 3 Lambda`` has a representative of norm ``<= 3R < diam + lambda1``,
which the lemma kills.  The exhaustive class scan of the paper (n = 10, zero
survivors out of 59048) becomes a corollary.  Here we verify the lemma on
random vectors and the strict inequality ``3R < diam + lambda1`` for the
tower lattices and ``K12``.

Usage::

    python -m chromatic_research.campaigns.planar_theorem_check
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import combigeo
from chromatic_research.core.k12 import build_k12
from chromatic_research.core.lamination import deep_hole, enumerate_upto, unit_facets
from chromatic_research.campaigns.dim7_laminate_e6s import e6star_geometry
from chromatic_research.campaigns.rho_scan import tower_bases
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)
THETA = OMEGA - OMEGA.conjugate()
SQRT73 = math.sqrt(7.0 / 3.0)


def _realify(vectors) -> np.ndarray:
    out = []
    for vector in vectors:
        row = []
        for z in vector:
            row += [z.real, z.imag]
        out.append(row)
    return np.asarray(out, dtype=float)


def a2_basis() -> np.ndarray:
    return _realify([(1,), (OMEGA,)])


def e8_theta_basis() -> np.ndarray:
    """``E8`` as the theta-construction over the tetracode ``[4,2,3]_3``:
    ``{x in Z[omega]^4 : x mod theta in C}``; ``lambda1^2 = 3``."""
    rows_c = [
        (1, 1, 1, 0),
        (0, 1, -1, 1),
        (THETA, 0, 0, 0),
        (0, THETA, 0, 0),
    ]
    generators = []
    for u in rows_c:
        generators.append(u)
        generators.append(tuple(OMEGA * z for z in u))
    basis = _realify(generators)
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    kiss = len(enumerate_upto(basis, lam1 + 1e-9))
    assert abs(lam1**2 - 3.0) < 1e-9 and kiss == 240, (lam1**2, kiss)
    return basis


def alpha_map(n_complex: int) -> np.ndarray:
    alpha = 3 + OMEGA
    block = np.array([[alpha.real, -alpha.imag], [alpha.imag, alpha.real]])
    return np.kron(np.eye(n_complex), block)


def check_planar(name: str, basis: np.ndarray, *, bound_factor: float = 2.2) -> dict:
    start = time.time()
    basis = np.asarray(basis, dtype=float)
    n = basis.shape[0]
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    facets = unit_facets(basis)
    a_map = alpha_map(n // 2)
    vectors = enumerate_upto(basis, bound_factor * lam1)
    worst_margin, equality_at_min, checked = math.inf, None, 0
    for w in vectors:
        v = w @ a_map.T
        separation = 2.0 * float(combigeo.dist_to_halfspaces((v / 2).tolist(), facets))
        floor = SQRT73 * float(np.linalg.norm(w))
        worst_margin = min(worst_margin, separation - floor)
        checked += 1
        if abs(np.linalg.norm(w) - lam1) < 1e-9 and equality_at_min is None:
            equality_at_min = separation - SQRT73 * lam1
    record = {
        "lattice": name, "n": n, "lambda1": lam1, "checked": checked,
        "worst_margin_D_minus_floor": worst_margin,
        "equality_gap_at_minimal": equality_at_min,
        "holds": bool(worst_margin > -1e-7),
        "seconds": round(time.time() - start, 1),
    }
    print(f"planar {name:5s}: {checked} vectors, worst D - sqrt(7/3)|w| = "
          f"{worst_margin:.3e}, equality gap at minimal = {equality_at_min:.3e} "
          f"[{record['seconds']}s]", flush=True)
    return record


def check_inradius(name: str, basis: np.ndarray, *, samples: int = 400) -> dict:
    basis = np.asarray(basis, dtype=float)
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    facets = unit_facets(basis)
    radius, _ = deep_hole(basis, n_dirs=900, seed=5)
    vectors = enumerate_upto(basis, 3.2 * radius)
    rng = np.random.default_rng(3)
    take = rng.choice(len(vectors), size=min(samples, len(vectors)), replace=False)
    worst = -math.inf
    for index in take:
        v = vectors[index]
        separation = 2.0 * float(combigeo.dist_to_halfspaces((v / 2).tolist(), facets))
        worst = max(worst, separation - (np.linalg.norm(v) - lam1))
    closure = 3 * radius - (2 * radius + lam1)     # < 0 <=> subgroup scan provably empty
    record = {
        "lattice": name, "n": int(basis.shape[0]), "lambda1": lam1,
        "covering_radius_measured": radius,
        "worst_D_minus_bound": worst,
        "lemma_holds": bool(worst <= 1e-7),
        "three_R_minus_diam_minus_lam1": closure,
        "subgroup_scan_provably_empty": bool(closure < 0),
    }
    print(f"inradius {name:6s}: worst D - (|v|-lam1) = {worst:.3e} (must be <= 0); "
          f"3R - diam - lam1 = {closure:.4f} (must be < 0)", flush=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    e6s, _, _ = e6star_geometry()
    planar = [
        check_planar("A2", a2_basis()),
        check_planar("E6*", e6s),
        check_planar("E8", e8_theta_basis()),
    ]

    towers = tower_bases(11)
    _, k12 = build_k12()
    inradius = [
        check_inradius("Lam9", towers["Lam9"]),
        check_inradius("Lam10", towers["Lam10"]),
        check_inradius("Lam11", towers["Lam11"]),
        check_inradius("K12", k12),
    ]

    payload = {"planar": planar, "inradius": inradius,
               "all_hold": bool(all(r["holds"] for r in planar)
                                and all(r["lemma_holds"] for r in inradius))}
    out = args.output or results_path("planar_inradius_checks.json")
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"saved {out}; ALL {'OK' if payload['all_hold'] else 'FAILED'}", flush=True)
    return 0 if payload["all_hold"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
