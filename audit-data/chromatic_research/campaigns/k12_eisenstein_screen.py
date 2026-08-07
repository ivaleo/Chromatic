"""Is ``7^6 = 117649`` reachable in ``R^12``?  No: the planar floor is attained.

The planar theorem only bounds ``D((3+omega)Lambda) >= sqrt(7/3) lambda1`` from
below, and the bound is tight exactly when the deep hole of the ``A2`` plane
sublattice ``Z[omega] w`` still lies inside ``V0(Lambda)``.  If some lattice cut
that vertex off, ``D`` would exceed the floor and ``K12`` (which misses the
Eisenstein threshold by only 6.9 %: ``rho = sqrt(8/3) = 1.6330`` against
``sqrt(7/3) = 1.5275``) would deliver ``chi(R^12) <= 7^6`` instead of the tower's
``3^12 = 531441``.

Only vectors shorter than ``2 |p| = 2 lambda1/sqrt3 = 1.1547 lambda1`` can cut
the vertex ``p`` off, so for ``K12`` (norms 4, 6, 8, ...) the minimal vectors are
the only candidates -- and taking halfspaces from *all* of them is both a
superset of ``V0`` (hence a rigorous lower bound on ``D``) and, here, decisive.

Usage::

    python -m chromatic_research.campaigns.k12_eisenstein_screen
"""

from __future__ import annotations

import json
import math

import numpy as np

import combigeo
from chromatic_research.core.k12 import build_k12
from chromatic_research.core.lamination import enumerate_upto
from chromatic_research.core.layer_lamination import eisenstein_map
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)


def main() -> None:
    _, basis = build_k12()
    basis = np.asarray(basis, float)
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    minimal = enumerate_upto(basis, lam1 * (1 + 1e-9))
    facets = [((v / n).tolist(), n / 2.0)
              for v, n in zip(minimal, np.linalg.norm(minimal, axis=1))]

    radius = math.sqrt(8.0 / 3.0)                 # R(K12), exact (SPLAG)
    diam = 2.0 * radius
    sub = basis @ eisenstein_map(3 + OMEGA, 6).T
    lam1_sub = float(np.linalg.norm(combigeo.shortest_vector(sub.tolist())))
    vectors = enumerate_upto(sub, diam + lam1_sub)
    norms = np.linalg.norm(vectors, axis=1)
    best = math.inf
    for i in np.argsort(norms):
        if norms[i] - diam >= best:
            break
        best = min(best, 2.0 * float(
            combigeo.dist_to_halfspaces((vectors[i] / 2).tolist(), facets)))

    floor = math.sqrt(7 / 3) * lam1
    record = dict(
        lambda1=lam1, kissing=len(minimal), covering_radius=radius, diam=diam,
        rho=diam / lam1, eisenstein_threshold=math.sqrt(7 / 3),
        lambda1_sublattice=lam1_sub, scanned=len(vectors),
        D_min_lower_bound=best, planar_floor=floor, excess=best - floor,
        admissible=bool(best >= diam), index=7 ** 6, tower_index=3 ** 12,
    )
    for key, value in record.items():
        print(f"  {key:>22} = {value}")
    assert record["kissing"] == 756, record
    assert abs(record["excess"]) < 1e-9, "planar floor should be attained"
    assert not record["admissible"]
    print("\n  planar floor attained exactly -> 7^6 = 117649 is NOT reachable on K12;"
          f"\n  the Eisenstein route in R^12 needs rho <= {math.sqrt(7/3):.6f}, "
          f"K12 has {record['rho']:.6f}")
    path = results_path("k12_eisenstein_screen.json")
    path.write_text(json.dumps(record, indent=2))
    print(f"  written {path}")


if __name__ == "__main__":
    main()
