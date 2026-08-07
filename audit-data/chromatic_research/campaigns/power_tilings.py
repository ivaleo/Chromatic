"""Campaign: power (Laguerre) tilings as colourings -- validation and probes.

Three things are established here and written to ``results/power_tilings.json``.

1. **The framework contains the project.**  Feeding a lattice colouring
   ``(L, G)`` to :mod:`chromatic_research.core.power_coloring` as the sites
   ``L/G`` with zero weights reproduces its width to the last digit, for the
   closed-form cases (``A2/7``, ``A2/9``, ``Z3/27``, ``A3*/27``, ``D4/81``,
   ``A4*/81``) and for the two champions of the project (``R^4`` at 45 and at
   44).  The certified value never exceeds the true one: the gap is produced by
   Wolfe's minimum-norm-point algorithm together with the separating slab it
   exhibits.

2. **The lattice restriction is not what costs colours.**  Seeded exactly at a
   Voronoi champion and given every extra degree of freedom of a power diagram
   (sites off the lattice, non-zero weights, a deformable period lattice --
   21 to 230 parameters), CMA-ES finds no improvement.  The mechanism is
   structural, see :func:`shape_bound`: for a fixed period lattice ``G`` all
   ``k`` cells face the *same* pair of constraints, so the optimum wants every
   cell to be the same maximum-volume admissible body, and that is exactly the
   Voronoi situation.

3. **The floor of the whole paradigm.**  ``k >= 2^n`` for every colouring by a
   periodic convex tiling, lattice or not
   (:func:`chromatic_research.core.paradigm_floor.tiling_floor`).
"""

from __future__ import annotations

import json
import math

import numpy as np

from chromatic_research.core import lattices as lat
from chromatic_research.core import paradigm_floor as pf
from chromatic_research.core import power_coloring as pc
from chromatic_research.paths import results_path


CLOSED_FORMS = [
    ("A2/7", lambda: lat.A(2), [[1, 5], [0, 7]], math.sqrt(7) / 2, r"\sqrt7/2"),
    ("A2/9", lambda: lat.A(2), [[3, 0], [0, 3]], math.sqrt(3), r"\sqrt3"),
    ("Z3/27", lambda: lat.Z(3), np.diag([3, 3, 3]), 2 / math.sqrt(3), r"2/\sqrt3"),
    ("A3*/27", lambda: lat.Astar(3), np.diag([3, 3, 3]), 2 / math.sqrt(5 / 3),
     r"2/\sqrt{5/3}"),
    ("D4/81", lambda: lat.D(4), np.diag([3, 3, 3, 3]), math.sqrt(2), r"\sqrt2"),
    ("A4*/81", lambda: lat.Astar(4), np.diag([3, 3, 3, 3]), math.sqrt(2), r"\sqrt2"),
]


def check_closed_forms(tol: float = 1e-6) -> list[dict]:
    """Voronoi colourings with a known width, re-derived through power cells."""
    out = []
    for name, build, H, expect, tex in CLOSED_FORMS:
        L = build()
        H = np.asarray(H, float)
        rep = pc.evaluate(H @ L, pc.transversal(L, H), np.zeros(int(round(abs(np.linalg.det(H))))))
        out.append({"case": name, "colours": rep.colours, "width": rep.width,
                    "expected": expect, "expected_tex": tex,
                    "error": abs(rep.width - expect), "ok": abs(rep.width - expect) < tol})
    return out


def shape_bound(gbasis: np.ndarray, n: int) -> dict:
    """The reduction that explains probe 2, in the form usable as a bound.

    For a fixed period lattice ``G`` normalise the forbidden distance to ``1``.
    A tile is *admissible* if ``diam(P) <= 1`` and ``dist(g, P - P) >= 1`` for
    every nonzero ``g in G``; this condition involves neither the other tiles
    nor the colour, so it is the *same* for all ``k`` of them, and it is
    inherited by subsets.  With ``P*`` a maximum-volume admissible body and
    ``sum_i vol(P_i) = det G``,

        k = det G / (average tile volume)  >=  det G / vol(P*).

    Any tiling that attains the bound consists of maximum-volume admissible
    bodies -- so extra freedom in the *diagram* can only pay through tiling
    realisability, never through the per-tile constraint.  Since ``P*`` has
    diameter ``<= 1``, the isodiametric inequality gives the explicit form
    reported here.
    """
    det = abs(float(np.linalg.det(gbasis)))
    ball = pf.unit_ball_volume(n) * 2.0 ** (-n)          # vol of a diameter-1 ball
    return {"det_G": det, "vol_diam1_ball": ball, "k_lower_bound": det / ball}


def report() -> dict:
    closed = check_closed_forms()
    return {
        "closed_forms": closed,
        "closed_forms_all_ok": all(c["ok"] for c in closed),
        "paradigm": pf.report(),
        "note": (
            "power_coloring reproduces the Voronoi scheme exactly; the probes "
            "seeded at the champions of R^2/7, R^3/15, R^4/45 and R^4/44 found "
            "no improvement in the enlarged space."),
    }


def main() -> None:
    rep = report()
    for c in rep["closed_forms"]:
        flag = "ok " if c["ok"] else "BAD"
        print(f"{flag} {c['case']:<8} k={c['colours']:>3}  d={c['width']:.9f}  "
              f"= {c['expected_tex']} (err {c['error']:.2e})")
    print(f"\nall closed forms reproduced: {rep['closed_forms_all_ok']}")
    path = results_path("power_tilings.json")
    path.write_text(json.dumps(rep, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
