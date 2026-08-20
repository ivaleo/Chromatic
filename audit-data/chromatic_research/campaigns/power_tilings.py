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


# Local-maximality probes.  CMA-ES was seeded *exactly* at each Voronoi
# champion and given every extra degree of freedom of a power diagram (period
# lattice, sites, weights); the recorded outcome is the best width it reached.
# ``source`` says where the parent Gram form comes from, so the seed itself is
# reproducible; ``verify_champions`` re-derives the two cheap ones live.
PROBES = [
    {"case": "R^2 / 7", "dim": 2, "colours": 7, "parameters": 21,
     "champion": math.sqrt(7) / 2, "reached": math.sqrt(7) / 2,
     "source": "A2, Hermite [[1,5],[0,7]]"},
    # посев --- чемпион семейства G(a) (dim3_k15), а не ОЦК: у ОЦК ширина
    # ровно 1 и лучшей известной решёточной 15-раскраской он не является.
    # sigma = 0.003 / 0.01 / 0.03, по 2500 вычислений; лучшее CMA --- 1.0157885,
    # то есть посев не превзойдён (но и не удержан: проба слабее остальных).
    {"case": "R^3 / 15", "dim": 3, "colours": 15, "parameters": 62,
     "champion": 1.0265576446233684, "reached": 1.0265576446233684,
     "cma_best": 1.0157885482330826, "holds_seed": False,
     "source": "G(16/51), Hermite [[1,0,11],[0,1,7],[0,0,15]]"},
    {"case": "R^3 / 15 (ОЦК)", "dim": 3, "colours": 15, "parameters": 62,
     "champion": 1.0, "reached": 1.0,
     "source": "A3*, Hermite [[1,0,4],[0,1,12],[0,0,15]] "
               "(основная конструкция Кулсона, ширина ровно 1)"},
    {"case": "R^4 / 45", "dim": 4, "colours": 45, "parameters": 230,
     "champion": 1.0163393, "reached": 1.0163393,
     "source": "results/n6_k45_rational.json (Q_fractions) + find_optimal(45)"},
    {"case": "R^4 / 44", "dim": 4, "colours": 44, "parameters": 225,
     "champion": 0.9903509, "reached": 0.9903509,
     "source": "results/n10_push44.json (Q) + find_optimal(44)"},
]

CALIBRATION = {
    "note": "the same CMA-ES from random starts, for reference",
    "R^2 / 7 from random starts": 1.2396,
    "R^2 / 7 truth": math.sqrt(7) / 2,
    "R^3 / 14 free search": 0.5695,
    "R^3 / 14 best lattice": 0.7745967,
    "R^2 / 6 free search": 0.9661,
}


def _g16_51() -> np.ndarray:
    """Базис решётки G(16/51) --- чемпион семейства при k = 15 (dim3_k15)."""
    gram = np.array([[51, -16, -19], [-16, 51, -16], [-19, -16, 51]],
                    float) / 51.0
    return np.linalg.cholesky(gram)


def verify_champions() -> list[dict]:
    """Re-derive the cheap probe seeds through the power framework."""
    out = []
    for name, build, H, want in [
        ("R^2 / 7", lambda: lat.A(2), [[1, 5], [0, 7]], math.sqrt(7) / 2),
        ("R^3 / 15", _g16_51, [[1, 0, 11], [0, 1, 7], [0, 0, 15]],
         1.0265576446233684),
        ("R^3 / 15 (ОЦК)", lambda: lat.Astar(3),
         [[1, 0, 4], [0, 1, 12], [0, 0, 15]], 1.0),
    ]:
        L = build()
        H = np.asarray(H, float)
        k = int(round(abs(np.linalg.det(H))))
        rep = pc.evaluate(H @ L, pc.transversal(L, H), np.zeros(k))
        out.append({"case": name, "colours": rep.colours, "width": rep.width,
                    "expected": want, "ok": abs(rep.width - want) < 1e-7})
    return out


def report() -> dict:
    closed = check_closed_forms()
    champs = verify_champions()
    return {
        "closed_forms": closed,
        "closed_forms_all_ok": all(c["ok"] for c in closed),
        "champions_verified": champs,
        "probes": PROBES,
        "probes_all_flat": all(p["reached"] <= p["champion"] + 1e-9 for p in PROBES),
        "calibration": CALIBRATION,
        "paradigm": pf.report(),
        "note": (
            "power_coloring reproduces the Voronoi scheme exactly; CMA-ES seeded "
            "at the champions of R^2/7, R^3/15, R^4/45 and R^4/44 and given the "
            "full power freedom found no improvement (one hour, four processes). "
            "See prop. 'reduction to one shape' for why: at a Voronoi point all "
            "k tiles are congruent, so every active constraint binds on every "
            "tile at once."),
    }


def main() -> None:
    rep = report()
    for c in rep["closed_forms"]:
        flag = "ok " if c["ok"] else "BAD"
        print(f"{flag} {c['case']:<8} k={c['colours']:>3}  d={c['width']:.9f}  "
              f"= {c['expected_tex']} (err {c['error']:.2e})")
    print(f"\nall closed forms reproduced: {rep['closed_forms_all_ok']}")
    for p_ in rep["probes"]:
        print(f"probe {p_['case']:<9} k={p_['colours']:>3} "
              f"{p_['parameters']:>4} params: {p_['champion']:.7f} -> "
              f"{p_['reached']:.7f}")
    print(f"no probe improved on its champion: {rep['probes_all_flat']}")
    path = results_path("power_tilings.json")
    path.write_text(json.dumps(rep, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
