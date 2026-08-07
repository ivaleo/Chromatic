"""How much room is left in the tiling paradigm, and where exactly it ends.

Every colouring this project can produce -- Voronoi cells of a lattice, layered
laminations, orthogonal products, and the power tilings of
:mod:`chromatic_research.core.power_coloring` -- has the same shape:

    space is cut into convex tiles of diameter ``< 1``, each tile gets a colour,
    and two tiles of one colour are more than ``1`` apart.

This module collects what can be said about *any* such colouring, with no
reference to how the tiling was built.

**The floor.**  Let the tiling be periodic with period lattice ``G``, let
``P_1 .. P_k`` be the tiles of one period and normalise the forbidden distance
to ``1``.  Fix the colour ``i``.  The bodies ``P_i + g + B(0, 1/2)``, ``g in
G``, are pairwise disjoint, because two tiles of one colour are more than ``1``
apart; hence

    det G  >=  vol(P_i + B(0, 1/2))                                       (1)

for every ``i``, while the tiles fill the period exactly:

    det G  =  sum_i vol(P_i).                                             (2)

Write ``V = max_i vol(P_i)`` and let ``j`` attain it.  Brunn-Minkowski applied
to (1) with ``i = j``, and then the isodiametric inequality ``V <= w_n 2^-n``
(the tile has diameter ``<= 1``), give

    det G  >=  (V^{1/n} + (w_n 2^-n)^{1/n})^n  >=  (2 V^{1/n})^n = 2^n V,

whereas (2) gives ``det G <= k V``.  Therefore

    **k >= 2^n**,   and more precisely   k >= (1 + theta^{1/n})^n,        (3)

with ``theta = w_n 2^-n / V >= 1`` the thinness of the largest tile.  Equality
would force every tile to be a ball, so (3) is never attained.

(3) holds for power tilings exactly as for Voronoi ones: enlarging the search
space from lattices to periodic power diagrams cannot break ``2^n``.  For
lattice colourings Coulson's ``2^{n+1} - 1`` is stronger; (3) is what survives
when the parent lattice is dropped.

**The accounting.**  For a Voronoi colouring of parent ``L`` with covering
radius ``R`` the admissibility condition ``D(v) >= 2R`` for ``v in G`` says
exactly ``G ∩ int 2C = {0}`` for the *rounded cell* ``C = V0 + R B``, i.e. ``G``
is a packing lattice of ``C``.  Hence the exact identity

    k  =  F(L) / delta_C(G),      F(L) = vol(V0 + R B) / det L,           (4)

``delta_C`` the packing density of ``C`` by ``G``.  ``F(L) >= (1 + Theta^{1/n})^n``
where ``Theta`` is the covering density of ``L``: the parent should be a *thin
covering*, the sublattice a *dense packing* of the rounded cell.  Every
construction of the project instead takes ``G = alpha L`` similar to ``L``,
which forces one lattice to be good at both.

**Where the Eisenstein ladder ends.**  For ``G = alpha L`` the admissibility is
a single inequality on ``rho = diam(V0)/lambda_1 = 2R/lambda_1``: ``rho <= 2``
for ``alpha = 3`` (``3^n`` colours) and ``rho <= sqrt(7/3) = 1.52753`` for
``alpha = 3 + omega`` (``7^{n/2} = 2.6458^n`` colours).  Now

    Theta(L) / delta(L) = (2R / lambda_1)^n = rho^n,                      (5)

and ``Theta >= 1``, so ``rho^n >= 1 / delta(L) >= 1 / delta_max(n)``.  By
Kabatiansky-Levenshtein ``delta_max(n) <= 2^{-(0.599 + o(1)) n}``, hence

    rho  >=  2^0.599 - o(1)  =  1.5147 - o(1).                            (6)

The Eisenstein step needs ``rho <= 1.52753``: it lives in a window **0.85 %**
wide above an absolute obstruction.  That is why ``7^{n/2}`` exists only in the
exceptional dimensions ``n = 2, 4, 6, 8, 24`` (where ``rho = sqrt 2`` or less)
and cannot be a general asymptotic mechanism -- while ``rho <= 2`` is loose
enough to be reachable in every dimension, which is precisely Larman-Rogers'
``3^n``.
"""

from __future__ import annotations

import json
import math

import numpy as np


def unit_ball_volume(n: int) -> float:
    return math.pi ** (n / 2) / math.gamma(n / 2 + 1)


def tiling_floor(n: int) -> int:
    """Universal floor (3): no convex-tiling colouring of R^n uses fewer."""
    return 2 ** n


def thinness(n: int, covering_radius: float, det: float) -> float:
    """``Theta = w_n R^n / det L`` -- covering density; ``>= 1`` always."""
    return unit_ball_volume(n) * covering_radius ** n / det


def brunn_floor(n: int, covering_radius: float, det: float) -> float:
    """``(1 + Theta^{1/n})^n`` -- rigorous lower bound for ``F(L)`` and for ``k``."""
    return (1.0 + thinness(n, covering_radius, det) ** (1.0 / n)) ** n


def packing_density(n: int, lambda1: float, det: float) -> float:
    return unit_ball_volume(n) * (lambda1 / 2.0) ** n / det


# --------------------------------------------------------------------------- #
# the landscape of the records                                                 #
# --------------------------------------------------------------------------- #

# The parent enters only through the scale-free pair (rho, delta): the covering
# density then follows from the identity (5), Theta = delta * rho^n, which is a
# cheap self-check -- it reproduces the tabulated Theta of A2 (1.2092), A3*
# (1.4635), E6* (2.6522), E8 (4.0587) and Leech (7.9035) to four digits.
#
# (n, colours, parent, rho = 2R/lambda_1, delta = packing density, width, source)
RECORDS = [
    (2, 7, "A2", 2 / math.sqrt(3), math.pi / math.sqrt(12), math.sqrt(7) / 2, "classical"),
    (3, 15, "A3* (bcc)", math.sqrt(5 / 3), math.pi * math.sqrt(3) / 8, 1.0, "Coulson"),
    (4, 45, "generic", None, None, 1.0163, "project"),
    (5, 132, "generic", None, None, 1.0109, "project"),
    (6, 343, "E6*", math.sqrt(2), 0.331531, math.sqrt(7 / 6), "ABPR"),
    (7, 1029, "lam E6*/343", None, None, 1.0329, "project"),
    (8, 2401, "E8", math.sqrt(2), 0.2536695, math.sqrt(7 / 6), "ABPR"),
    (9, 7203, "lam E8/2401", None, None, 1.0166, "project"),
    (10, 45619, "E8 x A2 product", None, None, math.sqrt(217 / 214), "project"),
    (11, 3 ** 11, "Lambda_11 tower", 2 / 1.116581, 0.06043, 1.116581, "project"),
    (12, 3 ** 12, "K12", math.sqrt(8 / 3), 0.0494539, math.sqrt(1.5), "project"),
    (24, 7 ** 12, "Leech", math.sqrt(2), 0.0019296, math.sqrt(7 / 6), "ABPR"),
]


def landscape() -> list[dict]:
    """Slack of every record against the floors of the paradigm."""
    rows = []
    for n, k, name, rho, delta, width, src in RECORDS:
        row = {"n": n, "colours": k, "parent": name, "width": width, "source": src,
               "base": k ** (1.0 / n), "floor_2n": tiling_floor(n),
               "slack_2n": k / tiling_floor(n)}
        if rho is not None:
            theta = delta * rho ** n                       # identity (5)
            fl = (1.0 + theta ** (1.0 / n)) ** n
            row |= {"rho": rho, "Theta": theta, "delta": delta,
                    "floor_parent": fl, "slack_parent": k / fl}
        rows.append(row)
    return rows


def eisenstein_window() -> dict:
    """The 0.85 % window (6) in which the ``7^{n/2}`` ladder step has to live."""
    need = math.sqrt(7.0 / 3.0)
    kl = 2.0 ** 0.599
    return {"rho_needed_eisenstein": need, "rho_needed_real_three": 2.0,
            "kl_asymptotic_floor": kl, "window_percent": 100.0 * (need / kl - 1.0),
            "base_eisenstein": math.sqrt(7.0), "base_real_three": 3.0}


def report() -> dict:
    return {"landscape": landscape(), "eisenstein_window": eisenstein_window(),
            "note": "floor_2n is rigorous for every convex-tiling colouring, "
                    "lattice or not; floor_parent = (1+Theta^{1/n})^n is the "
                    "Brunn-Minkowski bound for the named parent."}


if __name__ == "__main__":
    rep = report()
    hdr = f"{'n':>3} {'colours':>10} {'base':>7} {'parent':<18} {'rho':>7} " \
          f"{'Theta':>8} {'floor(par)':>11} {'k/floor':>8} {'k/2^n':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rep["landscape"]:
        print(f"{r['n']:>3} {r['colours']:>10} {r['base']:>7.4f} {r['parent']:<18} "
              f"{r.get('rho', float('nan')):>7.4f} {r.get('Theta', float('nan')):>8.3f} "
              f"{r.get('floor_parent', float('nan')):>11.1f} "
              f"{r.get('slack_parent', float('nan')):>8.2f} {r['slack_2n']:>8.1f}")
    w = rep["eisenstein_window"]
    print(f"\nEisenstein ladder needs rho <= {w['rho_needed_eisenstein']:.5f}; "
          f"Kabatiansky-Levenshtein forces rho >= {w['kl_asymptotic_floor']:.5f} "
          f"asymptotically -- a window of {w['window_percent']:.2f} %.")
    print(json.dumps({"eisenstein_window": w}, indent=2)[:0])
