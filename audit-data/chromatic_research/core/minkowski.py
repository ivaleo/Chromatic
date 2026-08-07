"""Lower bound on the index of a lattice colouring, from Minkowski's theorem.

The admissibility condition used throughout the project, ``D(v) = 2 dist(v/2, V0)
>= diam(V0) = 2R`` for every nonzero ``v`` of the colouring sublattice ``G``, is
exactly the geometric statement

    G ∩ int K = {0},        K = 2 (V0 ⊕ R B),

where ``V0`` is the Voronoi cell of the parent lattice ``L``, ``R`` its covering
radius and ``B`` the unit ball.  ``K`` is centrally symmetric and convex (a
Minkowski sum of a symmetric polytope and a ball), so Minkowski's convex body
theorem gives ``vol K <= 2^n det G`` and therefore

    k = [L:G] = det G / det L  >=  vol(V0 ⊕ R B) / det L.               (1)

This is a **rigorous lower bound valid for every sublattice of a given parent**,
obtained without any search.  Until now index minimality in this project was
established only by exhaustive enumeration (for instance 1.4e9 sublattices of
index 140 in A5*); (1) rules out whole ranges of indices for free.

Brunn-Minkowski turns (1) into a closed form that needs only the covering
radius, with no volume computation:

    k >= (det L^{1/n} + R w_n^{1/n})^n / det L,                          (2)

``w_n`` the volume of the unit ball.  Since ``V0 ⊆ R B`` always gives
``R >= (det L / w_n)^{1/n}``, (2) implies the parent-free bound ``k >= 2^n``
(Coulson's ``2^{n+1} - 1`` is stronger; the value of (1)-(2) is that they are
tied to a specific parent lattice).

Bound (1) is sharper than (2) but its volume is estimated by Monte Carlo, so
:func:`minkowski_bound` reports the rigorous closed form (2) alongside the
sampled value of (1) with its standard error.

Calibration on the constructions of the project (ratio ``k / vol(V0 ⊕ R B)``,
"how far the best known colouring is above the obstruction"):

    n=2 A2*/7    1.55     n=6 E6*/343    2.68
    n=3 A3*/15   1.44     n=7 E7*/1372   4.03
    n=4 D4/49    1.70     n=8 E8/2401    3.62
    n=5 A5*/140  2.52     n=9 A9*/17253 10.52

The dimension-9 outlier is what motivated the laminated construction in
:mod:`chromatic_research.core.lamination`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import combigeo


def unit_ball_volume(n: int) -> float:
    """Volume ``w_n`` of the unit ball in ``R^n``."""
    return math.pi ** (n / 2) / math.gamma(n / 2 + 1)


def brunn_minkowski_bound(n: int, covering_radius: float, det: float = 1.0) -> float:
    """Rigorous closed-form bound (2); needs only the covering radius."""
    return (det ** (1 / n) + covering_radius * unit_ball_volume(n) ** (1 / n)) ** n / det


def hermite_constant_bound(n: int) -> float:
    """Upper bound for Hermite's constant ``gamma_n`` (exact for ``n <= 8``)."""
    exact = {1: 1.0, 2: 2 / math.sqrt(3), 3: 2 ** (1 / 3), 4: math.sqrt(2),
             5: 8 ** (1 / 5), 6: (64 / 3) ** (1 / 6), 7: 64 ** (1 / 7), 8: 2.0}
    if n in exact:
        return exact[n]
    return 1 + n / 4              # Minkowski-Hlawka style bound, valid for all n


def inradius_floor(n: int, lambda1: float, diameter: float, det: float,
                   width: float = 1.0) -> float:
    """Index floor from the inradius lemma plus Hermite -- complementary to (1).

    ``B(0, lambda1/2) subset V0`` gives ``D(v) <= |v| - lambda1``, so a colouring
    of width ``l`` forces every nonzero ``v`` of ``G`` to satisfy

        |v| >= l * diam + lambda1,      i.e.   lambda1(G) >= l * diam + lambda1,

    and Hermite's inequality ``lambda1(G)^n <= gamma_n^{n/2} det G`` turns that
    into a bound on the index:

        k = det G / det L  >=  (l * diam + lambda1)^n / (gamma_n^{n/2} det L).

    This is *not* implied by the volumetric bound (1) and is sometimes sharper.
    In the plane at ``l = sqrt 7`` -- the width a spacer needs beside ``E8/2401``
    in the dimension-10 product -- it gives ``k >= 16.44`` against (1)'s
    ``15.57``, so it is what rules out a plane block of index 16.
    """
    required = width * diameter + lambda1
    return required ** n / (hermite_constant_bound(n) ** (n / 2) * det)


@dataclass
class MinkowskiBound:
    """All three bound fields are indices, i.e. already divided by ``det L``.

    ``dilated_cell_volume`` returns the raw volume; forgetting to divide it by the
    determinant makes a parent with ``det != 1`` look as if it violated its own
    bound, so the division happens here, once.
    """

    n: int
    covering_radius: float
    det: float
    closed_form: float          # rigorous bound (2), an index
    volume_bound: float         # sampled bound (1), an index
    volume_bound_stderr: float
    raw_volume: float           # vol(V0 + R B) itself

    @property
    def index_lower_bound(self) -> int:
        """Smallest index not excluded by the closed-form bound."""
        return math.ceil(self.closed_form - 1e-9)

    def as_json(self) -> dict:
        return {
            "n": self.n,
            "covering_radius": self.covering_radius,
            "det": self.det,
            "closed_form_bound": self.closed_form,
            "index_lower_bound": self.index_lower_bound,
            "volume_bound": self.volume_bound,
            "volume_bound_stderr": self.volume_bound_stderr,
            "raw_volume": self.raw_volume,
        }


def dilated_cell_volume(
    basis: np.ndarray,
    covering_radius: float,
    *,
    samples: int = 20000,
    seed: int = 0,
) -> tuple[float, float]:
    """Monte Carlo estimate of ``vol(V0 ⊕ R B)``.

    Sampling is uniform in the ball of radius ``2R``, which certainly contains
    ``V0 ⊕ R B`` because ``V0 ⊆ R B``.  Membership is decided by the distance to
    the cell (Dykstra projection onto the supporting halfspaces).
    """
    basis = np.asarray(basis, dtype=float)
    n = basis.shape[0]
    raw = combigeo.relevant_facets(basis.tolist())
    normals = np.array([f[0] for f in raw], dtype=float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.array([f[1] for f in raw], dtype=float)
    facets = [(normals[i].tolist(), float(offsets[i])) for i in range(len(offsets))]

    rng = np.random.default_rng(seed)
    outer = 2.0 * covering_radius
    outer_volume = unit_ball_volume(n) * outer**n

    directions = rng.standard_normal((samples, n))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = outer * rng.random(samples) ** (1.0 / n)
    points = directions * radii[:, None]

    hits = 0
    for point in points:
        if float(combigeo.dist_to_halfspaces(point.tolist(), facets)) <= covering_radius:
            hits += 1
    p = hits / samples
    return outer_volume * p, outer_volume * math.sqrt(max(p * (1 - p), 1e-12) / samples)


def minkowski_bound(
    basis: np.ndarray,
    covering_radius: float,
    *,
    samples: int = 20000,
    seed: int = 0,
) -> MinkowskiBound:
    """Index lower bound for a parent lattice with a known covering radius.

    ``covering_radius`` must be exact or an underestimate: underestimating only
    weakens the bound, whereas overestimating it would invalidate the bound.
    The random-direction LP estimator in :mod:`covrad` converges from below and
    is therefore safe to feed here.
    """
    basis = np.asarray(basis, dtype=float)
    n = basis.shape[0]
    det = abs(float(np.linalg.det(basis)))
    volume, stderr = dilated_cell_volume(basis, covering_radius, samples=samples, seed=seed)
    return MinkowskiBound(
        n=n,
        covering_radius=covering_radius,
        det=det,
        closed_form=brunn_minkowski_bound(n, covering_radius, det),
        volume_bound=volume / det,
        volume_bound_stderr=stderr / det,
        raw_volume=volume,
    )
