"""Lamination by a whole layer *lattice* instead of one dimension at a time.

The project's lamination stacks copies of a coloured base along a line: layers at
heights ``tZ``, offset by ``c``, colour lifted with modulus ``m``.  A line is the
worst covering there is, so each new dimension costs a full ``t^2/4`` of the
diameter budget for one dimension.  Replacing ``tZ`` by an ``m``-dimensional
layer lattice ``L`` buys ``m`` dimensions for a single ``R(L)^2``: the hexagonal
``A2`` with ``R = 1/2`` costs exactly what ``Z`` with ``t = 1`` costs and
delivers two dimensions rather than one.

    Lambda = { (x + C^T k, L^T k) : x in base, k in Z^m }
    Gamma  = M Lambda,   M = [[A, 0], [G, S]]

``A`` is the base character kernel (for ``E8`` at ``lambda1^2 = 3`` and the
Eisenstein character, ``A = (3+omega)`` and the index is ``2401``), ``S`` cuts a
sublattice of index ``N`` out of ``L``, ``G`` is an integer glue and ``C`` holds
the layer offsets.  The colour count is ``[base : ker] * N``.

**Budget theorem.**  ``V0(Lambda) subset V0(base) x R^m`` keeps every horizontal
separation, so ``D`` on the base block is still ``D_0``; and a point is within
``R_L`` of some layer and within ``R_0`` of the base inside it, so

    diam(Lambda) <= 2 sqrt(R_0^2 + R_L^2).                                  (P1)

Admissibility therefore *requires nothing of the layer* as long as

    R_L <= (diam_0 / 2) sqrt(d_0^2 - 1),        d_0 = D_0 / diam_0,

which says exactly: **the surplus width of a colouring is the covering-radius
budget available for extra dimensions.**  For ``E8/2401`` (``diam_0 = sqrt 6``,
``d_0^2 = 7/6``) the budget is ``R_L <= 1/2`` -- one unit-height line (dimension
9, index ``2401*4 = 9604``) or one hexagon of ``lambda1 = sqrt3/2`` (dimension
10, index ``2401*19 = 45619``).

**Where (P1) leaks.**  (P1) assumes a point can be worst-case for the layer and
worst-case for the base at the same time.  It cannot, and the gap is large: in
dimension 9 the certified diameter at ``t = 1.15`` was ``2.6026`` against (P1)'s
``2.7060``, which is what turned ``9604`` into ``7203``.  The same identity
governs the general case.  Write ``f(y) = dist(y, base)^2``.  For a layer lattice
whose Delaunay cells are inscribed in spheres of radius ``R_L`` (true for ``A2``,
and for ``Z``), barycentric weights ``w`` of ``z`` in its Delaunay cell ``T``
satisfy ``sum_k w_k |z - v_k|^2 = R_L^2 - |z - o|^2``, so

    R(Lambda)^2  <=  R_L^2  +  max_x max_T sum_{k in T} w_k f(x - c_k).      (P1')

The second term is a *weighted mean* of ``f`` over the ``|T|`` shifted copies of
the base, not the single worst value ``R_0^2`` that (P1) uses -- and choosing the
offsets ``C`` is precisely the act of making that mean small.  Certifying it is
an eight-dimensional piecewise-convex maximum over ``|T|`` shifted Voronoi
decompositions, i.e. the natural extension of
:func:`chromatic_research.core.piecewise_covrad.certify_two_layer` from two
layers to a Delaunay cell's worth of them.  Until that exists, the honest status
of a construction that needs (P1') is *measured*, exactly as ``7203`` was before
its certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import combigeo
from chromatic_research.core.lamination import deep_hole, enumerate_upto


def layer_budget(diam_base: float, width_base: float) -> float:
    """Largest covering radius a layer lattice may have, from (P1).

    ``R_L <= (diam_0/2) sqrt(d_0^2 - 1)``: the surplus width of the base is the
    whole resource.  Returns 0 for a base of width 1 (no room to laminate).
    """
    surplus = width_base * width_base - 1.0
    return 0.0 if surplus <= 0 else 0.5 * diam_base * math.sqrt(surplus)


def eisenstein_map(alpha: complex, n_complex: int) -> np.ndarray:
    block = np.array([[alpha.real, -alpha.imag], [alpha.imag, alpha.real]])
    return np.kron(np.eye(n_complex), block)


@dataclass
class LayerLamination:
    """A base lattice with an ``m``-dimensional layer group stacked on it."""

    base: np.ndarray                 # rows, the coloured base lattice
    kernel: np.ndarray               # integer rows: the base colour kernel in `base`
    layer: np.ndarray                # rows, the layer lattice L
    sub_layer: np.ndarray            # integer rows: Gamma_L inside L
    offsets: np.ndarray              # m x n real layer offsets C
    glue: np.ndarray                 # m x n integer glue G
    r_base_sq: float                 # R(base)^2, exact or an upper bound
    r_layer_sq: float                # R(L)^2, exact or an upper bound

    @property
    def index(self) -> int:
        return int(round(abs(np.linalg.det(self.kernel))
                         * abs(np.linalg.det(self.sub_layer))))

    @property
    def p1_diameter(self) -> float:
        """Rigorous ``diam <= 2 sqrt(R_0^2 + R_L^2)``."""
        return 2.0 * math.sqrt(self.r_base_sq + self.r_layer_sq)

    def lattices(self) -> tuple[np.ndarray, np.ndarray]:
        n, m = self.base.shape[0], self.layer.shape[0]
        lam = np.zeros((n + m, n + m))
        lam[:n, :n], lam[n:, :n], lam[n:, n:] = self.base, self.offsets, self.layer
        gam = np.zeros((n + m, n + m))
        gam[:n, :n] = self.kernel @ self.base
        gam[n:, :n] = self.glue @ self.base + self.sub_layer @ self.offsets
        gam[n:, n:] = self.sub_layer @ self.layer
        return lam, gam

    def separation(self, diameter: float, *, facet_cap: int | None = None) -> float:
        """``min D(v)`` over ``Gamma \\ {0}``, a rigorous lower bound.

        Halfspaces are taken from lattice vectors of norm ``<= diameter``; every
        Voronoi-relevant vector is that short, so the intersection is exactly
        ``V0`` and the value is exact.  With ``facet_cap`` a shorter prefix is
        used, which only enlarges the region and so still bounds ``D`` below.
        """
        lam, gam = self.lattices()
        cell = enumerate_upto(lam, diameter)
        if facet_cap and len(cell) > facet_cap:
            cell = cell[np.argsort(np.linalg.norm(cell, axis=1))[:facet_cap]]
        facets = [((v / n).tolist(), n / 2.0)
                  for v, n in zip(cell, np.linalg.norm(cell, axis=1))]
        return _min_separation(gam, diameter, facets)

    def measured_diameter(self, *, n_dirs: int = 400, seed: int = 0) -> float:
        """``2 R`` by vertex ascent -- approached from *below*, so a candidate
        only.  A construction whose width needs this is 'measured', not proved."""
        lam, _ = self.lattices()
        return 2.0 * deep_hole(lam, n_dirs=n_dirs, seed=seed)[0]


def _min_separation(sub: np.ndarray, diam: float, facets) -> float:
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(sub.tolist())))
    radius, best, done = lam1 * 1.0001, math.inf, 0.0
    for _ in range(12):
        vectors = enumerate_upto(sub, radius)
        norms = np.linalg.norm(vectors, axis=1)
        for i in np.argsort(norms):
            if norms[i] <= done:
                continue
            if norms[i] - diam >= best:
                break
            best = min(best, 2.0 * float(
                combigeo.dist_to_halfspaces((vectors[i] / 2).tolist(), facets)))
        done = radius
        if best <= radius - diam:
            return best
        radius *= 1.25
    raise RuntimeError("separation sweep did not close")


def eisenstein_layer(base: np.ndarray, kernel: np.ndarray, offset: np.ndarray,
                     scale: float, alpha: complex, r_base_sq: float,
                     ) -> LayerLamination:
    """One Eisenstein layer: ``L = scale * A2``, ``Gamma_L = alpha L``.

    The construction is made ``Z[omega]``-equivariant on purpose.  ``omega``
    permutes the three minimal directions of ``A2``, so a single orbit of layer
    constraints replaces three independent ones, and the free parameters collapse
    from a real ``2 x n`` offset matrix to one complex vector -- the same eight
    numbers the dimension-9 campaign searched.  Because ``gcd(alpha, 3+omega) =
    1`` whenever ``N(alpha)`` is prime to 7, translating ``offset`` by the base
    lattice already realises every glue class, so ``G = 0`` loses nothing.
    """
    n = base.shape[0]
    a2 = scale * np.array([[1.0, 0.0], [-0.5, math.sqrt(3) / 2]])
    sub = np.rint(np.linalg.solve(a2.T, (a2 @ eisenstein_map(alpha, 1).T).T).T)
    omega = complex(-0.5, math.sqrt(3) / 2)
    offsets = np.vstack([offset, offset @ eisenstein_map(omega, n // 2).T])
    return LayerLamination(base=base, kernel=kernel, layer=a2,
                           sub_layer=sub.astype(int), offsets=offsets,
                           glue=np.zeros((2, n), dtype=int),
                           r_base_sq=r_base_sq, r_layer_sq=scale * scale / 3.0)
