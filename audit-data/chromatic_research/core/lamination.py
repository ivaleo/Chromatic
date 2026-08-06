"""Laminated lifting: build a colouring of ``R^n`` out of one in ``R^{n-1}``.

Motivation.  The Minkowski screen in :mod:`chromatic_research.core.minkowski`
shows that in dimension 9 the published index ``17253`` sits ``10.5x`` above the
volume obstruction, while every other dimension of the project sits at
``1.4x .. 4.0x``.  Dimension 9 is therefore not short of optimisation effort in
its own right -- it is short of a *construction*.  The construction below reuses
the excellent 8-dimensional one instead of searching from scratch.

Construction.  Let ``L' ⊂ R^{n-1}`` be a parent lattice, ``G' = ker psi`` a
colouring sublattice of index ``k``, ``c ∈ R^{n-1}`` a layer offset and ``t > 0``
a layer height.  Put

    L = { (x + i c, i t) : x ∈ L', i ∈ Z },                (rank n)
    G = { (x + i c, i t) : psi(x) + i a = 0,  m | i },     [L:G] = k m

with a glue parameter ``a ∈ L'/G'``.  As an abstract group ``L = L' ⊕ Z g`` with
``g = (c, t)``, so ``(x, i) -> (psi(x) + i a, i mod m)`` really is a character.

Two facts make the lift cheap to control; both are one-line proofs and they are
what turns the surplus ``d' > 1`` of the base colouring into a budget.

  (P1)  For any point ``(x, z)`` take the nearest layer ``i`` (``|z - i t| <=
        t/2``) and the nearest ``L'``-point to ``x - i c``.  Then
        ``dist((x,z), L)^2 <= R(L')^2 + t^2/4``, hence

            diam(L) <= sqrt(4 R(L')^2 + t^2) = sqrt(diam(L')^2 + t^2).

        The bound holds for *every* offset ``c``, and is attained exactly when
        some point is a deep hole of two adjacent layers at once.  Perturbing
        ``c`` off the deep-hole cosets destroys that coincidence and makes the
        true diameter strictly smaller -- the main lever of the method.

  (P2)  ``L' x {0} ⊂ L`` forces ``V0(L) ⊆ V0(L') x R``, so for a horizontal
        ``v`` (layer coordinate zero, i.e. ``v ∈ G'``)

            D_L(v) >= D_{L'}(v) >= d' diam(L').

        Horizontal separations can only improve under lamination.

So the binding constraints split cleanly: horizontal vectors need
``d' diam(L') >= diam(L)``, i.e. ``t^2 <= (d'^2 - 1) diam(L')^2`` -- a short
layer -- while the layered vectors (``i != 0``) need ``m`` large enough that the
vertical steps clear the cell.  Only the second family has to be checked
numerically, and there are few of them: every ``v`` with ``|v| > 2 diam`` clears
automatically because ``D(v) >= |v| - diam``.

Dimension 9 from ``E8/2401`` (``d' = sqrt(7/6)``) gives ``chi(R^9) <= 9604``.
Applying the same arithmetic elsewhere shows why 9 is the only beneficiary: the
lift costs a factor ``m >= 4``, and ``4 * 343 = 1372 > 1323`` in ``R^7``,
``4 * 132 = 528 > 343`` in ``R^6``, ``4 * 1323 = 5292 > 2401`` in ``R^8``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import combigeo


def unit_facets(basis: np.ndarray) -> list[tuple[list[float], float]]:
    """Supporting halfspaces of the Voronoi cell, normals normalised to length 1."""
    out = []
    for vector, offset in combigeo.relevant_facets(np.asarray(basis, float).tolist()):
        vector = np.asarray(vector, dtype=float)
        out.append(((vector / np.linalg.norm(vector)).tolist(), float(offset)))
    return out


def deep_hole(basis: np.ndarray, *, n_dirs: int = 500, seed: int = 0
              ) -> tuple[float, np.ndarray]:
    """Covering radius and a point attaining it, by vertex ascent over the cell.

    ``covrad.covering_radius`` solves one LP per random direction; that finds the
    farthest vertex only if some sampled direction happens to point at it.  Here
    each direction is iterated (``u <- argmax_{x in V0} <u, x>``), which climbs
    to a locally farthest vertex in a handful of LPs and converges much faster.
    Like the original it approaches the true radius **from below**, which is the
    safe direction: an underestimated radius understates the diameter, and the
    rigorous bound :attr:`Lamination.safe_diameter` is what a claim rests on.

    Which deep hole is returned matters: ``E8`` has more than one orbit of them,
    and the laminations built on different orbits are genuinely different
    lattices (ratio ``1.0397`` versus ``0.8863`` at index 12005).
    """
    from scipy.optimize import linprog

    basis = np.asarray(basis, dtype=float)
    n = basis.shape[0]
    facets = unit_facets(basis)
    normals = np.array([f[0] for f in facets], dtype=float)
    offsets = np.array([f[1] for f in facets], dtype=float)
    rng = np.random.default_rng(seed)
    best, witness = 0.0, np.zeros(n)
    for _ in range(n_dirs):
        direction = rng.standard_normal(n)
        point = direction
        for _ in range(12):
            direction = direction / (np.linalg.norm(direction) + 1e-300)
            result = linprog(-direction, A_ub=normals, b_ub=offsets,
                             bounds=[(None, None)] * n, method="highs")
            if not result.success:
                break
            point = result.x
            if np.linalg.norm(point - direction * float(direction @ point)) < 1e-12:
                direction = point
                break
            direction = point
        norm = float(np.linalg.norm(point))
        if norm > best:
            best, witness = norm, point
    return best, witness


def _quadratic_form(gram: np.ndarray) -> np.ndarray:
    """Cholesky-style decomposition used by the Fincke-Pohst recursion."""
    q = np.array(gram, dtype=float, copy=True)
    n = len(q)
    for i in range(n):
        for j in range(i + 1, n):
            q[j, i] = q[i, j]
            q[i, j] = q[i, j] / q[i, i]
        for k in range(i + 1, n):
            for l in range(k, n):
                q[k, l] = q[k, l] - q[k, i] * q[i, l]
    return q


def enumerate_upto(basis: np.ndarray, bound: float) -> np.ndarray:
    """Every nonzero lattice vector of norm ``<= bound`` (Fincke-Pohst).

    A coefficient box is hopeless here: for the dimension-9 sublattices below it
    holds a few million points around the ~250 that actually qualify.
    """
    basis = np.asarray(basis, dtype=float)
    n = basis.shape[0]
    q = _quadratic_form(basis @ basis.T)
    limit = bound * bound
    coefficients = [0] * n
    found: list[list[int]] = []

    def recurse(index: int, partial: float) -> None:
        if index < 0:
            if any(coefficients):
                found.append(list(coefficients))
            return
        remaining = limit - partial
        if remaining < -1e-12:
            return
        centre = 0.0
        for j in range(index + 1, n):
            centre += q[index, j] * coefficients[j]
        radius = math.sqrt(max(remaining, 0.0) / q[index, index])
        for value in range(math.ceil(-radius - centre - 1e-9),
                           math.floor(radius - centre + 1e-9) + 1):
            coefficients[index] = value
            recurse(index - 1, partial + q[index, index] * (value + centre) ** 2)
        coefficients[index] = 0

    recurse(n - 1, 0.0)
    if not found:
        return np.zeros((0, n))
    return np.asarray(found, dtype=float) @ basis


def min_separation(
    sublattice: np.ndarray,
    diameter: float,
    facets: list[tuple[list[float], float]],
) -> float:
    """``min D(v)`` over nonzero sublattice vectors, ``D(v) = 2 dist(v/2, V0)``.

    Vectors are visited in increasing norm and the sweep stops as soon as
    ``|v| - diam`` exceeds the running minimum, since ``D(v) >= |v| - diam``.
    """
    vectors = enumerate_upto(sublattice, 2.0 * diameter)
    if len(vectors) == 0:
        return math.inf
    norms = np.linalg.norm(vectors, axis=1)
    best = math.inf
    for index in np.argsort(norms):
        if norms[index] - diameter >= best:
            break
        separation = 2.0 * float(
            combigeo.dist_to_halfspaces((vectors[index] / 2).tolist(), facets)
        )
        best = min(best, separation)
    return best


@dataclass
class Lamination:
    """A laminated parent lattice and the data needed to score its colourings."""

    base: np.ndarray                  # basis of L' (rows)
    base_covering_radius: float       # R(L'), exact or an underestimate
    offset: np.ndarray                # layer offset c
    height: float                     # layer height t
    basis: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        base = np.asarray(self.base, dtype=float)
        n = base.shape[0]
        basis = np.zeros((n + 1, n + 1))
        basis[:n, :n] = base
        basis[n, :n] = np.asarray(self.offset, dtype=float)
        basis[n, n] = float(self.height)
        self.basis = basis

    @property
    def safe_diameter(self) -> float:
        """Rigorous upper bound (P1); needs no covering-radius computation."""
        return math.sqrt(4 * self.base_covering_radius**2 + self.height**2)

    def measured_diameter(self, *, n_dirs: int = 500, seed: int = 5) -> float:
        """Diameter from the vertex-ascent LP estimator (converges from below)."""
        radius, _ = deep_hole(self.basis, n_dirs=n_dirs, seed=seed)
        return 2.0 * radius

    def ratio(self, sublattice_rows: np.ndarray, *, diameter: float | None = None) -> float:
        """``min D(v) / diam``; with ``diameter=None`` the rigorous (P1) bound is used."""
        diam = self.safe_diameter if diameter is None else diameter
        real = np.asarray(sublattice_rows, dtype=float) @ self.basis
        return min_separation(real, diam, unit_facets(self.basis)) / diam


def lift_character(
    base_rows: np.ndarray,
    base_moduli: list[int],
    glue: np.ndarray,
    modulus: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Characters of the lifted colouring: ``(psi(x) + i a, i mod m)``."""
    base_rows = np.asarray(base_rows, dtype=np.int64)
    count, n = base_rows.shape
    rows = np.zeros((count + 1, n + 1), dtype=np.int64)
    rows[:count, :n] = base_rows
    rows[:count, n] = np.asarray(glue, dtype=np.int64)
    rows[count, n] = 1
    return [row for row in rows], list(base_moduli) + [int(modulus)]


def kernel_rows(base_rows, base_moduli, glue, modulus) -> np.ndarray:
    """Basis (as ROWS) of the kernel of the lifted character."""
    from chromatic_research.core.prime_radon import kernel_basis

    rows, moduli = lift_character(base_rows, base_moduli, glue, modulus)
    n = len(rows[0])
    # kernel_basis returns generators as columns
    return np.asarray(kernel_basis(rows, moduli, n), dtype=np.int64).T
