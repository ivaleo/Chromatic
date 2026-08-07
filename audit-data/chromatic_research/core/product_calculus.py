"""Product calculus for interval colourings, and the alpha-ladder that feeds it.

**The rule.**  Let ``Lambda = (+) Lambda_i`` be an orthogonal direct sum with
``Gamma = (+) Gamma_i``.  The Voronoi cell is the product of the cells, so
``diam^2 = sum diam_i^2``; and because ``D(v)^2 = sum D_i(v_i)^2`` with
``D_i(0) = 0``, the minimum over ``Gamma \\ {0}`` is attained with a single
nonzero block.  Admissibility for the interval ``[1, l]`` therefore reads
``D_i,min^2 >= l^2 sum_j diam_j^2`` for every ``i``, and eliminating the free
scales of the blocks turns it into one inequality in the widths alone:

    sum_i 1 / d_i^2  <=  1,        l = ( sum_i 1/d_i^2 )^{-1/2},

where ``d_i = D_i,min / diam_i`` is the width of block ``i`` and the number of
colours is ``prod_i k_i``.  The optimal block scales are ``diam_i^2 ∝ 1/d_i^2``.
For ``Gamma_i = 3 Lambda_i`` this is the familiar ``sum rho_i^2 <= 4``, but the
general form accepts blocks of *different* index, which is what makes it bite:
``E8/2401`` spends ``6/7`` of the budget and a two-dimensional spacer of index
19 spends ``4/31``, so ``6/7 + 4/31 = 214/217 < 1`` and

    chi(R^10, [1, sqrt(217/214)]) <= 2401 * 19 = 45619      (against 3^10 = 59049).

**The alpha-ladder.**  Every block needs a *ladder* of (index, width) pairs, and
one family supplies it in closed form for any lattice.  If ``Lambda`` is a module
over an order ``R`` (``Z``, ``Z[i]``, ``Z[omega]``, Hurwitz ``H``) acting by
similarities, and ``alpha in R``, then ``Gamma = alpha Lambda`` has index
``|alpha|^n`` and, by the same projection argument as the planar theorem,

    D(alpha w) >= 2 |w| dist(alpha/2, U),        U = Voronoi cell of R w,

so ``d >= 2 dist(alpha/2, U) / rho`` with ``rho = diam(V0)/lambda1``.  ``U`` is a
segment, a square, a hexagon or a 24-cell; the four distance functions are
:func:`real_distance`, :func:`gaussian_distance`, :func:`eisenstein_distance`
and :func:`hurwitz_distance`.  Real ``alpha = m`` recovers ``d = (m-1)/rho``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

OMEGA = complex(-0.5, math.sqrt(3) / 2)


# --------------------------------------------------------------- the rule

def budget(widths) -> float:
    """``sum 1/d_i^2``.  Admissible iff ``<= 1``."""
    return float(sum(1.0 / (d * d) for d in widths))


def product_width(widths) -> float:
    """Width ``l`` of the orthogonal product, ``inf`` if a block is degenerate."""
    total = budget(widths)
    return math.inf if total <= 0 else total ** -0.5


def block_scales(widths) -> list[float]:
    """Relative cell diameters ``diam_i`` realising the optimum (up to a scale)."""
    return [1.0 / d for d in widths]


# ------------------------------------------------- unit cells of the four orders

def real_distance(alpha: float) -> float:
    """dist(alpha/2, [-1/2, 1/2]).  Gives ``D = (m-1) lambda1`` for integer m."""
    return max(abs(alpha) / 2.0 - 0.5, 0.0)


def gaussian_distance(alpha: complex) -> float:
    """dist(alpha/2, unit square) -- the cell of ``Z[i] w``."""
    p = np.array([alpha.real, alpha.imag]) / 2.0
    return float(np.linalg.norm(np.clip(np.abs(p) - 0.5, 0.0, None)))


def eisenstein_distance(alpha: complex) -> float:
    """dist(alpha/2, hexagon) -- the cell of ``Z[omega] w``, ``|w| = 1``."""
    normals = [np.array([math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)])
               for k in range(6)]
    return _dist_to_cell(np.array([alpha.real, alpha.imag]) / 2.0, normals, 0.5)


def hurwitz_distance(alpha) -> float:
    """dist(alpha/2, 24-cell) -- the cell of ``H w`` for Hurwitz quaternions.

    ``alpha`` is a real 4-vector ``(a, b, c, d)`` meaning ``a + bi + cj + dk``.
    """
    units = [np.eye(4)[i] * s for i in range(4) for s in (1, -1)]
    units += [np.array(s) / 2.0 for s in itertools.product((1, -1), repeat=4)]
    return _dist_to_cell(np.asarray(alpha, float) / 2.0, units, 0.5)


def _dist_to_cell(point, normals, offset) -> float:
    """Distance from ``point`` to ``{x : <x, u> <= offset |u|, all u}`` (|u| = 1).

    Active-set projection: the cells here have few facets, so trying every
    subset of the violated ones is both exact and instant.
    """
    normals = [np.asarray(u, float) for u in normals]
    violated = [u for u in normals if float(u @ point) > offset + 1e-12]
    if not violated:
        return 0.0
    best = math.inf
    for size in range(1, len(violated) + 1):
        for active in itertools.combinations(violated, size):
            A = np.array(active)
            gram = A @ A.T
            if abs(np.linalg.det(gram)) < 1e-12:
                continue
            lam = np.linalg.solve(gram, A @ point - offset)
            if np.any(lam < -1e-12):
                continue
            x = point - lam @ A
            if all(float(u @ x) <= offset + 1e-9 for u in normals):
                best = min(best, float(np.linalg.norm(point - x)))
    return best


# ------------------------------------------------------------------ ladders

@dataclass(frozen=True)
class Entry:
    """One rung: ``index`` colours in dimension ``dim`` with width ``width``."""
    dim: int
    index: int
    width: float
    source: str
    exact: bool = True

    @property
    def cost(self) -> float:
        """Share of the product budget this rung consumes."""
        return 1.0 / (self.width * self.width)


def eisenstein_norms(limit: int) -> list[tuple[int, complex]]:
    """``(N(alpha), alpha)`` for ``alpha in Z[omega]``, one representative per norm."""
    seen: dict[int, complex] = {}
    span = int(math.isqrt(limit)) + 2
    for a in range(-span, span + 1):
        for b in range(-span, span + 1):
            n = a * a - a * b + b * b
            if 1 < n <= limit:
                alpha = a + b * OMEGA
                if n not in seen or eisenstein_distance(alpha) > eisenstein_distance(seen[n]):
                    seen[n] = alpha
    return sorted(seen.items())


def eisenstein_ladder(dim: int, rho: float, name: str, limit: int = 60) -> list[Entry]:
    """``Gamma = alpha Lambda`` for an Eisenstein ``Lambda`` of real dimension ``dim``."""
    out = []
    for n, alpha in eisenstein_norms(limit):
        d = 2.0 * eisenstein_distance(alpha) / rho
        if d > 1.0:
            out.append(Entry(dim, n ** (dim // 2), d, f"{name} x ({alpha:.3g})"))
    return out


def real_ladder(dim: int, rho: float, name: str, limit: int = 8) -> list[Entry]:
    """``Gamma = m Lambda``: ``d = (m-1)/rho``, index ``m^dim``."""
    return [Entry(dim, m ** dim, (m - 1) / rho, f"{name} x {m}")
            for m in range(2, limit + 1) if (m - 1) / rho > 1.0]


def pareto(entries: list[Entry]) -> list[Entry]:
    """Keep rungs that no cheaper rung matches in width."""
    best: list[Entry] = []
    for e in sorted(entries, key=lambda e: (e.index, -e.width)):
        if all(e.width > b.width + 1e-12 for b in best):
            best.append(e)
    return best


# ------------------------------------------------------------ the optimisation

def best_partition(ladders: dict[int, list[Entry]], n: int,
                   ) -> tuple[int, list[Entry], float] | None:
    """Cheapest orthogonal product covering dimension ``n``.

    Dynamic programme over ``(dimension used, budget spent)``.  The budget is
    continuous, so it is carried as an exact float in the state list and pruned
    by dominance: for a fixed dimension only Pareto-optimal ``(colours, budget)``
    pairs can extend to an optimum.
    """
    states: dict[int, list[tuple[int, float, tuple[Entry, ...]]]] = {
        0: [(1, 0.0, ())]}
    for used in range(n + 1):
        for colours, spent, chain in states.get(used, []):
            for step, rungs in ladders.items():
                if used + step > n:
                    continue
                for rung in rungs:
                    new_spent = spent + rung.cost
                    if new_spent > 1.0 + 1e-12:      # equality = the plain bound
                        continue
                    key = used + step
                    entry = (colours * rung.index, new_spent, chain + (rung,))
                    bucket = states.setdefault(key, [])
                    if any(b[0] <= entry[0] and b[1] <= entry[1] for b in bucket):
                        continue
                    bucket[:] = [b for b in bucket
                                 if not (entry[0] <= b[0] and entry[1] <= b[1])]
                    bucket.append(entry)
    final = states.get(n, [])
    if not final:
        return None
    colours, spent, chain = min(final, key=lambda s: (s[0], s[1]))
    return colours, list(chain), spent ** -0.5
