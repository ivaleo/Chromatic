"""Semi-lattice colourings: several superimposed lattices instead of one.

**The scheme.**  Fix a lattice ``Gamma`` (the *colour period*) and ``N``
translation vectors ``t_1, ..., t_N``.  Put

    P = union_i (Gamma + t_i),

let the pieces of the colouring be the cells of ``P`` (Voronoi, or Laguerre if
weights ``w_i`` are used), and give a piece the colour ``i`` of the orbit of its
site.  Writing ``C_i`` for the cell of ``t_i``, the colouring is admissible for
the forbidden interval ``[1, l]`` after scaling iff

    (a)  max_i diam(C_i)  <=  1                                       (pieces)
    (b)  min_i min_{v in Gamma\\{0}} dist(C_i, C_i + v)  >=  l         (colours)

and then ``chi(R^n, [1, l]) <= N``.  The width of the construction is
``d = sep / diam``, exactly as in the lattice scheme.

**Why this is strictly more general.**  If ``P`` happens to be a lattice then
``Gamma`` is a sublattice of it and ``N = [P : Gamma]`` -- the classical scheme
of Ivanov / Arman-Bondarenko-Prymak-Radchenko.  Requiring ``P`` to be a lattice
is an *arithmetic* constraint that has nothing to do with the geometry: in the
plane a similar (hence optimally shaped) sublattice exists only when the index
is a norm of the Eisenstein integers,

    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27, 28, ...

so the classical width ladder is **not monotone in N** -- it collapses at every
non-norm.  Dropping the group structure of ``P`` removes the quantisation while
keeping ``Gamma`` (which is what carries the colours) a perfect lattice.

**Master inequality.**  ``dist(C, C+v) = dist(v, C-C) >= |v| - w_C(v/|v|)``
with ``w_C`` the width of ``C``, so a sufficient condition is

    w_C(v)  +  l * diam(C)  <=  |v|        for all v in Gamma \\ {0}, all C,

binding at the minimal vectors: ``W + l*diam <= lambda1(Gamma)``.

**Two floors.**  (i) *Absolute*: the difference body gives ``Gamma`` admissible
for ``K = (C-C) (+) l B``; Brunn-Minkowski plus the isodiametric inequality then
force ``N >= (1 + l)^n`` for every colouring of this kind, in any dimension --
see :func:`absolute_floor`.  (ii) *Same-shape*: if the cells of a known
construction are merely rescaled, the slack in the master inequality caps the
gain at ``N_floor = N (1 - (d-1) diam / lambda1(Gamma))^n`` --
see :func:`same_shape_floor`.  The second is what says where to look.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def absolute_floor(n: int, ell: float = 1.0) -> float:
    """``N >= (1+l)^n`` for every semi-lattice (hence every lattice) colouring.

    Proof sketch: the colour class ``Gamma + t_i`` avoids the interval iff
    ``Gamma`` meets ``int((C_i - C_i) (+) l B)`` only in ``0``; the translates of
    half that body by ``Gamma`` therefore pack, so ``vol`` of the half body is at
    most ``covol(Gamma)``.  Brunn-Minkowski and ``vol((C-C)/2) >= vol(C)`` turn
    this into ``covol(Gamma)^{1/n} (1 - N^{-1/n}) >= (l/2) kappa_n^{1/n}``, and
    the isodiametric inequality (cells have diameter <= 1) bounds
    ``covol(Gamma) <= N kappa_n / 2^n``.  Combining gives ``N^{1/n} >= 1 + l``.
    """
    return (1.0 + ell) ** n


def same_shape_floor(n: int, colours: int, d: float, diam_over_lambda1: float) -> float:
    """Best ``N`` reachable by rescaling the cells of a known construction.

    ``d`` is its width and ``diam_over_lambda1 = diam(cell) / lambda1(Gamma)``.
    The master inequality has slack ``lambda1 - (W + diam) = (d-1) diam``; using
    all of it scales the cells by ``s = 1 / (1 - (d-1) diam / lambda1)`` and
    divides the number of colours by ``s^n``.
    """
    return colours * (1.0 - (d - 1.0) * diam_over_lambda1) ** n


def eisenstein_norms(limit: int) -> list[int]:
    """Norms ``a^2 - ab + b^2`` up to ``limit`` -- the indices at which a plane
    lattice has a *similar* (regular-hexagonal) sublattice."""
    out = set()
    r = int(math.isqrt(limit)) + 2
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            k = a * a - a * b + b * b
            if 0 < k <= limit:
                out.add(k)
    return sorted(out)


def aligned_ideal_plane(N: int) -> float:
    """Width of a *perfect aligned* triangular ``P`` with ``N`` sites per cell.

    With ``lambda1(Gamma) = 1`` the cells are regular hexagons of diameter
    ``(2/sqrt3)/sqrt N`` whose narrow directions are those of ``Gamma``, so
    ``sep = 1 - 1/sqrt N`` and ``d = (sqrt N - 1) / (2/sqrt 3)``.  Such a
    configuration exists only for ``N`` a perfect square (``sqrt N`` must be a
    real Eisenstein integer); for every other ``N`` it is an upper bound.
    """
    return (math.sqrt(N) - 1.0) / (2.0 / math.sqrt(3.0))


def plane_floor(ell: float) -> float:
    """``N >= (sqrt3/2 + l)^2 * 4/3`` -- the planar case of the master
    inequality with regular hexagonal cells, i.e. the exact critical-lattice
    bound.  For ``l = sqrt 7`` it gives 16.4436, so 17 is the first index the
    geometry allows and 19 the first the arithmetic allows."""
    return (math.sqrt(3.0) / 2.0 + ell) ** 2 * 4.0 / 3.0


@dataclass(frozen=True)
class Record:
    name: str
    n: int
    colours: int
    d: float
    diam_over_lambda1: float

    @property
    def floor(self) -> float:
        return same_shape_floor(self.n, self.colours, self.d, self.diam_over_lambda1)

    @property
    def room(self) -> float:
        return self.colours / self.floor


def eisenstein_record(n: int, colours: int) -> Record:
    """``Gamma = (3+omega) Lambda`` with ``rho = diam/lambda1 = sqrt2``
    (A2, E6*, E8, Leech): ``d = sqrt(7/6)``, ``lambda1(Gamma) = sqrt7 lambda1``."""
    rho = math.sqrt(2.0) if n > 2 else 2.0 / math.sqrt(3.0)
    # lambda1^2 = 3 normalisation: D_min = sqrt7, diam = rho*sqrt3,
    # lambda1(Gamma) = sqrt7*sqrt3, so d = sqrt(7/3)/rho.
    return Record(f"(3+w)Lambda n={n}", n, colours,
                  math.sqrt(7.0 / 3.0) / rho, rho / math.sqrt(7.0))


def tower_record(n: int, colours: int, rho: float) -> Record:
    """``Gamma = 3 Lambda``: ``d = 2/rho``, ``lambda1(Gamma) = 3 lambda1``."""
    return Record(f"3Lambda n={n}", n, colours, 2.0 / rho, rho / 3.0)
