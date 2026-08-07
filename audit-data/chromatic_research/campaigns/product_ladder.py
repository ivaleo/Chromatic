"""Assemble the width ladders and run the product calculus over dimensions 2..26.

Every rung is either a closed form (the alpha-ladder of :mod:`product_calculus`
applied to a lattice of known ``rho = 2R/lambda1``) or a certified result of this
project.  The dynamic programme then minimises ``prod k_i`` over orthogonal
splittings ``n = sum n_i`` subject to ``sum 1/d_i^2 <= 1``.

Headline output:  ``chi(R^10, [1, sqrt(217/214)]) <= 45619`` from
``E8/2401 (+) A2/19``, against ``3^10 = 59049``.

Usage::

    python -m chromatic_research.campaigns.product_ladder
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

from chromatic_research.core.product_calculus import (
    Entry, best_partition, eisenstein_ladder, hurwitz_distance, pareto, real_ladder,
)
from chromatic_research.paths import results_path

SQRT2 = math.sqrt(2.0)

# rho = diam/lambda1 = 2R/lambda1 for the best lattice of each dimension
RHO = {
    1: 1.0,                     # Z
    2: 2 / math.sqrt(3),        # A2
    3: math.sqrt(5 / 3),        # A3* (bcc)
    4: SQRT2,                   # D4
    5: math.sqrt(7 / 3),        # A5*
    6: SQRT2,                   # E6*
    7: math.sqrt(7 / 3),        # E7*
    8: SQRT2,                   # E8
    9: math.sqrt(2.5),          # Lambda9   (R^2/lambda1^2 = 5/8)
    10: math.sqrt(8 / 3),       # Lambda10
    11: math.sqrt(3.0),         # Lambda11
    12: math.sqrt(8 / 3),       # K12
    16: math.sqrt(3.0),         # Lambda16
    24: SQRT2,                  # Leech
}
NAMES = {1: "Z", 2: "A2", 3: "A3*", 4: "D4", 5: "A5*", 6: "E6*", 7: "E7*", 8: "E8",
         9: "Lam9", 10: "Lam10", 11: "Lam11", 12: "K12", 16: "Lam16", 24: "Leech"}


def tower_rho(n_max: int) -> None:
    """Fill ``RHO`` for every dimension from the E8 lamination tower.

    ``diam^2_{n+1} <= (3/4) diam^2_n + 1`` at ``lambda1 = 1`` (the project's
    tower recursion, proved for all ``n >= 8``), starting from ``diam^2 = 2``.
    The fixed point is 4, so ``rho < 2`` and ``Gamma = 3 Lambda`` stays
    admissible in every dimension.  A dimension already carrying a sharper
    classical value keeps it.
    """
    squared = 2.0
    for n in range(9, n_max + 1):
        squared = 0.75 * squared + 1.0
        rho = math.sqrt(squared)
        if RHO.get(n, math.inf) > rho:
            RHO[n] = rho
            NAMES[n] = f"tower{n}"
        NAMES.setdefault(n, f"tower{n}")

EISENSTEIN_DIMS = (2, 4, 6, 8, 12, 24)

# results this project has certified; see RESULTS.md
CERTIFIED = [
    Entry(4, 45, 1.0, "chi(R^4) <= 45, generic lattice"),
    Entry(5, 132, 1.010897714, "chi(R^5) <= 132"),
    Entry(5, 140, 1.055597, "A5*/140 = sqrt(39/35)"),
    Entry(5, 196, 1.183216, "A5*/196 = sqrt(7/5)"),
    Entry(5, 270, 1.242118, "A5*/270"),
    Entry(5, 300, 1.264911, "A5*/300 = sqrt(8/5)"),
    Entry(7, 1029, 1.032881, "E6* laminated, m=3"),
    Entry(7, 1323, 1.007032, "chi(R^7) <= 1323"),
    Entry(9, 7203, 1.016591, "E8/2401 laminated, m=3"),
    Entry(9, 9604, 1.058127, "E8/2401 laminated, m=4"),
]

# maxima of an exhaustive 2D search (all Hermite forms x Nelder-Mead over shapes);
# only the rungs that beat the closed-form Eisenstein ladder are listed
NUMERIC_2D = [(15, 2.1866070), (20, 2.8628207), (22, 3.0157908), (24, 3.1787814)]


def hurwitz_ladder(limit: int = 40) -> list[Entry]:
    """``Gamma = alpha D4`` for Hurwitz ``alpha``; index ``N(alpha)^2``.

    The Hurwitz order is ``Z^4`` *together with* the half-integer points
    ``(a+bi+cj+dk)/2`` with ``a,b,c,d`` all odd.  Both halves must be swept:
    Lagrange makes every norm reachable by an integer quaternion already, but a
    half-integer element of the same norm sits in a different direction relative
    to the 24-cell and can be farther from it, which is what the rung measures.
    """
    out: dict[int, float] = {}
    span = 2 * int(math.isqrt(limit)) + 2
    for doubled in itertools.product(range(-span, span + 1), repeat=4):
        if len({v % 2 for v in doubled}) != 1:
            continue                      # all even (integer) or all odd (half)
        quadruple = [v / 2 for v in doubled]
        n = sum(v * v for v in quadruple)
        if not 1 < n <= limit or abs(n - round(n)) > 1e-9:
            continue
        n = round(n)
        out[n] = max(out.get(n, 0.0), SQRT2 * hurwitz_distance(quadruple))
    return [Entry(4, n * n, w, f"D4 x Hurwitz N={n}") for n, w in sorted(out.items())
            if w > 1.0]


def build_ladders(limit: int = 60, n_max: int = 26) -> dict[int, list[Entry]]:
    tower_rho(n_max)
    ladders: dict[int, list[Entry]] = {}
    for dim, rho in RHO.items():
        rungs = real_ladder(dim, rho, NAMES[dim])
        if dim in EISENSTEIN_DIMS:
            rungs += eisenstein_ladder(dim, rho, NAMES[dim], limit)
        ladders.setdefault(dim, []).extend(rungs)
    ladders[4].extend(hurwitz_ladder())
    for e in CERTIFIED:
        ladders.setdefault(e.dim, []).append(e)
    for k, w in NUMERIC_2D:
        ladders[2].append(Entry(2, k, w, f"A2-family generic index {k}", exact=False))
    return {d: pareto(v) for d, v in ladders.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dim", type=int, default=26)
    parser.add_argument("--exact-only", action="store_true",
                        help="drop rungs that rest on a numeric search")
    args = parser.parse_args()

    ladders = build_ladders(n_max=args.max_dim)
    if args.exact_only:
        ladders = {d: [e for e in v if e.exact] for d, v in ladders.items()}
        ladders = {d: v for d, v in ladders.items() if v}

    print("ladders (index -> width), Pareto-reduced:")
    for dim in sorted(ladders):
        rungs = ", ".join(f"{e.index}:{e.width:.4f}" for e in ladders[dim][:8])
        print(f"  n={dim:2d}  {rungs}")

    records = []
    print(f"\n{'n':>3} {'colours':>14} {'3^n':>14} {'gain':>7} {'width':>10}   split")
    for n in range(2, args.max_dim + 1):
        found = best_partition(ladders, n)
        if found is None:
            continue
        colours, chain, width = found
        split = " (+) ".join(f"{e.source}/{e.index}" for e in chain)
        rec = dict(n=n, colours=colours, width=width, three_n=3 ** n,
                   exact=all(e.exact for e in chain),
                   blocks=[dict(dim=e.dim, index=e.index, width=e.width,
                                source=e.source, exact=e.exact) for e in chain])
        records.append(rec)
        print(f"{n:3d} {colours:14d} {3**n:14d} {3**n/colours:7.2f} {width:10.6f}   {split}")

    path = results_path("product_ladder.json")
    path.write_text(json.dumps(dict(ladder_sizes={d: len(v) for d, v in ladders.items()},
                                    records=records), indent=2))
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
