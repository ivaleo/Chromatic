"""Planar semi-lattice colourings: rationalise a numeric configuration and
certify it exactly.

A *semi-lattice colouring* superimposes several lattices: ``P = U_i (Gamma+t_i)``
with ``N`` free translates, the pieces are the (Laguerre) cells of ``P`` and the
colour of a piece is the index of its orbit.  When ``P`` happens to be a lattice
this is the classical scheme with ``N = [P : Gamma]``; the point of dropping the
group structure is that the classical index must be *geometrically realisable*
-- in the plane a similar (regular hexagonal) sublattice exists only when the
index is a norm of ``Z[omega]`` (1, 3, 4, 7, 9, 12, 13, 16, 19, 21, ...) -- so
the classical width ladder collapses at every other index.

This module takes a numeric optimum, rounds it to rationals and hands it to
:mod:`chromatic_research.core.semilattice_cert`, which returns exact ``diam^2``
and ``sep^2``.  The resulting statement, ``chi(R^2, [1, l]) <= N`` for every
``l < sep/diam``, is then a theorem, not a measurement.

Usage::

    python -m chromatic_research.campaigns.planar_semilattice cfg.json --target 1.4
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction as F

import numpy as np

from chromatic_research.core.semilattice_cert import certify
from chromatic_research.paths import results_path


def to_basis_coords(basis, ts):
    return np.linalg.solve(np.asarray(basis, float).T, np.asarray(ts, float).T).T


def rational_gram(basis):
    """Exact Gram matrix of a rational or hexagonal ``Gamma``."""
    b = np.asarray(basis, float)
    g = b @ b.T
    return [[F(g[0][0]).limit_denominator(10**6), F(g[0][1]).limit_denominator(10**6)],
            [F(g[1][0]).limit_denominator(10**6), F(g[1][1]).limit_denominator(10**6)]]


def rationalise_and_certify(gram, coords, weights, dens=(60, 200, 1000, 5000, 20000),
                            nb_shells=3, verbose=True):
    """Round to rationals with growing denominators; return the best certificate."""
    best = None
    for q in dens:
        ts = [(F(float(c[0])).limit_denominator(q), F(float(c[1])).limit_denominator(q))
              for c in coords]
        ws = [F(float(w)).limit_denominator(q*q) for w in weights]
        if len({t for t in ts}) != len(ts):
            continue
        try:
            r = certify(gram, ts, ws, nb_shells=nb_shells)
        except Exception as exc:                       # pragma: no cover
            if verbose:
                print(f"  q={q}: {exc}")
            continue
        if r is None or "error" in r:
            if verbose:
                print(f"  q={q}: {r}")
            continue
        r = dict(r, denominator=q, ts=[(str(a), str(b)) for a, b in ts],
                 ws=[str(w) for w in ws])
        if verbose:
            print(f"  q={q:6d}: d^2 = {float(r['d2']):.9f}  d = {math.sqrt(float(r['d2'])):.9f}"
                  f"  tiling_exact={r['tiling_exact']}")
        if best is None or r["d2"] > best["d2"]:
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="JSON with keys gamma (2x2), ts (Nx2 cartesian), ws (N)")
    ap.add_argument("--target", type=float, default=None,
                    help="width of the best classical rung, to compare against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    basis = cfg.get("gamma") or [[1.0, 0.0], [0.5, math.sqrt(3)/2]]
    gram = rational_gram(basis)
    coords = to_basis_coords(basis, cfg["ts"])
    best = rationalise_and_certify(gram, coords, cfg["ws"])
    if best is None:
        print("no certificate found")
        return
    d = math.sqrt(float(best["d2"]))
    print(f"\nN = {len(cfg['ts'])}   exact d^2 = {best['d2']}   d = {d:.9f}")
    if args.target:
        print(f"classical best = {args.target:.9f}   gain = {100*(d/args.target-1):+.3f}%")
        print("VERDICT:", "beats the classical rung" if d > args.target else "does not beat it")
    if args.out:
        path = results_path(args.out)
        path.write_text(json.dumps(
            {"N": len(cfg["ts"]), "gram": [[str(x) for x in row] for row in gram],
             "d2": str(best["d2"]), "d": d, "denominator": best["denominator"],
             "tiling_exact": best["tiling_exact"], "n_vertices": best["n_vertices"],
             "ts": best["ts"], "ws": best["ws"], "classical": args.target}, indent=1))
        print(f"written {path}")


if __name__ == "__main__":
    main()
