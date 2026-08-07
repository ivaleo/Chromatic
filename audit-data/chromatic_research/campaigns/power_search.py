"""Search for power-tiling colourings that beat the lattice ones.

A colouring is a triple ``(G, T, w)``: the period lattice ``G``, ``k`` sites and
``k`` weights (see :mod:`chromatic_research.core.power_coloring`).  Its width is

    d = min_i min_{0 != g in G} dist(g, P_i - P_i)  /  max_i diam(P_i),

admissible as soon as ``d >= 1``, and then ``chi(R^n, [1, d]) <= k``.

The lattice colourings of the project are the slice ``w = 0``, ``T`` a
transversal of ``L / G``.  This module optimises over the full space with
CMA-ES: ``n(n+1)/2`` parameters for ``G`` (normalised to unit covolume),
``n (k-1)`` for the sites (in ``G``-fractional coordinates, one site pinned at
the origin) and ``k-1`` for the weights.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np

from chromatic_research.core import lattices as lat
from chromatic_research.core import power_coloring as pc
from chromatic_research.paths import results_path


# --------------------------------------------------------------------------- #
# parameter packing                                                            #
# --------------------------------------------------------------------------- #

def unpack(x: np.ndarray, n: int, k: int, *, fixed_gram: np.ndarray | None = None):
    """Split a CMA vector into ``(G, T, w)``."""
    pos = 0
    if fixed_gram is None:
        m = n * (n + 1) // 2
        tri = x[:m]
        pos = m
        G = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                G[i, j] = tri[idx] if i != j else math.exp(0.35 * tri[idx])
                idx += 1
        det = abs(np.linalg.det(G))
        if det < 1e-9:
            return None, None, None
        G /= det ** (1.0 / n)
    else:
        G = np.asarray(fixed_gram, float)

    frac = np.zeros((k, n))
    frac[1:] = x[pos:pos + n * (k - 1)].reshape(k - 1, n)
    pos += n * (k - 1)
    frac = frac - np.floor(frac)                    # sites live in one G-cell
    T = frac @ G

    # power distance is |x - t|^2 - w, so the weights carry units of length^2:
    # scale them by the cell size, otherwise every random draw buries a site
    unit = (abs(np.linalg.det(G)) / k) ** (2.0 / n)
    w = np.zeros(k)
    w[1:] = x[pos:pos + (k - 1)] * (0.25 * unit)
    w -= w.mean()
    return G, T, w


def params_from_lattice(L, H, *, drop: int | None = None) -> tuple[np.ndarray, int]:
    """CMA start vector reproducing a Voronoi colouring (optionally minus a site)."""
    L, H = np.asarray(L, float), np.asarray(H, float)
    G = H @ L
    T = pc.transversal(L, H)
    if drop is not None:
        T = np.delete(T, drop, axis=0)
    T = T - T[0]                                       # pin the first site at 0
    k, n = T.shape
    frac = T @ np.linalg.inv(G)
    chol = np.linalg.cholesky(G @ G.T)                 # lower triangular, same lattice
    chol /= abs(np.linalg.det(chol)) ** (1.0 / n)
    tri = []
    for i in range(n):
        for j in range(i + 1):
            tri.append(math.log(chol[i, i]) / 0.35 if i == j else chol[i, j])
    return np.concatenate([tri, frac[1:].ravel(), np.zeros(k - 1)]), k


def n_params(n: int, k: int, fixed_gram: bool = False) -> int:
    base = 0 if fixed_gram else n * (n + 1) // 2
    return base + n * (k - 1) + (k - 1)


# --------------------------------------------------------------------------- #
# objective                                                                    #
# --------------------------------------------------------------------------- #

def objective(x, n, k, *, fixed_gram=None, penalty=5.0):
    G, T, w = unpack(np.asarray(x, float), n, k, fixed_gram=fixed_gram)
    if G is None:
        return penalty
    try:
        rep = pc.evaluate(G, T, w)
    except Exception:
        return penalty
    if not rep.sound or rep.width <= 0:
        return penalty
    return -rep.width


# --------------------------------------------------------------------------- #
# lattice baselines                                                            #
# --------------------------------------------------------------------------- #

def hermite_forms(n: int, k: int):
    """Upper-triangular Hermite normal forms of determinant ``k`` in dimension ``n``."""
    def divisors(m):
        return [d for d in range(1, m + 1) if m % d == 0]

    out = []
    if n == 2:
        for a in divisors(k):
            d = k // a
            for b in range(d):
                out.append(np.array([[a, b], [0, d]], float))
    elif n == 3:
        for a in divisors(k):
            rest = k // a
            for b in divisors(rest):
                c = rest // b
                for p in range(b):
                    for q in range(c):
                        for r in range(c):
                            out.append(np.array([[a, p, q], [0, b, r], [0, 0, c]], float))
    else:
        raise ValueError("hermite_forms implemented for n = 2, 3")
    return out


def best_lattice(L, k, n):
    """Best width over all index-``k`` sublattices of ``L`` (Voronoi scheme)."""
    best, arg = -1.0, None
    for H in hermite_forms(n, k):
        T = pc.transversal(L, H)
        rep = pc.evaluate(H @ L, T, np.zeros(len(T)))
        if rep.sound and rep.width > best:
            best, arg = rep.width, H
    return best, arg


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #

def run(n: int, k: int, *, budget: int, sigma: float, seeds: int, tag: str,
        fixed_gram=None, seed_x=None, verbose=True):
    import cma

    dim = n_params(n, k, fixed_gram is not None)
    best = (-1.0, None)
    t0 = time.time()
    for s in range(seeds):
        x0 = (np.zeros(dim) if seed_x is None and s == 0
              else (seed_x if seed_x is not None and s == 0
                    else np.random.default_rng(1000 + s).normal(0, 0.6, dim)))
        es = cma.CMAEvolutionStrategy(
            list(np.asarray(x0, float)), sigma,
            {"maxfevals": budget, "verbose": -9, "seed": 12345 + s,
             "popsize": max(10, 4 + int(3 * math.log(dim)))})
        while not es.stop():
            xs = es.ask()
            fs = [objective(x, n, k, fixed_gram=fixed_gram) for x in xs]
            es.tell(xs, fs)
            if -min(fs) > best[0]:
                best = (-min(fs), np.array(xs[int(np.argmin(fs))]))
                if verbose:
                    print(f"    seed {s}  evals {es.countevals:6d}  d = {best[0]:.6f}"
                          f"  [{time.time()-t0:.0f}s]", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--colours", type=int, nargs="+", default=[14])
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--baseline", action="store_true",
                    help="also compute the best lattice colouring for each k")
    ap.add_argument("--tag", default="power")
    args = ap.parse_args()

    out = {"dim": args.dim, "runs": []}
    for k in args.colours:
        print(f"=== n = {args.dim}, k = {k} ===", flush=True)
        entry = {"colours": k}
        if args.baseline:
            names = {2: ["A2", "Z2"], 3: ["A3*", "Z3", "D3"]}[args.dim]
            builders = {"A2": lambda: lat.A(2), "Z2": lambda: lat.Z(2),
                        "A3*": lambda: lat.Astar(3), "Z3": lambda: lat.Z(3),
                        "D3": lambda: lat.D(3)}
            entry["lattice"] = {}
            for nm in names:
                b, H = best_lattice(builders[nm](), k, args.dim)
                entry["lattice"][nm] = {"width": b, "hermite": H.tolist() if H is not None else None}
                print(f"  lattice {nm:4s}: best d = {b:.6f}", flush=True)
        d, x = run(args.dim, k, budget=args.budget, sigma=args.sigma,
                   seeds=args.seeds, tag=args.tag)
        entry["power"] = {"width": d, "x": x.tolist() if x is not None else None}
        print(f"  power   : best d = {d:.6f}", flush=True)
        out["runs"].append(entry)

    path = results_path(f"{args.tag}_n{args.dim}.json")
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
