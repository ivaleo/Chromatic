"""Dimension 9 by lamination of the Eisenstein colouring ``E8/2401``.

The published bound for ``chi(R^9)`` is ``17253`` (Arman-Bondarenko-Prymak-Radchenko,
arXiv:2112.13438), obtained from ``A9*``.  The Minkowski screen
(:mod:`chromatic_research.core.minkowski`) puts that index ``10.5x`` above the
volume obstruction, against ``1.4x .. 4.0x`` in every other dimension of the
project -- dimension 9 lacks a construction rather than optimisation effort.

This campaign builds one by lamination (:mod:`chromatic_research.core.lamination`):
stack layers of ``E8`` at height ``t``, offset by ``c``, and lift the ``(3+omega)``
character with a glue parameter ``a`` and a layer modulus ``m``; the index is
``2401 m``.  Search space: ``c`` (8 continuous parameters, random walk), ``t``
(grid), ``a in F_7^4`` (pool), ``m`` (given).

Usage::

    python -m chromatic_research.campaigns.dim9_laminate search --modulus 4
    python -m chromatic_research.campaigns.dim9_laminate verify \
        --config audit-data/results/dim9_laminate_m4.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sympy import Matrix

import combigeo
from chromatic_research.campaigns.e8_neighbor_search import C2401_ROWS, e8_geometry
from chromatic_research.core.lamination import (
    Lamination, deep_hole, enumerate_upto, kernel_rows, min_separation, unit_facets,
)
from chromatic_research.core.prime_radon import nullspace_mod
from chromatic_research.paths import results_path


def base_characters() -> np.ndarray:
    """Four characters mod 7 whose common kernel is ``(3+omega) E8``."""
    for candidate in (nullspace_mod(C2401_ROWS, 7), nullspace_mod(C2401_ROWS.T, 7)):
        candidate = np.asarray(candidate, dtype=np.int64)
        if len(candidate) == 4 and np.all((C2401_ROWS @ candidate.T) % 7 == 0):
            return candidate
    raise AssertionError("could not recover the characters of (3+omega)E8")


def base_geometry() -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """``E8`` basis, its covering radius, a deep hole, and the mod-7 characters.

    The hole is only a starting point for the offset search; the offset that a
    configuration actually uses is always stored explicitly, because different
    deep-hole orbits of ``E8`` give genuinely different laminations.
    """
    basis, _, _ = e8_geometry()
    radius, hole = deep_hole(basis, n_dirs=500, seed=11)
    return basis, float(radius), np.asarray(hole, dtype=float), base_characters()


def score(lam: Lamination, rows: np.ndarray, glue_pool, modulus: int,
          *, n_dirs: int = 250) -> tuple[float, tuple[int, ...] | None, float]:
    """Best ratio over the glue pool, using the measured (LP) diameter."""
    diameter = lam.measured_diameter(n_dirs=n_dirs)
    facets = unit_facets(lam.basis)
    best: tuple[float, tuple[int, ...] | None] = (-1.0, None)
    for glue in glue_pool:
        kernel = kernel_rows(rows, [7] * 4, glue, modulus)
        if abs(round(np.linalg.det(kernel))) != 2401 * modulus:
            continue
        ratio = min_separation(kernel @ lam.basis, diameter, facets) / diameter
        if ratio > best[0]:
            best = (ratio, tuple(int(x) for x in glue))
    return best[0], best[1], diameter


def search(modulus: int, seed: int, t_low: float, t_high: float,
           t_step: float, output: Path) -> dict | None:
    basis, radius, hole, rows = base_geometry()
    rng = np.random.default_rng(seed)
    pool = [np.zeros(4, np.int64)] + [rng.integers(0, 7, 4) for _ in range(40)]
    offset, incumbent, record = hole.copy(), -1.0, None
    schedule = [(0.0, 1)] + [(sigma, 14) for sigma in (0.5, 0.35, 0.22, 0.12, 0.06, 0.03)]
    for sigma, repeats in schedule:
        for _ in range(repeats):
            candidate = offset if sigma == 0 else offset + rng.standard_normal(8) * sigma
            row = (-1.0, None, None, None)
            for height in np.arange(t_low, t_high, t_step):
                lam = Lamination(basis, radius, candidate, float(height))
                ratio, glue, diameter = score(lam, rows, pool, modulus)
                if ratio > row[0]:
                    row = (ratio, glue, round(float(height), 4), diameter)
            if row[0] > incumbent:
                incumbent, offset = row[0], candidate
                safe = math.sqrt(4 * radius**2 + row[2] ** 2)
                record = {
                    "index": 2401 * modulus, "modulus": modulus, "seed": seed,
                    "ratio_measured": row[0], "glue": list(row[1]), "height": row[2],
                    "diameter_measured": row[3], "diameter_safe": safe,
                    "offset": [float(x) for x in candidate],
                    "base_covering_radius": radius, "sigma": sigma,
                }
                print(f"  sigma={sigma:.2f} d={row[0]:.6f} t={row[2]} a={row[1]} "
                      f"diam={row[3]:.5f} (safe {safe:.5f})", flush=True)
                output.write_text(json.dumps(record, indent=1) + "\n")
    return record


def refine_offset(modulus: int, height: float, seed: int, budget: float,
                  output: Path, *, sigma: float = 0.45, popsize: int = 12) -> dict | None:
    """CMA over the 8-dimensional layer offset at a fixed height.

    At index 7203 the binding constraint turns out to be the *horizontal* one --
    the ratio equals the ceiling ``sqrt(7)/diam`` to five decimals, so every
    layered vector already clears and the whole problem is shrinking the
    diameter.  A random walk over the offset stalls around ``0.989``; CMA on the
    same eight parameters crosses 1.  Lowering the height shrinks the diameter
    but tightens the layers, so the trade has to be searched height by height.
    """
    import cma

    basis, radius, hole, rows = base_geometry()
    generator = np.random.default_rng(seed)
    pool = [np.zeros(4, np.int64)] + [generator.integers(0, 7, 4) for _ in range(24)]
    kernels = []
    for glue in pool:
        kernel = kernel_rows(rows, [7] * 4, glue, modulus)
        if abs(round(np.linalg.det(kernel))) == 2401 * modulus:
            kernels.append((tuple(int(x) for x in glue), kernel))

    def evaluate(offset, n_dirs=150):
        lam = Lamination(basis, radius, np.asarray(offset, float), height)
        diameter = lam.measured_diameter(n_dirs=n_dirs)
        facets = unit_facets(lam.basis)
        best = (-1.0, None)
        for glue, kernel in kernels:
            ratio = min_separation(kernel @ lam.basis, diameter, facets) / diameter
            if ratio > best[0]:
                best = (ratio, glue)
        return best[0], best[1], diameter

    start = time.time()
    ratio, glue, diameter = evaluate(hole, n_dirs=400)
    print(f"t={height} index={2401 * modulus}: deep-hole start d={ratio:.7f} "
          f"diam={diameter:.6f}", flush=True)
    strategy = cma.CMAEvolutionStrategy(list(map(float, hole)), sigma,
                                        {"popsize": popsize, "verbose": -9, "seed": seed})
    best, best_offset, evaluations, last_gain = ratio, np.asarray(hole, float), 0, time.time()
    record = None
    while not strategy.stop():
        if time.time() - start > budget:
            print(f"  [budget] stop after {evaluations} evaluations", flush=True)
            break
        if time.time() - last_gain > 300:
            print(f"  [no progress 300s] stop after {evaluations} evaluations", flush=True)
            break
        candidates = strategy.ask()
        losses = []
        for candidate in candidates:
            value, candidate_glue, candidate_diameter = evaluate(candidate)
            evaluations += 1
            losses.append(-value)
            if value > best:
                best, best_offset, last_gain = value, np.asarray(candidate), time.time()
                record = {"index": 2401 * modulus, "modulus": modulus, "height": height,
                          "seed": seed, "ratio_measured": value, "glue": list(candidate_glue),
                          "offset": best_offset.tolist(),
                          "diameter_measured": candidate_diameter,
                          "evaluations": evaluations}
                output.write_text(json.dumps(record, indent=1) + "\n")
                print(f"  eval {evaluations}: d={value:.7f} diam={candidate_diameter:.6f} "
                      f"glue={candidate_glue} [{time.time() - start:.0f}s]"
                      f"{'  >>> crossed 1' if value >= 1 else ''}", flush=True)
        strategy.tell(candidates, losses)
        if evaluations % 24 < popsize:
            print(f"    [heartbeat] {evaluations} evaluations, best {best:.7f}, "
                  f"{time.time() - start:.0f}s", flush=True)
    return record


def saturation_test(config: dict, ladder=((800, 5), (6000, 7), (15000, 31), (15000, 97))) -> dict:
    """Does the covering-radius estimator still move when the budget grows?

    It climbs to a locally farthest vertex from each random direction and so
    converges from BELOW; a value that stops moving over a twentyfold budget
    increase and several seeds is evidence that the global maximum was reached,
    which is exactly what a measured claim rests on.
    """
    basis, radius, _, _ = base_geometry()
    lam = Lamination(basis, radius, np.asarray(config["offset"], float),
                     float(config["height"]))
    seen = []
    for n_dirs, seed in ladder:
        seen.append({"n_dirs": n_dirs, "seed": seed,
                     "diameter": lam.measured_diameter(n_dirs=n_dirs, seed=seed)})
        print(f"  n_dirs={n_dirs:6d} seed={seed:3d} "
              f"diam={seen[-1]['diameter']:.9f}", flush=True)
    largest = max(entry["diameter"] for entry in seen)
    return {"ladder": seen, "diameter_saturated": largest,
            "diameter_rigorous_P1": lam.safe_diameter,
            "threshold_sqrt7": math.sqrt(7.0),
            "undetermined": bool(largest <= math.sqrt(7.0) <= lam.safe_diameter)}


def verify(config: dict, *, n_dirs: int = 3000) -> dict:
    """Re-check a configuration against the RIGOROUS diameter bound (P1).

    Horizontal vectors are covered by (P2) and need no numerics; they are
    measured anyway.  Layered vectors are checked with two independent
    projection routines (combigeo's Dykstra and an SLSQP projection).
    """
    from scipy.optimize import minimize

    basis, radius, _, rows = base_geometry()
    offset = np.asarray(config["offset"], dtype=float)
    height = float(config["height"])
    lam = Lamination(basis, radius, offset, height)
    safe = lam.safe_diameter
    measured = lam.measured_diameter(n_dirs=n_dirs)
    kernel = kernel_rows(rows, [7] * 4, config["glue"], config["modulus"])
    facets = unit_facets(lam.basis)
    normals = np.array([f[0] for f in facets], dtype=float)
    offsets = np.array([f[1] for f in facets], dtype=float)

    def slsqp_distance(point: np.ndarray) -> float:
        if np.all(normals @ point <= offsets + 1e-12):
            return 0.0
        constraint = {"type": "ineq", "fun": lambda y: offsets - normals @ y,
                      "jac": lambda y: -normals}
        result = minimize(lambda y: float((y - point) @ (y - point)),
                          np.zeros_like(point), jac=lambda y: 2 * (y - point),
                          constraints=[constraint], method="SLSQP",
                          options={"maxiter": 500, "ftol": 1e-15})
        return float(np.linalg.norm(result.x - point))

    vectors = enumerate_upto(kernel @ lam.basis, 2 * safe)
    horizontal = np.abs(vectors[:, 8]) < 1e-9
    norms = np.linalg.norm(vectors, axis=1)

    worst_horizontal = math.inf
    for vector in vectors[horizontal]:
        worst_horizontal = min(worst_horizontal, 2 * float(
            combigeo.dist_to_halfspaces((vector / 2).tolist(), facets)))

    worst_layered, witness, checked = math.inf, None, 0
    for index in [i for i in np.argsort(norms) if not horizontal[i]]:
        if norms[index] - safe >= worst_layered:
            break
        dykstra = 2 * float(combigeo.dist_to_halfspaces((vectors[index] / 2).tolist(), facets))
        slsqp = 2 * slsqp_distance(vectors[index] / 2)
        checked += 1
        if min(dykstra, slsqp) < worst_layered:
            worst_layered = min(dykstra, slsqp)
            witness = {"vector": vectors[index].tolist(), "dykstra": dykstra, "slsqp": slsqp}

    separation = min(worst_horizontal, worst_layered)
    return {
        **config,
        "exact_index": abs(int(Matrix(kernel.tolist()).det())),
        "diameter_safe": safe, "diameter_measured": measured,
        "n_vectors": int(len(vectors)), "n_horizontal": int(horizontal.sum()),
        "n_layered_checked": checked,
        "separation_horizontal": worst_horizontal,
        "separation_layered": worst_layered,
        "separation_min": separation,
        "ratio_rigorous_diameter": separation / safe,
        "ratio_measured_diameter": separation / measured,
        "layered_witness": witness,
        "valid": bool(separation / safe >= 1.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("--modulus", type=int, default=4)
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--t-low", type=float, default=0.85)
    s.add_argument("--t-high", type=float, default=1.35)
    s.add_argument("--t-step", type=float, default=0.05)
    s.add_argument("--output", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--config", type=Path, required=True)
    v.add_argument("--output", type=Path)
    r = sub.add_parser("refine")
    r.add_argument("--modulus", type=int, default=3)
    r.add_argument("--height", type=float, required=True)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--budget", type=float, default=1200.0)
    r.add_argument("--output", type=Path)
    s = sub.add_parser("saturation")
    s.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "refine":
        out = args.output or results_path(
            f"dim9_laminate_m{args.modulus}_t{args.height}.json")
        record = refine_offset(args.modulus, args.height, args.seed, args.budget, out)
        print(json.dumps(record, indent=1))
        return 0 if record and record["ratio_measured"] >= 1.0 else 2

    if args.command == "saturation":
        report = saturation_test(json.loads(args.config.read_text()))
        print(json.dumps(report, indent=1))
        return 0

    if args.command == "search":
        out = args.output or results_path(f"dim9_laminate_m{args.modulus}.json")
        record = search(args.modulus, args.seed, args.t_low, args.t_high,
                        args.t_step, out)
        print(json.dumps(record, indent=1))
        return 0 if record else 2

    config = json.loads(args.config.read_text())
    report = verify(config)
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nINDEX {report['exact_index']}  "
          f"d(rigorous diam)={report['ratio_rigorous_diameter']:.9f}  "
          f"d(measured diam)={report['ratio_measured_diameter']:.9f}  "
          f"{'VALID' if report['valid'] else 'NOT VALID'}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
