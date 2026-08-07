"""Dimension 7 by lamination of the Eisenstein colouring ``E6*/343``: index 1029.

``chi(R^7) <= 1323`` is the project's record (generic rational lattice).  The
lamination arithmetic said dimension 9 was the only beneficiary because the
lift costs ``m >= 4`` and ``4 * 343 = 1372 > 1323``.  But that ``m >= 4`` came
from the *E8* experiment: at ``m = 3`` over ``E8`` the layer vectors demand
``t >= 1.15`` while the rigorous diameter budget is ``t <= 1``.  Both numbers
are scale-invariant functions of the base (``d' = sqrt(7/6)``,
``rho = sqrt(2)`` -- identical for ``E6*``), yet the *hole structure* of
``E6*`` is different from ``E8``'s, so the layer-vector demand may differ.
If layers clear at some ``t <= 1``, then ``3 * 343 = 1029 < 1323`` -- a new
record for ``R^7``.  This campaign measures exactly that.

Everything is normalised to ``lambda1(E6*)^2 = 3`` so that all thresholds
match the dimension-9 campaign: ``diam(E6*) = sqrt(6)``, horizontal floor
``D = sqrt(7)`` (rigorous by the planar theorem / (P2)), rigorous regime
``t <= 1``.

Usage::

    python -m chromatic_research.campaigns.dim7_laminate_e6s refine --height 0.95
    python -m chromatic_research.campaigns.dim7_laminate_e6s sweep
    python -m chromatic_research.campaigns.dim7_laminate_e6s verify --config ...
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sympy import Matrix

import combigeo
from chromatic_research.core.lamination import (
    Lamination, deep_hole, enumerate_upto, kernel_rows, min_separation, unit_facets,
)
from chromatic_research.core.prime_radon import nullspace_mod
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)
THETA = OMEGA - OMEGA.conjugate()
ALPHA = 3 + OMEGA                          # |alpha|^2 = 7


def _realify(vectors) -> np.ndarray:
    out = []
    for vector in vectors:
        row = []
        for z in vector:
            row += [z.real, z.imag]
        out.append(row)
    return np.asarray(out, dtype=float)


def e6star_geometry() -> tuple[np.ndarray, float, np.ndarray]:
    """``E6*`` basis normalised to ``lambda1^2 = 3``, its exact covering radius,
    and the three mod-7 characters of ``(3+omega) E6*``.

    Complex ``E6 = {x in Z[omega]^3 : x1 = x2 = x3 mod theta}`` with Z-basis
    ``{u_i, omega u_i}``; ``E6*`` is the real dual.  The covering radius is
    exact: ``2R/lambda1 = sqrt(2)`` for ``E6*``.
    """
    generators = []
    for u in ((THETA, 0, 0), (1, 1, 1), (0, THETA, 0)):
        generators.append(u)
        generators.append(tuple(OMEGA * z for z in u))
    b_e6 = _realify(generators)
    b_dual = np.linalg.inv(b_e6).T
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(b_dual.tolist())))
    basis = b_dual * (math.sqrt(3.0) / lam1)

    alpha_block = np.array([[ALPHA.real, -ALPHA.imag], [ALPHA.imag, ALPHA.real]])
    a_alpha = np.kron(np.eye(3), alpha_block)
    c_real = basis @ a_alpha.T @ np.linalg.inv(basis)
    c_int = np.rint(c_real).astype(np.int64)
    assert np.max(np.abs(c_real - c_int)) < 1e-9, "alpha must preserve E6*"
    assert abs(int(Matrix(c_int.tolist()).det())) == 343

    for candidate in (nullspace_mod(c_int, 7), nullspace_mod(c_int.T, 7)):
        candidate = np.asarray(candidate, dtype=np.int64)
        if len(candidate) == 3 and np.all((c_int @ candidate.T) % 7 == 0):
            rows = candidate
            break
    else:
        raise AssertionError("could not recover the characters of (3+omega)E6*")

    radius = math.sqrt(3.0) / math.sqrt(2.0)        # exact: 2R/lambda1 = sqrt(2)
    return basis, radius, rows


def kernels_for(rows: np.ndarray, modulus: int, seed: int, pool_size: int = 40):
    rng = np.random.default_rng(seed)
    pool = [np.zeros(3, np.int64)] + [rng.integers(0, 7, 3) for _ in range(pool_size)]
    seen, kernels = set(), []
    for glue in pool:
        key = tuple(int(x) for x in glue)
        if key in seen:
            continue
        seen.add(key)
        kernel = kernel_rows(rows, [7] * 3, glue, modulus)
        if abs(round(np.linalg.det(kernel))) == 343 * modulus:
            kernels.append((key, kernel))
    return kernels


def refine_offset(modulus: int, height: float, seed: int, budget: float,
                  output: Path, *, sigma: float = 0.4, popsize: int = 12,
                  rigorous: bool = False, pool_size: int = 40,
                  start_offset=None) -> dict | None:
    """CMA over the 6-dimensional layer offset at a fixed height (cf. dim9).

    With ``rigorous=True`` the denominator is the (P1) diameter bound -- a
    constant -- so any ratio ``>= 1`` is immediately a rigorous claim (the
    trick that upgraded 9604 to a strict interval).  At ``t = 1`` the bound is
    exactly ``sqrt(7)`` and the objective is purely the layered separation.
    """
    import cma

    basis, radius, rows = e6star_geometry()
    if start_offset is not None:
        hole = np.asarray(start_offset, dtype=float)
    else:
        _, hole = deep_hole(basis, n_dirs=400, seed=11)
    kernels = kernels_for(rows, modulus, seed, pool_size=pool_size)
    print(f"E6* base: {len(kernels)} admissible glues at m={modulus}, "
          f"index {343 * modulus}", flush=True)

    def evaluate(offset, n_dirs=150):
        lam = Lamination(basis, radius, np.asarray(offset, float), height)
        diameter = lam.safe_diameter if rigorous \
            else lam.measured_diameter(n_dirs=n_dirs)
        facets = unit_facets(lam.basis)
        best = (-1.0, None)
        for glue, kernel in kernels:
            ratio = min_separation(kernel @ lam.basis, diameter, facets) / diameter
            if ratio > best[0]:
                best = (ratio, glue)
        return best[0], best[1], diameter

    start = time.time()
    ratio, glue, diameter = evaluate(hole, n_dirs=400)
    print(f"t={height} index={343 * modulus}: deep-hole start d={ratio:.7f} "
          f"diam={diameter:.6f} (rigorous {math.sqrt(6 + height**2):.6f}, "
          f"threshold sqrt7={math.sqrt(7):.6f})", flush=True)
    strategy = cma.CMAEvolutionStrategy(list(map(float, hole)), sigma,
                                        {"popsize": popsize, "verbose": -9, "seed": seed})
    best, best_offset, evaluations, last_gain = ratio, np.asarray(hole, float), 0, time.time()
    record = {"index": 343 * modulus, "modulus": modulus, "height": height,
              "seed": seed, "ratio_measured": ratio,
              "glue": list(glue) if glue else None,
              "offset": [float(x) for x in hole],
              "diameter_measured": diameter, "evaluations": 0}
    output.write_text(json.dumps(record, indent=1) + "\n")
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
                record = {"index": 343 * modulus, "modulus": modulus, "height": height,
                          "seed": seed, "ratio_measured": value,
                          "glue": list(candidate_glue), "offset": best_offset.tolist(),
                          "diameter_measured": candidate_diameter,
                          "evaluations": evaluations}
                output.write_text(json.dumps(record, indent=1) + "\n")
                print(f"  eval {evaluations}: d={value:.7f} diam={candidate_diameter:.6f} "
                      f"glue={candidate_glue} [{time.time() - start:.0f}s]"
                      f"{'  >>> crossed 1' if value >= 1 else ''}", flush=True)
        strategy.tell(candidates, losses)
        if evaluations % 48 < popsize:
            print(f"    [heartbeat] {evaluations} evaluations, best {best:.7f}, "
                  f"{time.time() - start:.0f}s", flush=True)
    record["ratio_final"] = best
    output.write_text(json.dumps(record, indent=1) + "\n")
    return record


def sweep(modulus: int, seed: int, heights, budget_each: float) -> list[dict]:
    results = []
    for height in heights:
        out = results_path(f"dim7_laminate_m{modulus}_t{height:.2f}.json")
        record = refine_offset(modulus, float(height), seed, budget_each, out)
        if record:
            results.append(record)
    return results


def verify(config: dict, *, n_dirs: int = 3000) -> dict:
    """Re-check against the RIGOROUS (P1) diameter; double projections (cf. dim9)."""
    basis, radius, rows = e6star_geometry()
    offset = np.asarray(config["offset"], dtype=float)
    height = float(config["height"])
    lam = Lamination(basis, radius, offset, height)
    safe = lam.safe_diameter
    measured = lam.measured_diameter(n_dirs=n_dirs)
    kernel = kernel_rows(rows, [7] * 3, config["glue"], config["modulus"])
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
    horizontal = np.abs(vectors[:, 6]) < 1e-9
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
        "valid_rigorous": bool(separation / safe >= 1.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("refine")
    r.add_argument("--modulus", type=int, default=3)
    r.add_argument("--height", type=float, required=True)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--budget", type=float, default=1500.0)
    r.add_argument("--rigorous", action="store_true",
                   help="use the (P1) diameter in the denominator; ratio >= 1 is a claim")
    r.add_argument("--pool-size", type=int, default=40)
    r.add_argument("--start-config", type=Path,
                   help="seed the CMA with the offset stored in this JSON")
    r.add_argument("--output", type=Path)
    s = sub.add_parser("sweep")
    s.add_argument("--modulus", type=int, default=3)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--budget-each", type=float, default=1200.0)
    s.add_argument("--heights", type=float, nargs="+",
                   default=[1.00, 0.95, 0.90, 1.05, 1.10, 1.15, 1.20])
    v = sub.add_parser("verify")
    v.add_argument("--config", type=Path, required=True)
    v.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "refine":
        out = args.output or results_path(
            f"dim7_laminate_m{args.modulus}_t{args.height:.2f}.json")
        start_offset = None
        if args.start_config:
            start_offset = json.loads(args.start_config.read_text())["offset"]
        record = refine_offset(args.modulus, args.height, args.seed, args.budget, out,
                               rigorous=args.rigorous, pool_size=args.pool_size,
                               start_offset=start_offset)
        print(json.dumps(record, indent=1))
        return 0 if record and record.get("ratio_final", 0) >= 1.0 else 2

    if args.command == "sweep":
        results = sweep(args.modulus, args.seed, args.heights, args.budget_each)
        best = max(results, key=lambda r: r.get("ratio_final", r["ratio_measured"])) \
            if results else None
        summary = results_path(f"dim7_laminate_m{args.modulus}_sweep.json")
        summary.write_text(json.dumps({"runs": results, "best": best}, indent=1) + "\n")
        print(json.dumps(best, indent=1))
        return 0 if best and best.get("ratio_final", 0) >= 1.0 else 2

    config = json.loads(args.config.read_text())
    report = verify(config)
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nINDEX {report['exact_index']}  "
          f"d(rigorous diam)={report['ratio_rigorous_diameter']:.9f}  "
          f"d(measured diam)={report['ratio_measured_diameter']:.9f}  "
          f"{'RIGOROUS VALID' if report['valid_rigorous'] else 'not rigorous'}")
    return 0 if report["valid_rigorous"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
