"""Lazy prime/Radon search with a complete sublattice separation oracle.

Building the full forbidden set of A9* requires enumerating every parent
lattice vector of norm below twice the Voronoi diameter.  Most of those
vectors can never belong to one particular high-index coloring kernel.  This
campaign reverses the order:

1. start with the cheap, certainly-forbidden shell ``0 < |v| < diam``;
2. find a modular kernel avoiding the current core with ``PrimarySearch``;
3. enumerate only vectors of that kernel with ``|v| < 2 diam``;
4. add every genuine geometric conflict to the core and repeat.

The separation oracle is complete because

    dist(V, v + V) >= |v| - diam(V).

Consequently a kernel accepted by the oracle is a valid fixed-lattice
coloring even though the full parent forbidden set was never materialized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sympy import Matrix

import combigeo
from lattices import CATALOG
from prime_campaign import parse_structures
from prime_radon import (
    PrimarySearch,
    hnf_columns,
    image_size,
    kernel_basis,
    killed_mask,
    smith_diagonal,
)


EXACT_DIAMETER_RATIOS = {
    "E8": math.sqrt(2.0),
    "A9*": math.sqrt(11.0 / 3.0),
}


def parent_geometry(
    name: str,
) -> tuple[np.ndarray, float, list[tuple[list[float], float]]]:
    if name not in EXACT_DIAMETER_RATIOS:
        raise ValueError(
            f"no exact covering-radius ratio registered for {name!r}"
        )
    basis = np.asarray(CATALOG[name](), dtype=np.float64)
    shortest = float(
        np.linalg.norm(combigeo.shortest_vector(basis.tolist()))
    )
    diameter = EXACT_DIAMETER_RATIOS[name] * shortest
    facets = combigeo.relevant_facets(basis.tolist())
    return basis, diameter, facets


def lattice_coordinates(
    physical_vectors: Sequence[Sequence[float]],
    inverse_basis: np.ndarray,
) -> np.ndarray:
    if not physical_vectors:
        return np.empty((0, inverse_basis.shape[0]), dtype=np.int64)
    physical = np.asarray(physical_vectors, dtype=np.float64)
    real_coordinates = physical @ inverse_basis
    coordinates = np.rint(real_coordinates).astype(np.int64)
    error = float(np.max(np.abs(real_coordinates - coordinates)))
    if error > 2e-6:
        raise RuntimeError(f"coordinate recovery error {error:.3g}")
    return coordinates


def canonical_rows(vectors: np.ndarray) -> np.ndarray:
    """Deduplicate integer rows while retaining both signs only once."""
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for vector in np.asarray(vectors, dtype=np.int64):
        if not np.any(vector):
            continue
        positive = tuple(int(value) for value in vector)
        negative = tuple(-value for value in positive)
        key = min(positive, negative)
        unique.setdefault(key, np.asarray(key, dtype=np.int64))
    if not unique:
        return np.empty((0, vectors.shape[1]), dtype=np.int64)
    return np.asarray(list(unique.values()), dtype=np.int64)


def initial_short_core(
    basis: np.ndarray, diameter: float
) -> np.ndarray:
    """Every nonzero vector shorter than diam is certainly forbidden."""
    physical = combigeo._vectors_near(
        basis.tolist(),
        [0.0] * len(basis),
        diameter - 1e-9,
    )
    return canonical_rows(lattice_coordinates(physical, np.linalg.inv(basis)))


def separate_kernel(
    basis: np.ndarray,
    diameter: float,
    facets: Sequence[tuple[Sequence[float], float]],
    kernel_columns: np.ndarray,
) -> dict:
    """Complete geometric validation of one coloring kernel."""
    n = len(basis)
    # HNF is canonical but can be badly conditioned.  Exact integer LLL on
    # its coordinate row basis preserves the sublattice and prevents the
    # floating LLL inside combigeo from mistaking it for a degenerate basis.
    coordinate_rows = np.asarray(
        Matrix(np.asarray(kernel_columns, dtype=np.int64).T.tolist()).lll().tolist(),
        dtype=np.int64,
    )
    sub_basis = coordinate_rows @ basis
    start = time.perf_counter()
    physical = combigeo._vectors_near(
        sub_basis.tolist(),
        [0.0] * n,
        2.0 * diameter + 1e-8,
    )
    inverse = np.linalg.inv(basis)
    conflicts: list[dict] = []
    minimum_ratio = float("inf")
    checked = 0
    for raw in physical:
        vector = np.asarray(raw, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-10:
            continue
        checked += 1
        distance = 2.0 * combigeo.dist_to_halfspaces(
            (0.5 * vector).tolist(), facets
        )
        ratio = distance / diameter
        minimum_ratio = min(minimum_ratio, ratio)
        if distance < diameter - 1e-8:
            coordinate = lattice_coordinates([vector], inverse)[0]
            conflicts.append(
                {
                    "coordinate": coordinate.astype(int).tolist(),
                    "distance_ratio": float(ratio),
                    "norm_squared": float(vector @ vector),
                }
            )
    conflict_coordinates = canonical_rows(
        np.asarray(
            [item["coordinate"] for item in conflicts],
            dtype=np.int64,
        )
        if conflicts
        else np.empty((0, n), dtype=np.int64)
    )
    conflicts.sort(key=lambda item: item["distance_ratio"])
    return {
        "valid": not len(conflicts),
        "checked_kernel_vectors": checked,
        "enumerated_including_zero": len(physical),
        "minimum_distance_ratio": (
            minimum_ratio if math.isfinite(minimum_ratio) else 1.0
        ),
        "conflict_count_with_sign": len(conflicts),
        "conflict_coordinates": conflict_coordinates,
        "conflicts": conflicts,
        "seconds": time.perf_counter() - start,
    }


def merge_core(core: np.ndarray, additions: np.ndarray) -> np.ndarray:
    return canonical_rows(np.vstack((core, additions)))


def run_structure(
    basis: np.ndarray,
    diameter: float,
    facets: Sequence[tuple[Sequence[float], float]],
    initial_core: np.ndarray,
    moduli: Sequence[int],
    *,
    rounds: int,
    restarts: int,
    sweeps: int,
    top: int,
    seed: int,
    checkpoint: callable,
) -> dict:
    n = len(basis)
    core = initial_core.copy()
    rows: list[np.ndarray] | None = None
    history: list[dict] = []
    best_ratio = -1.0
    best_payload: dict | None = None
    start = time.perf_counter()
    for round_number in range(rounds):
        print(
            f"  lazy round {round_number + 1}/{rounds}: |core|={len(core)}",
            flush=True,
        )
        search = PrimarySearch(
            core,
            moduli,
            seed=seed + 104729 * round_number,
        )
        result = search.run(
            restarts=restarts,
            max_sweeps=sweeps,
            top=top,
            progress_every=max(1, restarts // 3),
            initial_rows=rows,
        )
        round_payload = {
            "round": round_number + 1,
            "core_size": len(core),
            "search": result.as_json(),
        }
        if not result.found:
            round_payload["status"] = "heuristic_core_failure"
            history.append(round_payload)
            checkpoint(history, core, best_payload, False)
            print(
                f"  core search stopped with {result.killed} remaining "
                "constraints (not an impossibility proof)",
                flush=True,
            )
            break

        rows = [row.copy() for row in result.rows]
        kernel = hnf_columns(kernel_basis(rows, moduli, n))
        determinant = abs(int(round(np.linalg.det(kernel))))
        exact_image = image_size(rows, moduli, n)
        if determinant != exact_image:
            raise AssertionError(
                f"kernel determinant {determinant} != image {exact_image}"
            )
        separation = separate_kernel(
            basis, diameter, facets, kernel
        )
        ratio = float(separation["minimum_distance_ratio"])
        round_payload.update(
            {
                "status": (
                    "certified_valid"
                    if separation["valid"]
                    else "counterexamples_added"
                ),
                "kernel_basis_columns": kernel.astype(int).tolist(),
                "kernel_determinant": determinant,
                "kernel_smith": smith_diagonal(kernel),
                "separation": {
                    key: value
                    for key, value in separation.items()
                    if key != "conflict_coordinates"
                },
            }
        )
        history.append(round_payload)
        if ratio > best_ratio:
            best_ratio = ratio
            best_payload = round_payload
        print(
            f"    oracle: checked={separation['checked_kernel_vectors']} "
            f"conflicts={separation['conflict_count_with_sign']} "
            f"min-ratio={ratio:.12f} "
            f"time={separation['seconds']:.2f}s",
            flush=True,
        )
        checkpoint(history, core, best_payload, bool(separation["valid"]))
        if separation["valid"]:
            return {
                "status": "certified_valid",
                "history": history,
                "best": best_payload,
                "core": core,
                "seconds": time.perf_counter() - start,
            }
        previous_size = len(core)
        core = merge_core(core, separation["conflict_coordinates"])
        if len(core) == previous_size:
            raise AssertionError("separation oracle produced no new core row")
    return {
        "status": "budget_exhausted",
        "history": history,
        "best": best_payload,
        "core": core,
        "seconds": time.perf_counter() - start,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice")
    parser.add_argument("structures", type=parse_structures)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--top", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    basis, diameter, facets = parent_geometry(args.lattice)
    core = initial_short_core(basis, diameter)
    payload = {
        "method": "lazy prime/Radon search with complete kernel separation",
        "lattice": args.lattice,
        "n": len(basis),
        "diameter": diameter,
        "facet_count": len(facets),
        "initial_core_size": len(core),
        "structures": args.structures,
        "budget": {
            "rounds": args.rounds,
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "seed": args.seed,
        },
        "results": [],
    }
    print(
        f"{args.lattice}: n={len(basis)} diam={diameter:.12g} "
        f"facets={len(facets)} short-core={len(core)}",
        flush=True,
    )

    campaign_start = time.perf_counter()
    for structure_number, moduli in enumerate(args.structures):
        print(
            f"\n[{structure_number + 1}/{len(args.structures)}] "
            f"moduli={moduli} product={math.prod(moduli)}",
            flush=True,
        )
        record: dict = {
            "moduli": list(moduli),
            "target_product": math.prod(moduli),
        }
        payload["results"].append(record)

        def checkpoint(
            history: list[dict],
            current_core: np.ndarray,
            best: dict | None,
            valid: bool,
        ) -> None:
            record.update(
                {
                    "status": "certified_valid" if valid else "running",
                    "rounds_completed": len(history),
                    "current_core_size": len(current_core),
                    "history": history,
                    "best": best,
                }
            )
            args.output.write_text(json.dumps(payload, indent=2) + "\n")

        outcome = run_structure(
            basis,
            diameter,
            facets,
            core,
            moduli,
            rounds=args.rounds,
            restarts=args.restarts,
            sweeps=args.sweeps,
            top=args.top,
            seed=args.seed + 1_000_003 * structure_number,
            checkpoint=checkpoint,
        )
        record.update(
            {
                "status": outcome["status"],
                "rounds_completed": len(outcome["history"]),
                "current_core_size": len(outcome["core"]),
                "history": outcome["history"],
                "best": outcome["best"],
                "seconds": outcome["seconds"],
            }
        )
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        if outcome["status"] == "certified_valid":
            print("*** CERTIFIED FIXED-LATTICE COLORING FOUND ***", flush=True)

    payload["campaign_seconds"] = time.perf_counter() - campaign_start
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
