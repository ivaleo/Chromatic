"""Symmetry-orbit CEGAR for A9* lattice colorings.

A pointwise lazy cut excludes one forbidden vector from the next modular
kernel.  The A9* parent lattice has the full S_10 coordinate-permutation
symmetry, so a geometric conflict certifies an entire orbit of equally
forbidden vectors.  Adding one or two worst unseen orbits per round replaces
thousands of pointwise CEGAR iterations while remaining exact.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import combigeo
from chromatic_research.core.lazy_prime_campaign import (
    canonical_rows,
    initial_short_core,
    merge_core,
    parent_geometry,
    separate_kernel,
)
from chromatic_research.campaigns.prime_campaign import parse_structures
from chromatic_research.core.prime_radon import (
    PrimarySearch,
    hnf_columns,
    image_size,
    kernel_basis,
    smith_diagonal,
)
from chromatic_research.campaigns.symlat import An_star_ambient, to_ambient_int


def distinct_permutations(values: Sequence[int]) -> Iterable[tuple[int, ...]]:
    counter = collections.Counter(int(value) for value in values)
    keys = sorted(counter)
    length = len(values)
    prefix = [0] * length

    def visit(position: int) -> Iterable[tuple[int, ...]]:
        if position == length:
            yield tuple(prefix)
            return
        for value in keys:
            if counter[value] == 0:
                continue
            counter[value] -= 1
            prefix[position] = value
            yield from visit(position + 1)
            counter[value] += 1

    yield from visit(0)


def ambient_key(
    coordinate: np.ndarray, ambient_basis: np.ndarray
) -> tuple[int, ...]:
    ambient = tuple(
        int(value) for value in to_ambient_int(coordinate, ambient_basis)
    )
    positive = tuple(sorted(ambient))
    negative = tuple(sorted(-value for value in ambient))
    return min(positive, negative)


def orbit_coordinates(
    coordinate: np.ndarray, ambient_basis: np.ndarray
) -> np.ndarray:
    ambient = to_ambient_int(coordinate, ambient_basis)
    rows = []
    for permutation in distinct_permutations(ambient):
        values = np.asarray(permutation, dtype=np.int64)
        differences = values[:-1] - values[1:]
        if np.any(differences % 10):
            raise AssertionError("ambient A9* orbit left the weight lattice")
        rows.append(differences // 10)
    return canonical_rows(np.asarray(rows, dtype=np.int64))


def orbit_size_from_key(key: Sequence[int]) -> int:
    result = math.factorial(len(key))
    for multiplicity in collections.Counter(key).values():
        result //= math.factorial(multiplicity)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structures", type=parse_structures)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--cuts-per-round", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument(
        "--backend",
        choices=("radon", "cxx"),
        default="radon",
        help="use C++ coefficient moves when a full projective pool is too large",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--solver-restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if len(args.structures) != 1:
        parser.error("run one structure at a time")
    moduli = args.structures[0]
    basis, diameter, facets = parent_geometry("A9*")
    ambient_basis = An_star_ambient(9)
    core = initial_short_core(basis, diameter)
    rows: list[np.ndarray] | None = None
    seen_orbits: set[tuple[int, ...]] = set()
    payload: dict = {
        "method": "S10 symmetry-orbit lazy CEGAR with prime/Radon oracle",
        "lattice": "A9*",
        "n": 9,
        "moduli": moduli,
        "target_product": math.prod(moduli),
        "diameter": diameter,
        "facet_count": len(facets),
        "initial_core_size": len(core),
        "budget": {
            "rounds": args.rounds,
            "cuts_per_round": args.cuts_per_round,
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "backend": args.backend,
            "steps": args.steps,
            "solver_restarts": args.solver_restarts,
            "seed": args.seed,
        },
        "history": [],
        "best": None,
        "status": "running",
    }
    best_ratio = -1.0
    start = time.perf_counter()
    print(
        f"A9*: moduli={moduli} product={math.prod(moduli)} "
        f"initial-core={len(core)}",
        flush=True,
    )

    for round_number in range(args.rounds):
        round_start = time.perf_counter()
        search_seed = args.seed + 104729 * round_number
        if args.backend == "radon":
            search = PrimarySearch(
                core,
                moduli,
                seed=search_seed,
            )
            result = search.run(
                restarts=args.restarts,
                max_sweeps=args.sweeps,
                top=args.top,
                progress_every=max(1, args.restarts // 4),
                initial_rows=rows,
            )
            found = result.found
            candidate_rows = [row.copy() for row in result.rows]
            image_index = result.image_index
            search_payload = result.as_json()
        else:
            found_raw, raw_rows, raw_index = combigeo.min_conflicts(
                core.tolist(),
                moduli,
                9,
                args.steps,
                args.solver_restarts,
                search_seed,
            )
            candidate_rows = [
                np.asarray(row, dtype=np.int64) for row in raw_rows
            ]
            image_index = int(raw_index)
            found = bool(found_raw and image_index == math.prod(moduli))
            search_payload = {
                "found": found,
                "moduli": moduli,
                "rows": [
                    row.astype(int).tolist() for row in candidate_rows
                ],
                "image_index": image_index,
                "seed": search_seed,
                "steps": args.steps,
                "solver_restarts": args.solver_restarts,
            }
        if not found:
            payload["status"] = "heuristic_core_failure"
            payload["history"].append(
                {
                    "round": round_number + 1,
                    "core_size": len(core),
                    "search": search_payload,
                }
            )
            break
        rows = candidate_rows
        if image_size(rows, moduli, 9) != math.prod(moduli):
            raise AssertionError("candidate image is smaller than target")
        kernel = hnf_columns(kernel_basis(rows, moduli, 9))
        separation = separate_kernel(
            basis, diameter, facets, kernel
        )
        ratio = float(separation["minimum_distance_ratio"])
        candidate = {
            "round": round_number + 1,
            "rows": [row.astype(int).tolist() for row in rows],
            "image_index": image_index,
            "kernel_basis_columns": kernel.astype(int).tolist(),
            "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
            "kernel_smith": smith_diagonal(kernel),
            "separation": {
                key: value
                for key, value in separation.items()
                if key != "conflict_coordinates"
            },
        }
        if ratio > best_ratio:
            best_ratio = ratio
            payload["best"] = candidate
        if separation["valid"]:
            payload["status"] = "certified_valid"
            payload["history"].append(
                {
                    "round": round_number + 1,
                    "core_size_before": len(core),
                    "candidate": candidate,
                    "added_orbits": [],
                    "seconds": time.perf_counter() - round_start,
                }
            )
            print("*** CERTIFIED A9* COLORING FOUND ***", flush=True)
            break

        additions: list[np.ndarray] = []
        added_records: list[dict] = []
        for conflict in separation["conflicts"]:
            coordinate = np.asarray(
                conflict["coordinate"], dtype=np.int64
            )
            key = ambient_key(coordinate, ambient_basis)
            if key in seen_orbits:
                continue
            orbit = orbit_coordinates(coordinate, ambient_basis)
            expected = orbit_size_from_key(key)
            # Canonical +/- can at most halve the raw S10 orbit.
            if not (len(orbit) == expected or 2 * len(orbit) == expected):
                raise AssertionError(
                    f"unexpected orbit size {len(orbit)} from {expected}"
                )
            # Check representative and a few distributed points numerically.
            for test in orbit[
                np.linspace(
                    0, len(orbit) - 1, min(5, len(orbit)), dtype=int
                )
            ]:
                vector = test @ basis
                test_ratio = (
                    2.0
                    * combigeo.dist_to_halfspaces(
                        (0.5 * vector).tolist(), facets
                    )
                    / diameter
                )
                if not math.isclose(
                    test_ratio,
                    float(conflict["distance_ratio"]),
                    rel_tol=2e-8,
                    abs_tol=2e-8,
                ):
                    raise AssertionError("S10 orbit did not preserve distance")
            seen_orbits.add(key)
            additions.append(orbit)
            added_records.append(
                {
                    "representative": coordinate.astype(int).tolist(),
                    "distance_ratio": conflict["distance_ratio"],
                    "ambient_multiset": list(key),
                    "raw_orbit_size": expected,
                    "core_rows_up_to_sign": len(orbit),
                }
            )
            if len(added_records) >= args.cuts_per_round:
                break
        if not additions:
            payload["status"] = "orbit_core_stalled"
            break
        before = len(core)
        core = merge_core(core, np.vstack(additions))
        history = {
            "round": round_number + 1,
            "core_size_before": before,
            "core_size_after": len(core),
            "candidate": candidate,
            "added_orbits": added_records,
            "seconds": time.perf_counter() - round_start,
        }
        payload["history"].append(history)
        payload["rounds_completed"] = round_number + 1
        payload["current_core_size"] = len(core)
        payload["elapsed_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"round {round_number + 1}/{args.rounds}: "
            f"ratio={ratio:.12f} conflicts="
            f"{separation['conflict_count_with_sign']} "
            f"orbits={len(added_records)} core={before}->{len(core)} "
            f"time={history['seconds']:.1f}s",
            flush=True,
        )
    else:
        payload["status"] = "budget_exhausted"

    payload["rounds_completed"] = len(payload["history"])
    payload["current_core_size"] = len(core)
    payload["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL status={payload['status']} best={best_ratio:.12f} "
        f"core={len(core)} saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
