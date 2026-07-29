"""Targeted A9* campaign replacing the ABPR quotient factor 71 by 70.

The published index-17253 kernel has Smith factors

    3, 3, 3, 3, 213 = 3^5 * 71.

Rather than search an unrelated group of order 17010, this campaign freezes
the five exact F_3 annihilators of the published C9 matrix and searches only
for three characters modulo 2, 5, and 7.  Their product replaces the final
factor 71 by 70.  Constraints already separated by the fixed F_3 block are
discarded, and every proposed full kernel is checked by the complete lazy
geometry oracle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import combigeo
from lazy_prime_campaign import (
    canonical_rows,
    initial_short_core,
    merge_core,
    parent_geometry,
    separate_kernel,
)
from prime_radon import (
    hnf_columns,
    image_size,
    kernel_basis,
    killed_mask,
    nullspace_mod,
    smith_diagonal,
)


C9 = np.asarray(
    [
        [0, 0, -3, 1, 0, 0, -1, 1, 0],
        [1, 0, -3, 1, 1, 0, -1, 4, 1],
        [0, 0, -2, 1, 0, -1, -1, 1, 3],
        [0, 0, -3, 4, 0, 0, -1, 1, 0],
        [0, 3, -3, 1, 0, 0, -1, 1, 0],
        [3, 0, -3, 1, 0, 0, 2, 1, 0],
        [0, 0, -4, 2, 0, 3, -1, 2, 0],
        [0, 0, -3, 1, 3, 0, -1, 1, 0],
        [-1, 0, -3, 1, -1, 1, 1, 1, -1],
    ],
    dtype=np.int64,
)


def full_payload(
    rows: list[np.ndarray],
    moduli: list[int],
    kernel: np.ndarray,
    separation: dict,
    seed: int,
) -> dict:
    return {
        "seed": seed,
        "moduli": moduli,
        "rows": [row.astype(int).tolist() for row in rows],
        "image_index": image_size(rows, moduli, 9),
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "separation": {
            key: value
            for key, value in separation.items()
            if key != "conflict_coordinates"
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--solver-restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=917010)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    basis, diameter, facets = parent_geometry("A9*")
    fixed_rows = [
        np.asarray(row, dtype=np.int64)
        for row in nullspace_mod(C9.T, 3)
    ]
    if len(fixed_rows) != 5:
        raise AssertionError("the ABPR C9 matrix must have five F_3 annihilators")
    fixed_moduli = [3] * 5
    variable_moduli = [2, 5, 7]
    full_moduli = fixed_moduli + variable_moduli
    target = 17010

    short_core = initial_short_core(basis, diameter)
    residual = short_core[
        killed_mask(short_core, fixed_rows, fixed_moduli)
    ]
    core = canonical_rows(residual)
    print(
        f"A9*: fixed F3-rank=5 short-core={len(short_core)} "
        f"residual-core={len(core)} target={target}",
        flush=True,
    )

    payload: dict = {
        "method": (
            "ABPR-primary continuation: freeze five F3 characters and "
            "replace F71 by F2 x F5 x F7 with multi-candidate lazy separation"
        ),
        "lattice": "A9*",
        "n": 9,
        "diameter": diameter,
        "facet_count": len(facets),
        "published_index": 17253,
        "target_index": target,
        "fixed_moduli": fixed_moduli,
        "fixed_rows": [row.astype(int).tolist() for row in fixed_rows],
        "variable_moduli": variable_moduli,
        "initial_short_core": len(short_core),
        "initial_residual_core": len(core),
        "budget": {
            "rounds": args.rounds,
            "candidates_per_round": args.candidates,
            "steps": args.steps,
            "solver_restarts": args.solver_restarts,
            "seed": args.seed,
        },
        "history": [],
        "best": None,
        "status": "running",
    }
    best: dict | None = None
    best_ratio = -1.0
    start = time.perf_counter()

    for round_number in range(args.rounds):
        additions: list[np.ndarray] = []
        summaries: list[dict] = []
        round_best = -1.0
        round_start = time.perf_counter()
        for candidate_number in range(args.candidates):
            seed = (
                args.seed
                + 104729 * round_number
                + 1009 * candidate_number
            )
            found, raw_rows, index = combigeo.min_conflicts(
                core.tolist(),
                variable_moduli,
                9,
                args.steps,
                args.solver_restarts,
                seed,
            )
            if not found or index != 70:
                summaries.append(
                    {
                        "candidate": candidate_number + 1,
                        "seed": seed,
                        "variable_image": int(index),
                        "core_valid": bool(found),
                    }
                )
                continue
            variable_rows = [
                np.asarray(row, dtype=np.int64) for row in raw_rows
            ]
            rows = fixed_rows + variable_rows
            if image_size(rows, full_moduli, 9) != target:
                raise AssertionError("combined image does not have order 17010")
            kernel = hnf_columns(kernel_basis(rows, full_moduli, 9))
            separation = separate_kernel(
                basis, diameter, facets, kernel
            )
            conflicts = separation["conflict_coordinates"]
            if len(conflicts) and not bool(
                np.all(
                    killed_mask(conflicts, fixed_rows, fixed_moduli)
                )
            ):
                raise AssertionError("a full-kernel conflict escaped fixed rows")
            additions.append(conflicts)
            ratio = float(separation["minimum_distance_ratio"])
            round_best = max(round_best, ratio)
            summaries.append(
                {
                    "candidate": candidate_number + 1,
                    "seed": seed,
                    "variable_image": int(index),
                    "core_valid": True,
                    "valid": bool(separation["valid"]),
                    "checked_kernel_vectors": separation[
                        "checked_kernel_vectors"
                    ],
                    "conflicts_with_sign": separation[
                        "conflict_count_with_sign"
                    ],
                    "minimum_distance_ratio": ratio,
                    "oracle_seconds": separation["seconds"],
                }
            )
            candidate = full_payload(
                rows, full_moduli, kernel, separation, seed
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best = candidate
            if separation["valid"]:
                payload["status"] = "certified_valid"
                best = candidate
                break

        history_record = {
            "round": round_number + 1,
            "core_size_before": len(core),
            "candidate_summaries": summaries,
            "round_best_ratio": round_best,
            "seconds": time.perf_counter() - round_start,
        }
        payload["history"].append(history_record)
        payload["best"] = best
        payload["rounds_completed"] = round_number + 1
        payload["elapsed_seconds"] = time.perf_counter() - start
        print(
            f"  round {round_number + 1}/{args.rounds}: "
            f"core={len(core)} round-best={round_best:.12f} "
            f"global-best={best_ratio:.12f} "
            f"time={history_record['seconds']:.2f}s",
            flush=True,
        )
        if payload["status"] == "certified_valid":
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print("*** CERTIFIED A9* INDEX-17010 COLORING FOUND ***", flush=True)
            break
        nonempty = [addition for addition in additions if len(addition)]
        if not nonempty:
            payload["status"] = "heuristic_core_failure"
            break
        core = merge_core(core, np.vstack(nonempty))
        payload["current_residual_core_size"] = len(core)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        payload["status"] = "budget_exhausted"

    payload["elapsed_seconds"] = time.perf_counter() - start
    payload["current_residual_core_size"] = len(core)
    payload["final_residual_core_coordinates_up_to_sign"] = core.astype(
        int
    ).tolist()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
