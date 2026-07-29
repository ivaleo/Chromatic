"""Multi-candidate lazy coloring search using the C++ coefficient solver.

This complements ``lazy_prime_campaign.py`` for structures containing a
large-prime block whose complete projective/Radon pool is too large (notably
F_7^9).  In every cutting-plane round several independent core-valid kernels
are generated, all are checked by the complete sublattice geometry oracle,
and the union of their counterexamples is added to the core.  The geometrically
best kernel is retained instead of accepting the first algebraic solution.
"""

from __future__ import annotations

import argparse
import json
import math
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
from prime_campaign import parse_structures
from prime_radon import (
    hnf_columns,
    image_size,
    kernel_basis,
    smith_diagonal,
)


def candidate_payload(
    rows: list[np.ndarray],
    moduli: Sequence[int],
    kernel: np.ndarray,
    separation: dict,
    *,
    seed: int,
) -> dict:
    return {
        "seed": seed,
        "moduli": list(moduli),
        "rows": [row.astype(int).tolist() for row in rows],
        "image_index": image_size(rows, moduli, len(rows[0])),
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
    parser.add_argument("lattice")
    parser.add_argument("structures", type=parse_structures)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--solver-restarts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    basis, diameter, facets = parent_geometry(args.lattice)
    initial_core = initial_short_core(basis, diameter)
    payload: dict = {
        "method": (
            "multi-candidate lazy C++ min-conflicts with complete "
            "kernel separation"
        ),
        "lattice": args.lattice,
        "n": len(basis),
        "diameter": diameter,
        "facet_count": len(facets),
        "initial_core_size": len(initial_core),
        "budget": {
            "rounds": args.rounds,
            "candidates_per_round": args.candidates,
            "steps": args.steps,
            "solver_restarts": args.solver_restarts,
            "seed": args.seed,
        },
        "results": [],
    }
    print(
        f"{args.lattice}: n={len(basis)} diam={diameter:.12g} "
        f"facets={len(facets)} short-core={len(initial_core)}",
        flush=True,
    )
    campaign_start = time.perf_counter()

    for structure_number, moduli in enumerate(args.structures):
        target = math.prod(moduli)
        core = initial_core.copy()
        history: list[dict] = []
        global_best: dict | None = None
        global_best_ratio = -1.0
        status = "budget_exhausted"
        structure_start = time.perf_counter()
        record = {
            "moduli": list(moduli),
            "target_product": target,
            "status": "running",
            "history": history,
        }
        payload["results"].append(record)
        print(
            f"\n[{structure_number + 1}/{len(args.structures)}] "
            f"moduli={moduli} product={target}",
            flush=True,
        )

        for round_number in range(args.rounds):
            additions: list[np.ndarray] = []
            summaries: list[dict] = []
            round_best: dict | None = None
            round_best_ratio = -1.0
            search_start = time.perf_counter()
            for candidate_number in range(args.candidates):
                candidate_seed = (
                    args.seed
                    + 10_000_019 * structure_number
                    + 104_729 * round_number
                    + 1009 * candidate_number
                )
                found, raw_rows, index = combigeo.min_conflicts(
                    core.tolist(),
                    list(moduli),
                    len(basis),
                    args.steps,
                    args.solver_restarts,
                    candidate_seed,
                )
                if not found or index != target:
                    summaries.append(
                        {
                            "candidate": candidate_number + 1,
                            "seed": candidate_seed,
                            "core_valid": bool(found),
                            "image_index": int(index),
                        }
                    )
                    continue
                rows = [
                    np.asarray(row, dtype=np.int64) for row in raw_rows
                ]
                kernel = hnf_columns(
                    kernel_basis(rows, moduli, len(basis))
                )
                determinant = abs(int(round(np.linalg.det(kernel))))
                if determinant != target:
                    raise AssertionError(
                        f"determinant {determinant} != target {target}"
                    )
                separation = separate_kernel(
                    basis, diameter, facets, kernel
                )
                ratio = float(separation["minimum_distance_ratio"])
                additions.append(separation["conflict_coordinates"])
                summary = {
                    "candidate": candidate_number + 1,
                    "seed": candidate_seed,
                    "core_valid": True,
                    "image_index": int(index),
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
                summaries.append(summary)
                full = candidate_payload(
                    rows,
                    moduli,
                    kernel,
                    separation,
                    seed=candidate_seed,
                )
                if ratio > round_best_ratio:
                    round_best_ratio = ratio
                    round_best = full
                if ratio > global_best_ratio:
                    global_best_ratio = ratio
                    global_best = full
                if separation["valid"]:
                    status = "certified_valid"
                    round_best = full
                    break

            round_record = {
                "round": round_number + 1,
                "core_size_before": len(core),
                "candidate_summaries": summaries,
                "round_best": round_best,
                "seconds": time.perf_counter() - search_start,
            }
            history.append(round_record)
            print(
                f"  round {round_number + 1}/{args.rounds}: "
                f"core={len(core)} tried={len(summaries)} "
                f"round-best={round_best_ratio:.12f} "
                f"global-best={global_best_ratio:.12f} "
                f"time={round_record['seconds']:.2f}s",
                flush=True,
            )
            if status == "certified_valid":
                record.update(
                    {
                        "status": status,
                        "rounds_completed": len(history),
                        "current_core_size": len(core),
                        "best": round_best,
                        "history": history,
                        "seconds": time.perf_counter() - structure_start,
                    }
                )
                args.output.write_text(
                    json.dumps(payload, indent=2) + "\n"
                )
                print("*** CERTIFIED FIXED-LATTICE COLORING FOUND ***", flush=True)
                break
            nonempty = [item for item in additions if len(item)]
            if not nonempty:
                status = "heuristic_core_failure"
                print("  no full-index core-valid candidate found", flush=True)
                break
            core = merge_core(core, np.vstack(nonempty))
            record.update(
                {
                    "status": "running",
                    "rounds_completed": len(history),
                    "current_core_size": len(core),
                    "best": global_best,
                    "history": history,
                    "seconds": time.perf_counter() - structure_start,
                }
            )
            args.output.write_text(json.dumps(payload, indent=2) + "\n")

        if status != "certified_valid":
            record.update(
                {
                    "status": status,
                    "rounds_completed": len(history),
                    "current_core_size": len(core),
                    "best": global_best,
                    "history": history,
                    "seconds": time.perf_counter() - structure_start,
                    "final_core_coordinates_up_to_sign": core.astype(
                        int
                    ).tolist(),
                }
            )
            args.output.write_text(json.dumps(payload, indent=2) + "\n")

    payload["campaign_seconds"] = time.perf_counter() - campaign_start
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
