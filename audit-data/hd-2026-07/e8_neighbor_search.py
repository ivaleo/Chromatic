"""Search determinant-2400 integer neighbours of the E8/2401 similarity.

The published coloring is multiplication by 3+omega on the Eisenstein
description of E8.  In an exact integer coordinate basis it is an 8x8 matrix
of determinant 2401 and separation ratio sqrt(7/6) > 1.

All first cofactors are divisible by 343, so changing one entry (or any
rank-one update) cannot lower the determinant by one.  This script samples
sparse higher-rank integer perturbations, filters |det|=2400 in vectorized
floating arithmetic, confirms every hit with exact integer arithmetic, and
then applies the complete E8 sublattice separation oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sympy import Matrix

import combigeo
from lazy_prime_campaign import separate_kernel
from prime_radon import smith_diagonal


BINT = np.asarray(
    [
        [3, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 2, 0, 0, 0, 1, 0],
        [1, 0, 2, 0, 0, 0, 0, 1],
    ],
    dtype=np.int64,
)


C2401_ROWS = np.asarray(
    [
        [1, 3, 0, 0, 0, 0, 0, 0],
        [-1, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 3, 0, 0, 0, 0],
        [0, 0, -1, 4, 0, 0, 0, 0],
        [-1, 1, -1, 1, 3, 1, 0, 0],
        [0, 1, 0, 1, -1, 2, 0, 0],
        [-1, 1, -2, 2, 0, 0, 3, 1],
        [0, 1, 0, 2, 0, 0, -1, 2],
    ],
    dtype=np.int64,
)


def e8_geometry() -> tuple[
    np.ndarray, float, list[tuple[list[float], float]]
]:
    transform = np.zeros((8, 8), dtype=np.float64)
    for index in range(4):
        transform[2 * index, 2 * index : 2 * index + 2] = [1.0, 0.0]
        transform[2 * index + 1, 2 * index : 2 * index + 2] = [
            -0.5,
            math.sqrt(3.0) / 2.0,
        ]
    basis = BINT @ transform
    shortest = float(
        np.linalg.norm(combigeo.shortest_vector(basis.tolist()))
    )
    diameter = math.sqrt(2.0) * shortest
    facets = combigeo.relevant_facets(basis.tolist())
    return basis, diameter, facets


def exact_det(matrix: np.ndarray) -> int:
    return int(Matrix(np.asarray(matrix, dtype=np.int64).tolist()).det())


def parse_sparsities(text: str) -> list[int]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("sparsities must be a nonempty list")
    result = [int(value) for value in raw]
    if any(value < 2 for value in result):
        raise argparse.ArgumentTypeError("every sparsity must be at least two")
    return result


def candidate_record(
    matrix_rows: np.ndarray,
    separation: dict,
    *,
    sample: int,
    nominal_sparsity: int,
) -> dict:
    perturbation = matrix_rows - C2401_ROWS
    return {
        "sample": sample,
        "nominal_sparsity": nominal_sparsity,
        "matrix_rows": matrix_rows.astype(int).tolist(),
        "kernel_basis_columns": matrix_rows.T.astype(int).tolist(),
        "determinant": exact_det(matrix_rows),
        "smith": smith_diagonal(matrix_rows.T),
        "perturbation_nonzero_entries": int(np.count_nonzero(perturbation)),
        "perturbation_frobenius_norm": float(np.linalg.norm(perturbation)),
        "separation": {
            key: value
            for key, value in separation.items()
            if key != "conflict_coordinates"
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--batch", type=int, default=20_000)
    parser.add_argument(
        "--sparsities",
        type=parse_sparsities,
        default=[4, 6, 8, 10, 12],
    )
    parser.add_argument("--seed", type=int, default=82400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    basis, diameter, facets = e8_geometry()
    baseline = separate_kernel(
        basis, diameter, facets, C2401_ROWS.T
    )
    if not baseline["valid"]:
        raise AssertionError("published E8/2401 kernel did not validate")
    print(
        f"E8 baseline: det={exact_det(C2401_ROWS)} "
        f"ratio={baseline['minimum_distance_ratio']:.12f} "
        f"facets={len(facets)}",
        flush=True,
    )

    payload: dict = {
        "method": (
            "sparse integer determinant-level neighbours of the exact "
            "E8/2401 Eisenstein similarity"
        ),
        "dimension": 8,
        "target_determinant": 2400,
        "parent_diameter": diameter,
        "parent_facet_count": len(facets),
        "source_matrix_rows": C2401_ROWS.astype(int).tolist(),
        "source_determinant": exact_det(C2401_ROWS),
        "source_separation_ratio": baseline["minimum_distance_ratio"],
        "budget": {
            "samples": args.samples,
            "batch": args.batch,
            "sparsities": args.sparsities,
            "seed": args.seed,
        },
        "exact_determinant_hits": 0,
        "unique_exact_hits": 0,
        "evaluated_candidates": 0,
        "best": None,
        "valid_candidate": None,
    }
    rng = np.random.default_rng(args.seed)
    seen: set[bytes] = set()
    best_ratio = -1.0
    start = time.perf_counter()
    samples_done = 0

    for sparsity_number, sparsity in enumerate(args.sparsities):
        target_samples = args.samples // len(args.sparsities)
        if sparsity_number < args.samples % len(args.sparsities):
            target_samples += 1
        local_done = 0
        while local_done < target_samples:
            count = min(args.batch, target_samples - local_done)
            matrices = np.repeat(
                C2401_ROWS[None, :, :], count, axis=0
            )
            positions = rng.integers(
                0, 64, size=(count, sparsity), dtype=np.int64
            )
            signs = (
                2
                * rng.integers(
                    0, 2, size=(count, sparsity), dtype=np.int64
                )
                - 1
            )
            batch_ids = np.repeat(np.arange(count), sparsity)
            np.add.at(
                matrices,
                (
                    batch_ids,
                    (positions.ravel() // 8),
                    (positions.ravel() % 8),
                ),
                signs.ravel(),
            )
            determinants = np.linalg.det(
                matrices.astype(np.float64)
            )
            rounded = np.rint(determinants).astype(np.int64)
            hit_ids = np.flatnonzero(np.abs(rounded) == 2400)
            payload["exact_determinant_hits"] += int(len(hit_ids))
            for hit_id in hit_ids:
                matrix = matrices[int(hit_id)]
                if abs(exact_det(matrix)) != 2400:
                    continue
                key = matrix.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                payload["unique_exact_hits"] = len(seen)
                try:
                    separation = separate_kernel(
                        basis, diameter, facets, matrix.T
                    )
                except Exception as error:
                    print(
                        f"  oracle skipped ill-conditioned hit: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                payload["evaluated_candidates"] += 1
                ratio = float(separation["minimum_distance_ratio"])
                record = candidate_record(
                    matrix,
                    separation,
                    sample=samples_done + int(hit_id) + 1,
                    nominal_sparsity=sparsity,
                )
                if ratio > best_ratio:
                    best_ratio = ratio
                    payload["best"] = record
                    print(
                        f"  new best: ratio={ratio:.12f} "
                        f"conflicts={separation['conflict_count_with_sign']} "
                        f"|E|={record['perturbation_nonzero_entries']} "
                        f"sample={record['sample']}",
                        flush=True,
                    )
                    args.output.write_text(
                        json.dumps(payload, indent=2) + "\n"
                    )
                if separation["valid"]:
                    payload["valid_candidate"] = record
                    payload["samples_completed"] = samples_done + int(
                        hit_id
                    ) + 1
                    payload["elapsed_seconds"] = (
                        time.perf_counter() - start
                    )
                    args.output.write_text(
                        json.dumps(payload, indent=2) + "\n"
                    )
                    print("*** VALID E8 INDEX-2400 NEIGHBOUR FOUND ***", flush=True)
                    return 0
            local_done += count
            samples_done += count
            if samples_done % max(args.batch, 200_000) == 0:
                print(
                    f"  progress samples={samples_done}/{args.samples} "
                    f"sparsity={sparsity} hits={len(seen)} "
                    f"best={best_ratio:.12f} "
                    f"elapsed={time.perf_counter()-start:.1f}s",
                    flush=True,
                )

    payload["samples_completed"] = samples_done
    payload["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL samples={samples_done} exact-hits={len(seen)} "
        f"best={best_ratio:.12f} saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
