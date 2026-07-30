"""Determinant-aware destroy-and-exact-repair lattice continuation.

Randomly perturbing an integer sublattice basis and then filtering for one
prescribed determinant is extremely wasteful.  This campaign instead leaves
one matrix entry free and repairs it exactly.  For an integer matrix A,

    det(A + delta E_ij) = det(A) + delta * cofactor_ij(A).

After ``destroy-1`` random entry mutations, the final integer ``delta`` is
therefore forced whenever the cofactor divides the determinant deficit.  The
method produces exact target-index candidates before any geometric work.

Every repaired hit is confirmed with exact integer arithmetic, canonicalized
by column HNF, screened by a shortest-vector necessary condition, and finally
checked with the complete finite Voronoi separation oracle.  Presets cover the
record constructions in dimensions 5--9.
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
from a9_replace71_campaign import C9_FUNDAMENTAL
from conflict_graph import C5
from e8_neighbor_search import C2401_ROWS, e8_geometry
from lazy_prime_campaign import parent_geometry, separate_kernel
from metric_deform import exhaustive_covering_radius
from prime_radon import (
    hnf_columns,
    kernel_basis,
    load_forbidden,
    smith_diagonal,
)
from symlat import kernel_minimal_to_fundamental


HERE = Path(__file__).resolve().parent


def exact_det(matrix: np.ndarray) -> int:
    return int(Matrix(np.asarray(matrix, dtype=np.int64).tolist()).det())


def _geometry_from_basis(
    basis: np.ndarray, diameter: float | None = None
) -> tuple[np.ndarray, float, list[tuple[list[float], float]]]:
    basis = np.asarray(basis, dtype=np.float64)
    facets = combigeo.relevant_facets(basis.tolist())
    if diameter is None:
        radius, _ = exhaustive_covering_radius(facets)
        diameter = 2.0 * radius
    return basis, float(diameter), facets


def _d6_source() -> np.ndarray:
    records = json.loads((HERE / "interval_fast_results.json").read_text())
    record = next(
        item
        for item in records
        if item["name"] == "E6*" and int(item["k"]) == 343
    )
    return hnf_columns(
        kernel_basis(
            [np.asarray(row, dtype=np.int64) for row in record["phi"]],
            record["e_list"],
            6,
        )
    )


def load_preset(
    name: str,
) -> tuple[str, np.ndarray, float, list[tuple[list[float], float]], np.ndarray]:
    """Return lattice label, basis, diameter, facets, source kernel columns."""
    if name == "d5":
        basis, _, diameter = load_forbidden("A5*")
        kernel = kernel_minimal_to_fundamental(C5)
        basis, diameter, facets = _geometry_from_basis(basis, diameter)
        return "A5*", basis, diameter, facets, kernel
    if name == "d6":
        basis, _, diameter = load_forbidden("E6*")
        basis, diameter, facets = _geometry_from_basis(basis, diameter)
        return "E6*", basis, diameter, facets, _d6_source()
    if name == "d7":
        payload = json.loads((HERE / "metric_deform_e7_1323.json").read_text())
        basis = np.asarray(payload["best"]["basis"], dtype=np.float64)
        diameter = float(payload["best"]["diameter"])
        kernel = np.asarray(payload["kernel_basis_columns"], dtype=np.int64)
        basis, diameter, facets = _geometry_from_basis(basis, diameter)
        return "deformed E7*-ABPR", basis, diameter, facets, kernel
    if name == "d8":
        basis, diameter, facets = e8_geometry()
        return "E8", basis, diameter, facets, C2401_ROWS.T.copy()
    if name == "d9":
        basis, diameter, facets = parent_geometry("A9*")
        return "A9*", basis, diameter, facets, C9_FUNDAMENTAL.copy()
    raise ValueError(f"unknown preset {name!r}")


def parse_int_list(text: str) -> list[int]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("expected a nonempty JSON integer list")
    return [int(value) for value in raw]


def _float_det(matrix: np.ndarray) -> int:
    value = float(np.linalg.det(np.asarray(matrix, dtype=np.float64)))
    rounded = int(round(value))
    if not math.isfinite(value) or abs(value - rounded) > 1e-4:
        return exact_det(matrix)
    return rounded


def _float_cofactor(matrix: np.ndarray, row: int, col: int) -> int:
    minor = np.delete(np.delete(matrix, row, axis=0), col, axis=1)
    return ((-1) ** (row + col)) * _float_det(minor)


def _candidate_record(
    kernel: np.ndarray,
    separation: dict,
    *,
    sample: int,
    destroy: int,
    source_rows: np.ndarray,
    shortest_ratio: float,
) -> dict:
    rows = np.asarray(
        Matrix(np.asarray(kernel, dtype=np.int64).T.tolist()).lll().tolist(),
        dtype=np.int64,
    )
    return {
        "sample": sample,
        "destroy": destroy,
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_lll_rows": rows.astype(int).tolist(),
        "determinant": abs(exact_det(kernel)),
        "smith": smith_diagonal(kernel),
        "shortest_vector_norm_ratio": shortest_ratio,
        "distance_ratio": float(separation["minimum_distance_ratio"]),
        "valid": bool(separation["valid"]),
        "checked_kernel_vectors": int(separation["checked_kernel_vectors"]),
        "conflict_count_with_sign": int(
            separation["conflict_count_with_sign"]
        ),
        "conflicts": separation["conflicts"][:40],
        "lll_row_perturbation_frobenius_norm": float(
            np.linalg.norm(rows - source_rows)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=["d5", "d6", "d7", "d8", "d9"])
    parser.add_argument("--target", type=int)
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument(
        "--destroy", type=parse_int_list, default=[3, 4, 5, 6]
    )
    parser.add_argument("--mutation-bound", type=int, default=2)
    parser.add_argument("--repair-bound", type=int, default=12)
    parser.add_argument("--shortest-cutoff", type=float, default=0.78)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.top < 1:
        parser.error("samples and top must be positive")
    if args.mutation_bound < 1 or args.repair_bound < 1:
        parser.error("mutation and repair bounds must be positive")

    lattice, basis, diameter, facets, source_kernel = load_preset(args.preset)
    source_index = abs(exact_det(source_kernel))
    target = source_index - 1 if args.target is None else int(args.target)
    if target < 2:
        parser.error("target determinant must be at least two")
    source_separation = separate_kernel(
        basis, diameter, facets, source_kernel
    )
    if not source_separation["valid"]:
        raise AssertionError(
            f"preset source failed: ratio="
            f"{source_separation['minimum_distance_ratio']}"
        )
    source_rows = np.asarray(
        Matrix(source_kernel.T.tolist()).lll().tolist(), dtype=np.int64
    )
    source_signed_det = exact_det(source_rows)
    signed_target = (1 if source_signed_det > 0 else -1) * target
    n = len(source_rows)
    if any(value < 2 or value > n * n for value in args.destroy):
        parser.error("destroy values must lie between 2 and n^2")

    payload: dict = {
        "method": "determinant-aware destroy-and-exact-repair continuation",
        "preset": args.preset,
        "dimension": n,
        "lattice": lattice,
        "coordinate_convention": "parent row-basis coordinates",
        "source_index": source_index,
        "target_index": target,
        "source_kernel_basis_columns": source_kernel.astype(int).tolist(),
        "source_lll_rows": source_rows.astype(int).tolist(),
        "source_smith": smith_diagonal(source_kernel),
        "source_distance_ratio": source_separation[
            "minimum_distance_ratio"
        ],
        "parent_diameter": diameter,
        "parent_facet_count": len(facets),
        "budget": {
            "samples": args.samples,
            "destroy": args.destroy,
            "mutation_bound": args.mutation_bound,
            "repair_bound": args.repair_bound,
            "shortest_cutoff": args.shortest_cutoff,
            "top": args.top,
            "seed": args.seed,
        },
        "raw_divisible_repairs": 0,
        "exact_determinant_hits": 0,
        "unique_hnf_hits": 0,
        "shortest_vector_passes": 0,
        "full_geometry_evaluations": 0,
        "best": [],
        "valid_candidate": None,
    }
    print(
        f"{args.preset}/{lattice}: source={source_index} "
        f"ratio={source_separation['minimum_distance_ratio']:.12f} "
        f"target={target} smith={payload['source_smith']}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    seen: set[bytes] = set()
    best: list[dict] = []
    start = time.perf_counter()
    progress = max(10_000, args.samples // 10)

    for sample in range(1, args.samples + 1):
        destroy = args.destroy[(sample - 1) % len(args.destroy)]
        positions = rng.choice(n * n, size=destroy, replace=False)
        matrix = source_rows.copy()
        for position in positions[:-1]:
            delta = 0
            while delta == 0:
                delta = int(
                    rng.integers(
                        -args.mutation_bound,
                        args.mutation_bound + 1,
                    )
                )
            matrix[position // n, position % n] += delta
        repair_position = int(positions[-1])
        repair_row, repair_col = divmod(repair_position, n)
        determinant = _float_det(matrix)
        cofactor = _float_cofactor(matrix, repair_row, repair_col)
        deficit = signed_target - determinant
        if cofactor == 0 or deficit % cofactor:
            continue
        repair = deficit // cofactor
        if repair == 0 or abs(repair) > args.repair_bound:
            continue
        payload["raw_divisible_repairs"] += 1
        matrix[repair_row, repair_col] += repair
        if exact_det(matrix) != signed_target:
            raise AssertionError("cofactor repair failed exact determinant check")
        payload["exact_determinant_hits"] += 1
        kernel = hnf_columns(matrix.T)
        key = kernel.tobytes()
        if key in seen:
            continue
        seen.add(key)
        payload["unique_hnf_hits"] = len(seen)

        reduced_rows = np.asarray(
            Matrix(kernel.T.tolist()).lll().tolist(), dtype=np.int64
        )
        sub_basis = reduced_rows @ basis
        shortest = float(
            np.linalg.norm(combigeo.shortest_vector(sub_basis.tolist()))
        )
        shortest_ratio = shortest / diameter
        if shortest_ratio < args.shortest_cutoff:
            continue
        payload["shortest_vector_passes"] += 1
        try:
            separation = separate_kernel(
                basis, diameter, facets, kernel
            )
        except Exception as error:
            print(
                f"  oracle skipped sample {sample}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        payload["full_geometry_evaluations"] += 1
        record = _candidate_record(
            kernel,
            separation,
            sample=sample,
            destroy=destroy,
            source_rows=source_rows,
            shortest_ratio=shortest_ratio,
        )
        best.append(record)
        best.sort(
            key=lambda item: (
                item["valid"],
                item["distance_ratio"],
                item["shortest_vector_norm_ratio"],
            ),
            reverse=True,
        )
        del best[args.top :]
        payload["best"] = best
        if best and best[0] is record:
            print(
                f"  new best sample={sample}: "
                f"ratio={record['distance_ratio']:.12f} "
                f"short={shortest_ratio:.6f} "
                f"conflicts={record['conflict_count_with_sign']} "
                f"destroy={destroy}",
                flush=True,
            )
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
        if record["valid"]:
            payload["valid_candidate"] = record
            payload["samples_completed"] = sample
            payload["elapsed_seconds"] = time.perf_counter() - start
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                f"*** VALID INDEX-{target} CANDIDATE FOUND ***", flush=True
            )
            return 0

        if sample % progress == 0:
            ratio = best[0]["distance_ratio"] if best else -1.0
            print(
                f"  progress {sample}/{args.samples}: "
                f"repairs={payload['exact_determinant_hits']} "
                f"unique={len(seen)} full={payload['full_geometry_evaluations']} "
                f"best={ratio:.12f} elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    payload["samples_completed"] = args.samples
    payload["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    ratio = best[0]["distance_ratio"] if best else -1.0
    print(
        f"FINAL samples={args.samples} repairs="
        f"{payload['exact_determinant_hits']} unique={len(seen)} "
        f"full={payload['full_geometry_evaluations']} "
        f"best={ratio:.12f} saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
