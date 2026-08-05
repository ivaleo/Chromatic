"""Threshold oracle for one large-prime and one small coloring block.

For a fixed parent metric and a fixed small-block row, requiring every
conflicting vector to have distance ratio at least ``tau`` is equivalent to

    a . v != 0 (mod p)

for every vector ``v`` whose small-block residue is zero and whose geometric
ratio is below ``tau``.  This script sends that finite system to CP-SAT.  The
large row is normalized projectively by splitting the search into the ``n``
charts

    a_0 = ... = a_{j-1} = 0,  a_j = 1.

An ``OPTIMAL`` infeasibility result in every chart is therefore an exact
finite-field UNSAT result for the supplied floating-point threshold set.  It
is not a proof over all parent metrics.  Any feasible row is rechecked by the
complete geometric separation oracle before it is saved.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from ortools.sat.python import cp_model

import combigeo
from chromatic_research.core.block_row_metric_opt import candidate_record
from chromatic_research.core.prime_radon import projective_forms
from chromatic_research.core.prime_row_opt import (
    _forbidden_with_weights,
    _is_prime,
    _source_lattice,
)


def canonical_mod_rows(vectors: np.ndarray, prime: int) -> np.ndarray:
    """Deduplicate nonzero projective rows modulo ``prime``."""
    unique: set[tuple[int, ...]] = set()
    for raw in np.asarray(vectors, dtype=np.int64):
        row = raw % prime
        nonzero = np.flatnonzero(row)
        if not len(nonzero):
            # Such a vector is orthogonal to every large-block row and must
            # remain explicit so the caller can declare this small row UNSAT.
            unique.add(tuple(0 for _ in row))
            continue
        pivot = int(nonzero[0])
        row = row * pow(int(row[pivot]), -1, prime) % prime
        unique.add(tuple(int(value) for value in row))
    if not unique:
        return np.empty((0, np.asarray(vectors).shape[1]), dtype=np.int64)
    return np.asarray(sorted(unique), dtype=np.int64)


def solve_large_row(
    bad_rows: np.ndarray,
    prime: int,
    *,
    time_limit: float,
    workers: int,
    seed: int,
    hint_row: np.ndarray | None = None,
) -> tuple[np.ndarray | None, list[dict], bool]:
    """Solve all projective charts; return row, chart records, complete UNSAT."""
    bad_rows = np.asarray(bad_rows, dtype=np.int64) % prime
    n = bad_rows.shape[1]
    if hint_row is not None:
        hint_row = np.asarray(hint_row, dtype=np.int64) % prime
        if hint_row.shape != (n,) or not np.any(hint_row):
            raise ValueError("hint row must be a nonzero vector of length n")
        hint_pivot = int(np.flatnonzero(hint_row)[0])
        hint_row = (
            hint_row
            * pow(int(hint_row[hint_pivot]), -1, prime)
            % prime
        )
    else:
        hint_pivot = -1
    chart_records: list[dict] = []
    all_charts_proved_unsat = True

    if np.any(np.all(bad_rows == 0, axis=1)):
        return None, [
            {
                "chart": None,
                "status": "INFEASIBLE_ZERO_CONSTRAINT",
                "wall_seconds": 0.0,
            }
        ], True

    for pivot in range(n):
        model = cp_model.CpModel()
        variables = []
        for coordinate in range(n):
            if coordinate < pivot:
                variables.append(model.new_constant(0))
            elif coordinate == pivot:
                variables.append(model.new_constant(1))
            else:
                variables.append(
                    model.new_int_var(0, prime - 1, f"a_{coordinate}")
                )

        for constraint_index, vector in enumerate(bad_rows):
            residue = model.new_int_var(
                1, prime - 1, f"r_{constraint_index}"
            )
            expression = sum(
                int(coefficient) * variable
                for coefficient, variable in zip(vector, variables)
                if coefficient
            )
            model.add_modulo_equality(residue, expression, prime)
        if pivot == hint_pivot:
            for variable, value in zip(variables, hint_row):
                model.add_hint(variable, int(value))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = int(workers)
        solver.parameters.random_seed = int(seed + pivot)
        started = time.perf_counter()
        status = solver.solve(model)
        elapsed = time.perf_counter() - started
        status_name = solver.status_name(status)
        chart_records.append(
            {
                "chart": pivot,
                "status": status_name,
                "wall_seconds": elapsed,
                "branches": int(solver.num_branches),
                "conflicts": int(solver.num_conflicts),
            }
        )
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            row = np.asarray(
                [solver.value(variable) for variable in variables],
                dtype=np.int64,
            )
            return row, chart_records, False
        if status != cp_model.INFEASIBLE:
            all_charts_proved_unsat = False

    return None, chart_records, all_charts_proved_unsat


def solve_large_row_enumerator(
    bad_rows: np.ndarray,
    prime: int,
    *,
    executable: Path,
    workers: int,
    seed: int,
) -> tuple[np.ndarray | None, list[dict], bool]:
    """Call the exhaustive C++ projective enumerator."""
    bad_rows = np.asarray(bad_rows, dtype=np.int64) % prime
    n = bad_rows.shape[1]
    if len(bad_rows):
        # Only the early-rejection speed depends on order.  Shuffling with a
        # recorded seed avoids an accidental pathological lexicographic order.
        order = np.random.default_rng(seed).permutation(len(bad_rows))
        bad_rows = bad_rows[order]
    lines = [f"{prime} {n} {len(bad_rows)}"]
    lines.extend(" ".join(str(int(value)) for value in row) for row in bad_rows)
    completed = subprocess.run(
        [str(executable), str(workers)],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    result["stderr"] = completed.stderr
    result["backend"] = "exact_cpp_projective_enumeration"
    expected_rows = (prime**n - 1) // (prime - 1)
    status = result.get("status")
    if status == "FEASIBLE":
        return (
            np.asarray(result["row"], dtype=np.int64),
            [result],
            False,
        )
    if status != "INFEASIBLE":
        return None, [result], False
    if result.get("reason") != "zero_constraint":
        tested = int(result.get("tested", -1))
        if tested != expected_rows:
            raise AssertionError(
                f"enumerator claimed UNSAT after {tested} rows; "
                f"expected {expected_rows}"
            )
    return None, [result], True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--large-prime", type=int, required=True)
    parser.add_argument("--small-modulus", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--small-row",
        type=json.loads,
        help="optional fixed small row; otherwise scan its full projective pool",
    )
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print every Nth completed small row (feasible rows always print)",
    )
    parser.add_argument(
        "--enumerator",
        type=Path,
        help=(
            "use the compiled threshold_enum.cpp executable instead of "
            "CP-SAT; this exhaustively scans every projective row"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not _is_prime(args.large_prime):
        parser.error("--large-prime must be prime")
    if args.small_modulus < 2:
        parser.error("--small-modulus must be at least two")
    if math.gcd(args.large_prime, args.small_modulus) != 1:
        parser.error("block moduli must be coprime")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if args.time_limit <= 0 or args.workers < 1 or args.progress_every < 1:
        parser.error("time limit, workers, and progress interval must be positive")

    metric = json.loads(args.metric.read_text())
    lattice = _source_lattice(args.metric, metric)
    basis = np.asarray(metric["best"]["basis"], dtype=np.float64)
    diameter = float(metric["best"]["diameter"])
    facets = combigeo.relevant_facets(basis.tolist())
    forbidden, ratios, weights = _forbidden_with_weights(
        basis, diameter, max(1.0, args.threshold)
    )
    n = len(basis)
    small_pool = projective_forms(n, args.small_modulus)
    if args.small_row is not None:
        supplied = np.asarray(args.small_row, dtype=np.int64)
        if supplied.shape != (n,):
            parser.error(f"--small-row must contain exactly {n} entries")
        supplied %= args.small_modulus
        matches = [
            row for row in small_pool if np.array_equal(row, supplied)
        ]
        if not matches:
            parser.error("--small-row is not a normalized projective row")
        small_pool = np.asarray(matches, dtype=np.int64)

    source_record = metric.get("source_record", {})
    source_large_row: np.ndarray | None = None
    source_small_row: np.ndarray | None = None
    if (
        source_record.get("moduli")
        == [args.large_prime, args.small_modulus]
        and len(source_record.get("rows", [])) == 2
    ):
        source_large_row = (
            np.asarray(source_record["rows"][0], dtype=np.int64)
            % args.large_prime
        )
        source_small_row = (
            np.asarray(source_record["rows"][1], dtype=np.int64)
            % args.small_modulus
        )
        if args.small_row is None:
            for index, row in enumerate(small_pool):
                if np.array_equal(row, source_small_row):
                    small_pool[[0, index]] = small_pool[[index, 0]]
                    break

    started = time.perf_counter()
    payload: dict = {
        "method": (
            "projective-chart CP-SAT threshold oracle for a large prime block"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "dimension": n,
        "large_prime": args.large_prime,
        "small_modulus": args.small_modulus,
        "threshold": args.threshold,
        "forbidden_projective_pairs": int(len(forbidden)),
        "small_rows_total": int(len(small_pool)),
        "settings": {
            "time_limit_per_chart": args.time_limit,
            "workers": args.workers,
            "seed": args.seed,
        },
        "small_rows": [],
        "best_candidate": None,
        "valid_candidate": None,
    }

    best_ratio = -math.inf
    all_small_rows_proved_unsat = True
    for small_index, small_row in enumerate(small_pool):
        small_zero = (forbidden @ small_row) % args.small_modulus == 0
        bad_mask = small_zero & (ratios < args.threshold)
        bad_rows = canonical_mod_rows(
            forbidden[bad_mask], args.large_prime
        )
        row_started = time.perf_counter()
        if args.enumerator is None:
            large_row, charts, proved_unsat = solve_large_row(
                bad_rows,
                args.large_prime,
                time_limit=args.time_limit,
                workers=args.workers,
                seed=args.seed + 100 * small_index,
                hint_row=(
                    source_large_row
                    if source_small_row is not None
                    and np.array_equal(small_row, source_small_row)
                    else None
                ),
            )
        else:
            large_row, charts, proved_unsat = solve_large_row_enumerator(
                bad_rows,
                args.large_prime,
                executable=args.enumerator,
                workers=args.workers,
                seed=args.seed + 100 * small_index,
            )
        row_record: dict = {
            "small_row": small_row.astype(int).tolist(),
            "bad_vectors_before_modular_dedup": int(bad_mask.sum()),
            "bad_projective_rows_mod_large_prime": int(len(bad_rows)),
            "charts": charts,
            "proved_unsat": bool(proved_unsat),
            "elapsed_seconds": time.perf_counter() - row_started,
        }
        if large_row is not None:
            record = candidate_record(
                label=f"threshold-{args.threshold:.12g}",
                beta=float(args.threshold),
                rows=[large_row, small_row],
                moduli=[args.large_prime, args.small_modulus],
                forbidden=forbidden,
                ratios=ratios,
                weights=weights,
                basis=basis,
                diameter=diameter,
                facets=facets,
                search_seconds=time.perf_counter() - started,
                search_metadata={
                    "small_index": small_index,
                    "bad_constraints": int(len(bad_rows)),
                    "charts": charts,
                },
            )
            row_record["candidate"] = record
            ratio = record.get("minimum_conflict_ratio")
            comparable_ratio = 1.0 if ratio is None else float(ratio)
            if comparable_ratio > best_ratio:
                best_ratio = comparable_ratio
                payload["best_candidate"] = record
            if record.get("complete_separation", {}).get("valid"):
                payload["valid_candidate"] = record
            all_small_rows_proved_unsat = False
            print(
                f"small {small_index + 1}/{len(small_pool)} FEASIBLE: "
                f"min-ratio={ratio} killed={record['killed']}",
                flush=True,
            )
        elif (
            (small_index + 1) % args.progress_every == 0
            or small_index + 1 == len(small_pool)
        ):
            all_small_rows_proved_unsat &= proved_unsat
            print(
                f"small {small_index + 1}/{len(small_pool)} "
                f"{'UNSAT' if proved_unsat else 'UNKNOWN'}: "
                f"bad={len(bad_rows)}",
                flush=True,
            )
        else:
            all_small_rows_proved_unsat &= proved_unsat
        payload["small_rows"].append(row_record)
        payload["small_rows_completed"] = small_index + 1
        payload["all_completed_small_rows_proved_unsat"] = (
            all_small_rows_proved_unsat
        )
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        if payload["valid_candidate"] is not None:
            print("*** VALID THRESHOLD KERNEL FOUND ***", flush=True)
            return 0

    payload["all_small_rows_proved_unsat"] = all_small_rows_proved_unsat
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"finished: all-small-rows-UNSAT={all_small_rows_proved_unsat} "
        f"best-ratio={payload['best_candidate'] and best_ratio}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
