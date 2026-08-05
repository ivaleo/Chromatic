"""CP-SAT threshold oracle for the D6 quotient [7,4,4,3].

The block best-response search can stop with two conflicts because removing
them requires several character rows to change together.  Cartesian mask
enumeration is exact but has about 10^12 tuples.  This model keeps the modular
arithmetic symbolic:

* the modulo-7 and modulo-3 rows are projectively normalized by a one-hot
  first-nonzero pivot;
* the two modulo-4 rows are represented in a unique systematic basis.  Their
  reduction modulo 2 is in RREF, and the 15 possible pivot patterns are solved
  separately; every nonpivot entry has an exact parity and lift bit;
* every below-threshold lattice vector has a disjunction requiring at least
  one of the four modular dot products to be nonzero.
* a soft variant can minimize either the number of violated disjunctions or
  positive integerized powers of their geometric deficits.  The latter avoids
  preferring two catastrophic short vectors to many almost-admissible ones.

Thus a FEASIBLE result is an exact solution of the finite modular threshold
problem for the supplied floating-point metric.  INFEASIBLE is complete only
when all 15 modulo-2 pivot patterns are proved infeasible.  Any feasible
kernel is still checked by the complete geometric separation oracle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from ortools.sat.python import cp_model

import combigeo
from chromatic_research.core.block_row_metric_opt import candidate_record
from chromatic_research.core.prime_row_opt import _forbidden_with_weights, _source_lattice
from chromatic_research.campaigns.threshold_multiblock_search import row_module_key


def rref_mod2(
    rows: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """RREF and pivot columns over F_2."""
    matrix = np.asarray(rows, dtype=np.int64).copy() % 2
    row_count, column_count = matrix.shape
    rank = 0
    pivots: list[int] = []
    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(rank, row_count)
                if matrix[row, column]
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(row_count):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        pivots.append(column)
        rank += 1
        if rank == row_count:
            break
    if rank != row_count:
        raise ValueError("rows are not independent modulo 2")
    return matrix, tuple(pivots)


def canonical_free_rows_mod4(
    rows: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Unique systematic basis of a free rank-two row module modulo 4."""
    matrix = np.asarray(rows, dtype=np.int64) % 4
    if matrix.ndim != 2 or matrix.shape[0] != 2:
        raise ValueError("expected two modulo-4 rows")
    reduced, pivots_raw = rref_mod2(matrix)
    if len(pivots_raw) != 2:
        raise ValueError("modulo-4 rows are not free of rank two")
    pivots = (int(pivots_raw[0]), int(pivots_raw[1]))
    pivot_matrix = matrix[:, pivots]
    a, b = map(int, pivot_matrix[0])
    c, d = map(int, pivot_matrix[1])
    determinant = (a * d - b * c) % 4
    inverse_det = pow(determinant, -1, 4)
    inverse = (
        inverse_det
        * np.asarray([[d, -b], [-c, a]], dtype=np.int64)
    ) % 4
    canonical = inverse @ matrix % 4
    if not np.array_equal(
        canonical[:, pivots], np.eye(2, dtype=np.int64)
    ):
        raise AssertionError("failed to normalize modulo-4 pivot columns")
    if not np.array_equal(canonical % 2, reduced):
        raise AssertionError("systematic lift changed the modulo-2 row space")
    return canonical, pivots


def normalized_prime_row(
    row: Sequence[int] | np.ndarray, prime: int
) -> tuple[np.ndarray, int]:
    row = np.asarray(row, dtype=np.int64) % prime
    nonzero = np.flatnonzero(row)
    if not len(nonzero):
        raise ValueError("prime row must be nonzero")
    pivot = int(nonzero[0])
    normalized = row * pow(int(row[pivot]), -1, prime) % prime
    return normalized, pivot


def integer_conflict_weights(
    ratios: Sequence[float] | np.ndarray,
    threshold: float,
    power: float,
    scale: int,
) -> np.ndarray:
    """Positive integer approximation to a geometric deficit objective.

    Count-only MaxSAT treats one vector at ratio zero as preferable to many
    vectors just below the threshold.  That is the wrong ordering for the
    max-min coloring objective.  Integerized deficit powers preserve CP-SAT's
    exact arithmetic while making a deep violation much more expensive than a
    shallow one.  ``ceil`` keeps every forbidden vector at positive cost, so a
    zero objective is still equivalent to threshold feasibility.
    """
    values = np.asarray(ratios, dtype=np.float64)
    if (
        not math.isfinite(threshold)
        or threshold <= 0
        or not math.isfinite(power)
        or power <= 0
        or scale < 1
    ):
        raise ValueError("threshold, power, and scale must be positive")
    normalized = np.maximum(0.0, (threshold - values) / threshold)
    return np.maximum(
        1,
        np.ceil(float(scale) * np.power(normalized, power)).astype(np.int64),
    )


def source_rows(metric: dict, n: int) -> dict[int, list[np.ndarray]]:
    record = metric.get("source_record", {})
    moduli = record.get("moduli")
    rows = record.get("rows")
    if (
        not isinstance(moduli, list)
        or not isinstance(rows, list)
        or len(moduli) != len(rows)
    ):
        return {}
    grouped: dict[int, list[np.ndarray]] = {}
    for modulus, row in zip(moduli, rows):
        array = np.asarray(row, dtype=np.int64)
        if array.shape != (n,):
            return {}
        grouped.setdefault(int(modulus), []).append(array)
    return grouped


def best_campaign_hint(
    payload: dict, n: int
) -> tuple[dict[int, list[np.ndarray]], dict | None]:
    """Best exact-image [7,4,4,3] record from a modular campaign."""
    records = [
        record
        for record in payload.get("results", [])
        if (
            isinstance(record, dict)
            and record.get("moduli") == [7, 4, 4, 3]
            and isinstance(record.get("rows"), list)
            and len(record["rows"]) == 4
        )
    ]
    for key in (
        "candidate",
        "objective_best",
        "best_by_minimum_ratio",
        "valid_candidate",
    ):
        record = payload.get(key)
        if (
            isinstance(record, dict)
            and record.get("moduli") == [7, 4, 4, 3]
            and record not in records
        ):
            records.append(record)
    if not records:
        return {}, None
    records.sort(
        key=lambda record: (
            int(record.get("killed", 10**9)),
            -float(record.get("minimum_conflict_ratio") or -1.0),
        )
    )
    best = records[0]
    metric_like = {
        "source_record": {
            "moduli": best["moduli"],
            "rows": best["rows"],
        }
    }
    return source_rows(metric_like, n), best


def prime_row_variables(
    model: cp_model.CpModel,
    prime: int,
    n: int,
    label: str,
    hint: np.ndarray | None,
) -> list[cp_model.IntVar]:
    """Projectively normalized nonzero prime row."""
    entries = [
        model.new_int_var(0, prime - 1, f"{label}_{column}")
        for column in range(n)
    ]
    pivots = [
        model.new_bool_var(f"{label}_pivot_{column}")
        for column in range(n)
    ]
    model.add_exactly_one(pivots)
    for pivot, indicator in enumerate(pivots):
        for column in range(pivot):
            model.add(entries[column] == 0).only_enforce_if(indicator)
        model.add(entries[pivot] == 1).only_enforce_if(indicator)
    if hint is not None:
        normalized, hint_pivot = normalized_prime_row(hint, prime)
        for entry, value in zip(entries, normalized):
            model.add_hint(entry, int(value))
        for pivot, indicator in enumerate(pivots):
            model.add_hint(indicator, int(pivot == hint_pivot))
    return entries


def mod4_systematic_variables(
    model: cp_model.CpModel,
    n: int,
    pivots: tuple[int, int],
    hint: np.ndarray | None,
) -> list[list[cp_model.IntVar]]:
    """Canonical free rank-two modulo-4 block for one RREF pivot pattern."""
    rows: list[list[cp_model.IntVar]] = [
        [model.new_int_var(0, 3, f"q4_{row}_{column}") for column in range(n)]
        for row in range(2)
    ]
    pivot_set = set(pivots)
    for row_index, pivot in enumerate(pivots):
        for column in range(n):
            entry = rows[row_index][column]
            if column in pivot_set:
                model.add(
                    entry == int(column == pivot)
                )
                continue
            lift = model.new_bool_var(
                f"q4_lift_{row_index}_{column}"
            )
            if column < pivot:
                parity: int | cp_model.IntVar = 0
            else:
                parity = model.new_bool_var(
                    f"q4_parity_{row_index}_{column}"
                )
            model.add(entry == parity + 2 * lift)
    if hint is not None:
        for row, hint_row in zip(rows, hint):
            for entry, value in zip(row, hint_row):
                model.add_hint(entry, int(value))
    return rows


def modular_nonzero(
    model: cp_model.CpModel,
    row: Sequence[cp_model.IntVar],
    vector: np.ndarray,
    modulus: int,
    label: str,
) -> cp_model.IntVar:
    coefficients = np.asarray(vector, dtype=np.int64) % modulus
    expression = sum(
        int(coefficient) * entry
        for coefficient, entry in zip(coefficients, row)
        if coefficient
    )
    remainder = model.new_int_var(0, modulus - 1, f"{label}_rem")
    model.add_modulo_equality(remainder, expression, modulus)
    nonzero = model.new_bool_var(f"{label}_nonzero")
    model.add(remainder >= 1).only_enforce_if(nonzero)
    model.add(remainder == 0).only_enforce_if(nonzero.negated())
    return nonzero


def solve_pattern(
    bad: np.ndarray,
    pivots: tuple[int, int],
    *,
    hints: dict[int, list[np.ndarray]],
    time_limit: float,
    workers: int,
    seed: int,
    minimize_conflicts: bool = False,
    violation_weights: Sequence[int] | np.ndarray | None = None,
) -> tuple[str, list[np.ndarray] | None, dict]:
    n = bad.shape[1]
    if violation_weights is not None:
        objective_weights = np.asarray(
            violation_weights, dtype=np.int64
        )
        if (
            objective_weights.shape != (len(bad),)
            or np.any(objective_weights <= 0)
        ):
            raise ValueError(
                "violation weights must be positive and match bad vectors"
            )
        if not minimize_conflicts:
            raise ValueError(
                "violation weights require minimize_conflicts=True"
            )
    else:
        objective_weights = None
    model = cp_model.CpModel()
    row7 = prime_row_variables(
        model,
        7,
        n,
        "q7",
        hints.get(7, [None])[0] if hints.get(7) else None,
    )
    row3 = prime_row_variables(
        model,
        3,
        n,
        "q3",
        hints.get(3, [None])[0] if hints.get(3) else None,
    )
    q4_hint = None
    if len(hints.get(4, [])) == 2:
        canonical, source_pivots = canonical_free_rows_mod4(hints[4])
        if source_pivots == pivots:
            q4_hint = canonical
    rows4 = mod4_systematic_variables(
        model, n, pivots, q4_hint
    )

    violations: list[cp_model.IntVar] = []
    for vector_index, vector in enumerate(bad):
        nonzero = [
            modular_nonzero(
                model, row7, vector, 7, f"v{vector_index}_q7"
            ),
            modular_nonzero(
                model, rows4[0], vector, 4, f"v{vector_index}_q4a"
            ),
            modular_nonzero(
                model, rows4[1], vector, 4, f"v{vector_index}_q4b"
            ),
            modular_nonzero(
                model, row3, vector, 3, f"v{vector_index}_q3"
            ),
        ]
        if minimize_conflicts:
            violation = model.new_bool_var(
                f"v{vector_index}_violation"
            )
            model.add_bool_or([*nonzero, violation])
            violations.append(violation)
        else:
            model.add_bool_or(nonzero)
    if minimize_conflicts:
        if objective_weights is None:
            model.minimize(sum(violations))
        else:
            model.minimize(
                sum(
                    int(weight) * violation
                    for weight, violation in zip(
                        objective_weights, violations
                    )
                )
            )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = False
    started = time.perf_counter()
    status = solver.solve(model)
    elapsed = time.perf_counter() - started
    status_name = solver.status_name(status)
    metadata = {
        "pivots": list(pivots),
        "status": status_name,
        "wall_seconds": elapsed,
        "branches": int(solver.num_branches),
        "conflicts": int(solver.num_conflicts),
        "best_objective_bound": float(solver.best_objective_bound),
        "minimize_conflicts": minimize_conflicts,
        "weighted_conflicts": objective_weights is not None,
    }
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return status_name, None, metadata
    if minimize_conflicts:
        metadata["objective_value"] = float(solver.objective_value)
    rows = [
        np.asarray([solver.value(entry) for entry in row7], dtype=np.int64),
        np.asarray(
            [solver.value(entry) for entry in rows4[0]], dtype=np.int64
        ),
        np.asarray(
            [solver.value(entry) for entry in rows4[1]], dtype=np.int64
        ),
        np.asarray([solver.value(entry) for entry in row3], dtype=np.int64),
    ]
    return status_name, rows, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-patterns", type=int, default=15)
    parser.add_argument("--seed", type=int, default=6337101)
    parser.add_argument(
        "--hint-campaign",
        type=Path,
        help=(
            "optional modular campaign whose lowest-conflict [7,4,4,3] "
            "record replaces metric.source_record as the CP-SAT hint"
        ),
    )
    parser.add_argument(
        "--minimize-conflicts",
        action="store_true",
        help=(
            "solve a soft minimum-conflict model in every pivot pattern; "
            "stop early only if the exact optimum reaches zero"
        ),
    )
    parser.add_argument(
        "--conflict-weight-power",
        type=float,
        help=(
            "with --minimize-conflicts, minimize positive integerized "
            "(threshold-ratio)^power costs instead of the raw conflict count"
        ),
    )
    parser.add_argument(
        "--conflict-weight-scale",
        type=int,
        default=1_000_000_000,
        help="largest integerized cost used by --conflict-weight-power",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if args.time_limit <= 0 or args.workers < 1:
        parser.error("time limit and worker count must be positive")
    if not 1 <= args.max_patterns <= 15:
        parser.error("--max-patterns must lie in [1,15]")
    if args.conflict_weight_power is not None:
        if not args.minimize_conflicts:
            parser.error(
                "--conflict-weight-power requires --minimize-conflicts"
            )
        if (
            not math.isfinite(args.conflict_weight_power)
            or args.conflict_weight_power <= 0
        ):
            parser.error("--conflict-weight-power must be positive")
    if args.conflict_weight_scale < 1:
        parser.error("--conflict-weight-scale must be positive")

    metric = json.loads(args.metric.read_text())
    lattice = _source_lattice(args.metric, metric)
    basis = np.asarray(metric["best"]["basis"], dtype=np.float64)
    diameter = float(metric["best"]["diameter"])
    forbidden, ratios, weights = _forbidden_with_weights(
        basis, diameter, max(1.0, args.threshold)
    )
    bad_mask = ratios < args.threshold
    bad = forbidden[bad_mask]
    bad_ratios = ratios[bad_mask]
    violation_weights = None
    if args.conflict_weight_power is not None:
        violation_weights = integer_conflict_weights(
            bad_ratios,
            args.threshold,
            args.conflict_weight_power,
            args.conflict_weight_scale,
        )
    n = len(basis)
    hints = source_rows(metric, n)
    hint_record = None
    if args.hint_campaign is not None:
        hint_payload = json.loads(args.hint_campaign.read_text())
        campaign_hints, hint_record = best_campaign_hint(hint_payload, n)
        if not campaign_hints:
            parser.error(
                "--hint-campaign contains no usable [7,4,4,3] record"
            )
        hints = campaign_hints
    source_pivots = None
    if len(hints.get(4, [])) == 2:
        _, source_pivots = canonical_free_rows_mod4(hints[4])
    patterns = list(itertools.combinations(range(n), 2))
    if source_pivots in patterns:
        patterns.remove(source_pivots)
        patterns.insert(0, source_pivots)
    patterns = patterns[: args.max_patterns]
    facets = combigeo.relevant_facets(basis.tolist())
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "CP-SAT joint modular threshold model with projective prime "
            "normalization and systematic free rank-two Z/4 block"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "dimension": n,
        "structure": [7, 4, 4, 3],
        "target_index": 336,
        "threshold": args.threshold,
        "forbidden_projective_pairs": int(len(forbidden)),
        "below_threshold_pairs": int(len(bad)),
        "budget": {
            "time_limit_per_pattern": args.time_limit,
            "workers": args.workers,
            "max_patterns": args.max_patterns,
            "seed": args.seed,
            "minimize_conflicts": args.minimize_conflicts,
            "conflict_weight_power": args.conflict_weight_power,
            "conflict_weight_scale": (
                args.conflict_weight_scale
                if args.conflict_weight_power is not None
                else None
            ),
        },
        "source_pivots_mod2": (
            list(source_pivots) if source_pivots is not None else None
        ),
        "hint_campaign": (
            str(args.hint_campaign)
            if args.hint_campaign is not None
            else None
        ),
        "hint_record": (
            {
                "label": hint_record.get("label"),
                "killed": hint_record.get("killed"),
                "minimum_conflict_ratio": hint_record.get(
                    "minimum_conflict_ratio"
                ),
                "rows": hint_record.get("rows"),
            }
            if hint_record is not None
            else None
        ),
        "patterns": [],
        "candidate": None,
        "candidates": [],
        "valid_candidate": None,
        "complete_unsat": False,
        "positive_conflict_lower_bound_proved": False,
    }
    best_key: tuple[float, int, float] | None = None

    for pattern_index, pivots in enumerate(patterns):
        status, rows, metadata = solve_pattern(
            bad,
            pivots,
            hints=hints,
            time_limit=args.time_limit,
            workers=args.workers,
            seed=args.seed + pattern_index,
            minimize_conflicts=args.minimize_conflicts,
            violation_weights=violation_weights,
        )
        payload["patterns"].append(metadata)
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"pivots={pivots} status={status} "
            f"time={metadata['wall_seconds']:.2f}s "
            f"branches={metadata['branches']}",
            flush=True,
        )
        if rows is None:
            continue
        record = candidate_record(
            label="cpsat-threshold",
            beta=float(args.threshold),
            rows=rows,
            moduli=[7, 4, 4, 3],
            forbidden=forbidden,
            ratios=ratios,
            weights=weights,
            basis=basis,
            diameter=diameter,
            facets=facets,
            search_seconds=time.perf_counter() - started,
            search_metadata=metadata,
        )
        payload["candidates"].append(record)
        minimum_ratio = float(
            record["minimum_conflict_ratio"]
            if record["minimum_conflict_ratio"] is not None
            else math.inf
        )
        objective_value = float(
            metadata.get("objective_value", record["killed"])
        )
        candidate_key = (
            objective_value,
            int(record["killed"]),
            -minimum_ratio,
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            payload["candidate"] = record
        if record.get("complete_separation", {}).get("valid"):
            payload["valid_candidate"] = record
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"FEASIBLE killed={record['killed']} "
            f"min-ratio={record['minimum_conflict_ratio']} "
            f"valid={payload['valid_candidate'] is not None}",
            flush=True,
        )
        if (
            payload["valid_candidate"] is not None
            or not args.minimize_conflicts
        ):
            return 0

    all_patterns_complete = len(patterns) == 15
    payload["complete_unsat"] = (
        not args.minimize_conflicts
        and all_patterns_complete
        and all(
            pattern["status"] == "INFEASIBLE"
            for pattern in payload["patterns"]
        )
    )
    payload["positive_conflict_lower_bound_proved"] = (
        args.minimize_conflicts
        and all_patterns_complete
        and payload["candidate"] is not None
        and int(payload["candidate"]["killed"]) > 0
        and all(
            pattern["status"] == "OPTIMAL"
            for pattern in payload["patterns"]
        )
    )
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "FINAL "
        f"best-killed="
        f"{payload['candidate']['killed'] if payload['candidate'] else None} "
        f"complete_unsat={payload['complete_unsat']} "
        "positive-lower-bound="
        f"{payload['positive_conflict_lower_bound_proved']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
