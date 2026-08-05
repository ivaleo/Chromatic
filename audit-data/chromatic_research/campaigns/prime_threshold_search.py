"""Exact fixed-metric threshold search for one prime quotient row.

For a prime index ``p``, every one-row coloring kernel is the kernel of a
nonzero functional ``a`` over ``F_p``.  At a geometric threshold ``tau`` the
fixed metric is feasible exactly when

    a . v != 0 (mod p)

for every projective forbidden vector whose distance ratio is below ``tau``.
The companion C++ enumerator visits each projective row once, in the charts
whose first nonzero coordinate is normalized to one.

With ``--maximize`` this script binary-searches the finite set of attained
distance ratios after first deciding the requested target.  The result is an
exact optimum over all one-row prime kernels for the supplied floating-point
metric and forbidden-vector list.  It is not an impossibility proof over
other metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import combigeo
from chromatic_research.core.block_row_metric_opt import candidate_record
from chromatic_research.core.prime_row_opt import (
    _forbidden_with_weights,
    _is_prime,
    _source_lattice,
)
from chromatic_research.campaigns.threshold_block_search import (
    canonical_mod_rows,
    solve_large_row_enumerator,
)


Decision = tuple[np.ndarray | None, dict, bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def maximize_discrete_threshold(
    levels: Sequence[float],
    target_infeasible_record: dict,
    decide: Callable[[float], Decision],
) -> tuple[float, np.ndarray, list[dict]]:
    """Find the largest feasible attained level by monotone binary search."""

    unique = np.asarray(sorted(set(float(value) for value in levels)))
    if not len(unique):
        raise ValueError("cannot maximize an empty threshold set")
    decisions = [target_infeasible_record]
    low = 0
    high = len(unique)
    best_row: np.ndarray | None = None

    # The lowest attained level has no strictly lower constraints and is
    # therefore feasible, but run the same exhaustive backend so the witness
    # and audit trail are explicit.
    row, record, proved_unsat = decide(float(unique[low]))
    decisions.append(record)
    if row is None or proved_unsat:
        raise AssertionError("the minimum attained threshold must be feasible")
    best_row = row

    while low + 1 < high:
        middle = (low + high) // 2
        row, record, proved_unsat = decide(float(unique[middle]))
        decisions.append(record)
        if row is not None:
            low = middle
            best_row = row
        elif proved_unsat:
            high = middle
        else:
            raise RuntimeError("enumerator returned neither FEASIBLE nor UNSAT")

    if best_row is None:
        raise AssertionError("binary search lost its feasible witness")
    return float(unique[low]), best_row, decisions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--maximize",
        action="store_true",
        help=(
            "if the target is infeasible, find the exact best attained "
            "threshold over all projective rows"
        ),
    )
    parser.add_argument("--enumerator", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not _is_prime(args.prime):
        parser.error("--prime must be prime")
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.enumerator.is_file():
        parser.error(f"enumerator does not exist: {args.enumerator}")

    metric = json.loads(args.metric.read_text())
    lattice = _source_lattice(args.metric, metric)
    basis = np.asarray(metric["best"]["basis"], dtype=np.float64)
    diameter = float(metric["best"]["diameter"])
    facets = combigeo.relevant_facets(basis.tolist())
    forbidden, ratios, weights = _forbidden_with_weights(
        basis,
        diameter,
        max(1.0, args.threshold),
    )
    started = time.perf_counter()

    def decide(threshold: float) -> Decision:
        mask = ratios < threshold
        bad = canonical_mod_rows(forbidden[mask], args.prime)
        row_started = time.perf_counter()
        row, charts, proved_unsat = solve_large_row_enumerator(
            bad,
            args.prime,
            executable=args.enumerator,
            workers=args.workers,
            seed=args.seed + len(decision_records),
        )
        record = {
            "threshold": float(threshold),
            "bad_vectors_before_modular_dedup": int(mask.sum()),
            "bad_projective_rows_mod_prime": int(len(bad)),
            "status": "FEASIBLE" if row is not None else "INFEASIBLE",
            "proved_unsat": bool(proved_unsat),
            "row": row.astype(int).tolist() if row is not None else None,
            "enumeration": charts,
            "elapsed_seconds": time.perf_counter() - row_started,
        }
        decision_records.append(record)
        print(
            f"threshold={threshold:.12g} bad={len(bad)} "
            f"status={record['status']} "
            f"elapsed={record['elapsed_seconds']:.3f}s",
            flush=True,
        )
        return row, record, proved_unsat

    decision_records: list[dict] = []
    target_row, target_record, target_unsat = decide(args.threshold)
    frontier_threshold: float | None = None
    frontier_row: np.ndarray | None = target_row
    if target_row is not None:
        frontier_threshold = float(args.threshold)
    elif args.maximize:
        levels = ratios[ratios < args.threshold]

        # ``decide`` already appends to decision_records.  The helper's return
        # list is useful for unit testing with a pure fake oracle, but the live
        # payload retains the single append-only chronological audit trail.
        frontier_threshold, frontier_row, _ = maximize_discrete_threshold(
            levels,
            target_record,
            decide,
        )

    candidate = None
    valid_candidate = None
    if frontier_row is not None:
        candidate = candidate_record(
            label=(
                f"exact-prime-threshold-{frontier_threshold:.12g}"
                if frontier_threshold is not None
                else "exact-prime-threshold"
            ),
            beta=float(frontier_threshold or args.threshold),
            rows=[frontier_row],
            moduli=[args.prime],
            forbidden=forbidden,
            ratios=ratios,
            weights=weights,
            basis=basis,
            diameter=diameter,
            facets=facets,
            search_seconds=time.perf_counter() - started,
            search_metadata={
                "exact_projective_enumeration": True,
                "target_threshold": args.threshold,
                "frontier_threshold": frontier_threshold,
            },
        )
        if candidate.get("complete_separation", {}).get("valid"):
            valid_candidate = candidate

    source = args.enumerator.with_suffix(".cpp")
    payload = {
        "method": (
            "exact projective finite-field threshold enumeration with "
            "monotone attained-level search"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "dimension": int(len(basis)),
        "prime": args.prime,
        "target_index": args.prime,
        "target_threshold": args.threshold,
        "maximize": bool(args.maximize),
        "forbidden_projective_pairs": int(len(forbidden)),
        "attained_ratio_levels": int(len(np.unique(ratios))),
        "target_proved_unsat": bool(target_unsat),
        "frontier_threshold": frontier_threshold,
        "decisions": decision_records,
        "candidate": candidate,
        "valid_candidate": valid_candidate,
        "enumerator": {
            "path": str(args.enumerator),
            "sha256": _sha256(args.enumerator),
            "source_path": str(source) if source.is_file() else None,
            "source_sha256": _sha256(source) if source.is_file() else None,
            "workers": args.workers,
            "seed": args.seed,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL target_unsat={target_unsat} "
        f"frontier={frontier_threshold} "
        f"candidate_ratio="
        f"{candidate.get('minimum_conflict_ratio') if candidate else None} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
