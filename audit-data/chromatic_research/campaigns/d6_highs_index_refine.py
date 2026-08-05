"""Refine lower-index E6* kernel portfolios with the HiGHS geometry loop.

The fixed-form nonmonotonic scans found index 329, 322, and 315 kernels at
the same E6* ratio from which the successful 336 metric deformation started.
This script gives those lower-index kernels their own refreshed
HiGHS/PSD-eigenvector-cut continuation, from both the exact E6* form and a
strong deformed reference form.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.active_metric_refine import _load_problem
from chromatic_research.campaigns.d6_discrete_highs_cycle import kernel_key, optimize_kernel
from chromatic_research.campaigns.d6_highs_kernel_race import complete_ratio
from chromatic_research.core.prime_radon import hnf_columns


def parse_indices(text: str) -> list[int]:
    try:
        values = [int(piece.strip()) for piece in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "indices must be comma-separated integers"
        ) from exc
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("indices must be at least two")
    return values


def parse_starts(text: str) -> list[str]:
    values = [piece.strip() for piece in text.split(",") if piece.strip()]
    allowed = {"e6-generic", "reference"}
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "starts must be a nonempty list without repetitions"
        )
    if any(value not in allowed for value in values):
        raise argparse.ArgumentTypeError(
            "starts may contain only e6-generic and reference"
        )
    return values


def select_portfolio(
    campaigns: Sequence[tuple[Path, dict]],
    indices: Sequence[int],
    per_index: int,
) -> list[dict]:
    """Deduplicate exact HNFs and retain the best fixed-form candidates."""
    wanted = set(int(value) for value in indices)
    grouped: dict[int, dict[tuple[int, ...], dict]] = {
        value: {} for value in wanted
    }
    for path, campaign in campaigns:
        for record in campaign.get("results", []):
            index = int(record.get("image_index", -1))
            raw_kernel = record.get("kernel_basis_columns")
            if index not in wanted or raw_kernel is None:
                continue
            kernel = hnf_columns(np.asarray(raw_kernel, dtype=np.int64))
            key = kernel_key(kernel)
            score = complete_ratio(record)
            incumbent = grouped[index].get(key)
            if incumbent is not None and score <= incumbent["source_ratio"]:
                continue
            grouped[index][key] = {
                "index": index,
                "campaign": str(path),
                "label": record.get("label"),
                "source_ratio": score,
                "kernel_basis_columns": kernel.astype(int).tolist(),
                "kernel_smith": record.get("kernel_smith"),
                "moduli": record.get("moduli"),
                "rows": record.get("rows"),
            }
    selected: list[dict] = []
    for index in indices:
        ranked = sorted(
            grouped[int(index)].values(),
            key=lambda item: item["source_ratio"],
            reverse=True,
        )
        selected.extend(ranked[:per_index])
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument(
        "--indices", type=parse_indices, default=[329, 322, 315]
    )
    parser.add_argument("--per-index", type=int, default=6)
    parser.add_argument(
        "--starts",
        type=parse_starts,
        default=["e6-generic", "reference"],
    )
    parser.add_argument("--generic-scale", type=float, default=1e-7)
    parser.add_argument("--generic-seed", type=int, default=6339701)
    parser.add_argument("--passes", type=int, default=12)
    parser.add_argument("--outer-rounds", type=int, default=50)
    parser.add_argument("--cuts-per-round", type=int, default=256)
    parser.add_argument(
        "--violation-tolerance", type=float, default=2e-8
    )
    parser.add_argument("--positive-floor", type=float, default=1e-7)
    parser.add_argument("--gram-bound-factor", type=float, default=16.0)
    parser.add_argument(
        "--projection-solver",
        choices=("CLARABEL", "SCS"),
        default="CLARABEL",
    )
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.per_index < 1
        or args.passes < 1
        or args.outer_rounds < 1
        or args.cuts_per_round < 1
        or args.violation_tolerance <= 0
        or args.positive_floor <= 0
        or args.gram_bound_factor <= 1
        or args.max_h_norm <= 0
        or args.generic_scale <= 0
    ):
        parser.error("invalid lower-index refinement budget")

    metric_payload = json.loads(args.metric.read_text())
    (
        _,
        _,
        _,
        _,
        _,
        reference_evaluator,
    ) = _load_problem(args.metric, args.temperature, args.max_h_norm)
    reference_parameters = np.asarray(
        metric_payload["best"]["parameters"], dtype=np.float64
    )
    rng = np.random.default_rng(args.generic_seed)
    generic_direction = rng.normal(size=len(reference_parameters))
    generic_direction /= np.linalg.norm(generic_direction)
    starts = {
        "e6-generic": args.generic_scale * generic_direction,
        "reference": reference_parameters,
    }
    campaigns = [
        (path, json.loads(path.read_text())) for path in args.campaigns
    ]
    portfolio = select_portfolio(
        campaigns, args.indices, args.per_index
    )
    found_indices = {candidate["index"] for candidate in portfolio}
    missing = [
        index for index in args.indices if index not in found_indices
    ]
    if missing:
        raise RuntimeError(f"no exact kernels found for indices {missing}")

    started = time.perf_counter()
    payload: dict = {
        "method": (
            "multi-start lower-index HNF refinement with refreshed HiGHS "
            "PSD/eigenvector-cut passes"
        ),
        "reference_metric": str(args.metric),
        "campaigns": [str(path) for path in args.campaigns],
        "n": len(reference_evaluator.basis0),
        "dimension": len(reference_evaluator.basis0),
        "settings": {
            "indices": args.indices,
            "per_index": args.per_index,
            "starts": args.starts,
            "generic_scale": args.generic_scale,
            "generic_seed": args.generic_seed,
            "passes": args.passes,
            "outer_rounds": args.outer_rounds,
            "cuts_per_round": args.cuts_per_round,
            "violation_tolerance": args.violation_tolerance,
            "positive_floor": args.positive_floor,
            "gram_bound_factor": args.gram_bound_factor,
            "projection_solver": args.projection_solver,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
        },
        "portfolio": portfolio,
        "runs": [],
        "best_by_index": {},
        "valid_numerical_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    total = len(portfolio) * len(args.starts)
    print(
        f"indices={args.indices} kernels={len(portfolio)} "
        f"starts={args.starts} runs={total}",
        flush=True,
    )
    run_index = 0
    for candidate in portfolio:
        kernel = np.asarray(
            candidate["kernel_basis_columns"], dtype=np.int64
        )
        if abs(int(round(np.linalg.det(kernel)))) != candidate["index"]:
            raise RuntimeError("stored HNF determinant disagrees with index")
        for start_label in args.starts:
            run_index += 1
            try:
                optimized = optimize_kernel(
                    reference_evaluator.basis0,
                    kernel,
                    starts[start_label],
                    passes=args.passes,
                    temperature=args.temperature,
                    max_h_norm=args.max_h_norm,
                    projection_solver=args.projection_solver,
                    outer_rounds=args.outer_rounds,
                    cuts_per_round=args.cuts_per_round,
                    violation_tolerance=args.violation_tolerance,
                    positive_floor=args.positive_floor,
                    gram_bound_factor=args.gram_bound_factor,
                )
            except Exception as exc:
                record = {
                    "run": run_index,
                    "index": candidate["index"],
                    "label": candidate["label"],
                    "start": start_label,
                    "source_ratio": candidate["source_ratio"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                payload["runs"].append(record)
                save()
                print(
                    f"  {run_index:2d}/{total} index={candidate['index']} "
                    f"start={start_label} ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            record = {
                "run": run_index,
                "index": candidate["index"],
                "label": candidate["label"],
                "start": start_label,
                "source_ratio": candidate["source_ratio"],
                "optimization": optimized,
            }
            payload["runs"].append(record)
            save()
            print(
                f"  {run_index:2d}/{total} index={candidate['index']} "
                f"start={start_label} "
                f"initial={optimized['initial']['min_ratio']:.12f} "
                f"best={optimized['best']['min_ratio']:.12f} "
                f"passes={len(optimized['history'])}",
                flush=True,
            )

    successful = [
        record
        for record in payload["runs"]
        if isinstance(record.get("optimization"), dict)
    ]
    for index in args.indices:
        choices = [
            record
            for record in successful
            if record["index"] == index
        ]
        if choices:
            payload["best_by_index"][str(index)] = max(
                choices,
                key=lambda record: float(
                    record["optimization"]["best"]["min_ratio"]
                ),
            )
    payload["valid_numerical_witness"] = any(
        record["optimization"]["best"]["min_ratio"] >= 1.0
        for record in successful
    )
    save()
    summary = ", ".join(
        f"{index}="
        f"{payload['best_by_index'][str(index)]['optimization']['best']['min_ratio']:.12f}"
        for index in args.indices
        if str(index) in payload["best_by_index"]
    )
    print(
        f"FINAL {summary} valid={payload['valid_numerical_witness']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
