"""Race an HNF kernel portfolio through successive HiGHS metric passes.

The discrete/continuous cycle ranks kernels first at the metric where they
were discovered.  That is inexpensive, but it can miss a kernel whose own
best metric is far from the probe metric.  This script removes that bias with
a multi-fidelity race:

1. deduplicate every archived discrete candidate by exact HNF;
2. give every non-source kernel one HiGHS PSD/eigenvector-cut pass;
3. keep the strongest requested number of survivors;
4. refresh the geometry and repeat on the survivors.

Every score used for survival is a complete ``MetricEvaluator`` oracle score.
The calculation is numerical discovery, not an exact coloring certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from active_metric_refine import _load_problem
from d6_discrete_highs_cycle import kernel_key, optimize_kernel
from prime_radon import hnf_columns


def parse_survivors(text: str) -> list[int]:
    """Parse a strictly decreasing positive survivor schedule."""
    try:
        values = [int(piece.strip()) for piece in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "survivors must be comma-separated integers"
        ) from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(
            "survivors must be positive comma-separated integers"
        )
    if any(right >= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError(
            "survivor counts must be strictly decreasing"
        )
    return values


def complete_ratio(record: dict) -> float:
    """Return the strongest complete score stored for a discrete record."""
    separation = record.get("complete_separation")
    if isinstance(separation, dict):
        ratio = separation.get("minimum_distance_ratio")
        if ratio is not None:
            return float(ratio)
    ratio = record.get("minimum_conflict_ratio")
    return -math.inf if ratio is None else float(ratio)


def deduplicate_portfolio(
    cycle: dict,
    source_kernel: np.ndarray,
) -> list[dict]:
    """Keep the best probe realization of each non-source exact HNF."""
    source_key = kernel_key(source_kernel)
    unique: dict[tuple[int, ...], dict] = {}
    probes = cycle.get("probes", [])
    for record in cycle.get("discrete_candidates", []):
        raw_kernel = record.get("kernel_basis_columns")
        probe_index = record.get("probe_index")
        if raw_kernel is None or probe_index is None:
            continue
        probe_index = int(probe_index)
        if probe_index < 0 or probe_index >= len(probes):
            continue
        kernel = hnf_columns(np.asarray(raw_kernel, dtype=np.int64))
        key = kernel_key(kernel)
        if key == source_key:
            continue
        score = complete_ratio(record)
        incumbent = unique.get(key)
        if incumbent is not None and score <= incumbent["probe_ratio"]:
            continue
        unique[key] = {
            "key": list(key),
            "kernel_basis_columns": kernel.astype(int).tolist(),
            "kernel_smith": record.get("kernel_smith"),
            "discrete_label": record.get("label"),
            "probe_index": probe_index,
            "probe_label": probes[probe_index].get("label"),
            "probe_ratio": score,
            "parameters": probes[probe_index]["parameters"],
            "current": None,
            "rounds": [],
            "active": True,
        }
    return sorted(
        unique.values(),
        key=lambda item: item["probe_ratio"],
        reverse=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cycle", type=Path)
    parser.add_argument(
        "--survivors",
        type=parse_survivors,
        default=[64, 24, 8],
        help=(
            "survivors after successive rounds; the final group receives "
            "one additional round"
        ),
    )
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
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.outer_rounds < 1
        or args.cuts_per_round < 1
        or args.violation_tolerance <= 0
        or args.positive_floor <= 0
        or args.gram_bound_factor <= 1
        or args.max_h_norm <= 0
        or args.save_every < 1
    ):
        parser.error("invalid HiGHS kernel-race budget")

    cycle = json.loads(args.cycle.read_text())
    metric_path = Path(cycle["source_metric"])
    (
        _,
        _,
        _,
        _,
        source_kernel,
        source_evaluator,
    ) = _load_problem(metric_path, args.temperature, args.max_h_norm)
    source_parameters = np.asarray(
        cycle["source"]["parameters"], dtype=np.float64
    )
    source = source_evaluator.evaluate(
        source_parameters, with_witnesses=True
    )
    portfolio = deduplicate_portfolio(cycle, source_kernel)
    if not portfolio:
        raise RuntimeError("cycle contains no alternative exact HNF kernels")
    if args.survivors[0] >= len(portfolio):
        parser.error(
            "first survivor count must be smaller than the portfolio"
        )

    started = time.perf_counter()
    payload: dict = {
        "method": (
            "successive-halving HNF portfolio race with refreshed HiGHS "
            "PSD/eigenvector-cut metric passes"
        ),
        "source_cycle": str(args.cycle),
        "source_metric": str(metric_path),
        "n": len(source_kernel),
        "dimension": len(source_kernel),
        "source": source.as_json(),
        "settings": {
            "survivors": args.survivors,
            "outer_rounds": args.outer_rounds,
            "cuts_per_round": args.cuts_per_round,
            "violation_tolerance": args.violation_tolerance,
            "positive_floor": args.positive_floor,
            "gram_bound_factor": args.gram_bound_factor,
            "projection_solver": args.projection_solver,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
            "save_every": args.save_every,
        },
        "unique_alternative_kernels": len(portfolio),
        "portfolio": portfolio,
        "round_summaries": [],
        "best_alternative": None,
        "best": source.as_json(),
        "valid_numerical_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    active = portfolio
    schedule = args.survivors + [args.survivors[-1]]
    print(
        f"source={source.min_ratio:.12f} "
        f"unique alternatives={len(portfolio)} "
        f"schedule={schedule}",
        flush=True,
    )
    for round_index, survivor_count in enumerate(schedule, start=1):
        round_started = time.perf_counter()
        successful = 0
        print(
            f"round {round_index}/{len(schedule)} "
            f"active={len(active)} target={survivor_count}",
            flush=True,
        )
        for candidate_index, state in enumerate(active, start=1):
            kernel = np.asarray(
                state["kernel_basis_columns"], dtype=np.int64
            )
            start_parameters = np.asarray(
                state["parameters"], dtype=np.float64
            )
            try:
                optimized = optimize_kernel(
                    source_evaluator.basis0,
                    kernel,
                    start_parameters,
                    passes=1,
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
                state["rounds"].append(
                    {
                        "round": round_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"  {candidate_index:3d}/{len(active)} ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            successful += 1
            state["rounds"].append(
                {
                    "round": round_index,
                    "optimization": optimized,
                }
            )
            state["current"] = optimized["best"]
            state["parameters"] = optimized["best"]["parameters"]
            print(
                f"  {candidate_index:3d}/{len(active)} "
                f"probe={state['probe_ratio']:.12f} "
                f"best={state['current']['min_ratio']:.12f}",
                flush=True,
            )
            if candidate_index % args.save_every == 0:
                save()

        ranked = sorted(
            (
                state
                for state in active
                if isinstance(state.get("current"), dict)
            ),
            key=lambda state: float(state["current"]["min_ratio"]),
            reverse=True,
        )
        keep = min(survivor_count, len(ranked))
        survivors = ranked[:keep]
        survivor_ids = {id(state) for state in survivors}
        for state in active:
            state["active"] = id(state) in survivor_ids
        summary = {
            "round": round_index,
            "active": len(active),
            "successful": successful,
            "survivors": keep,
            "best_ratio": (
                ranked[0]["current"]["min_ratio"] if ranked else None
            ),
            "cutoff_ratio": (
                survivors[-1]["current"]["min_ratio"]
                if survivors
                else None
            ),
            "elapsed_seconds": time.perf_counter() - round_started,
        }
        payload["round_summaries"].append(summary)
        active = survivors
        save()
        print(
            f"round {round_index} best={summary['best_ratio']} "
            f"cutoff={summary['cutoff_ratio']}",
            flush=True,
        )
        if not active:
            break

    ranked_all = sorted(
        (
            state
            for state in portfolio
            if isinstance(state.get("current"), dict)
        ),
        key=lambda state: float(state["current"]["min_ratio"]),
        reverse=True,
    )
    if ranked_all:
        payload["best_alternative"] = ranked_all[0]
        if ranked_all[0]["current"]["min_ratio"] > source.min_ratio:
            payload["best"] = ranked_all[0]["current"]
    payload["valid_numerical_witness"] = (
        payload["best"]["min_ratio"] >= 1.0
    )
    save()
    print(
        f"FINAL source={source.min_ratio:.12f} "
        f"best-alternative="
        f"{payload['best_alternative']['current']['min_ratio']:.12f} "
        f"best={payload['best']['min_ratio']:.12f} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
