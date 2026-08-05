"""Refine the leading trajectories from a HiGHS HNF kernel race.

This is the convergence stage for ``d6_highs_kernel_race.py``.  It starts
from each candidate's own best metric, refreshes the exact geometry after
every HiGHS outer pass, and keeps the best complete-oracle result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.active_metric_refine import _load_problem
from chromatic_research.campaigns.d6_discrete_highs_cycle import optimize_kernel


def last_gain(state: dict) -> float:
    """Return the last successful race-round gain for one candidate."""
    values = [
        float(round_record["optimization"]["best"]["min_ratio"])
        for round_record in state.get("rounds", [])
        if isinstance(round_record.get("optimization"), dict)
    ]
    if len(values) < 2:
        return 0.0
    return values[-1] - values[-2]


def select_candidates(
    race: dict,
    top: int,
    rising: int,
) -> list[dict]:
    """Take current leaders plus fast-rising trajectories, without repeats."""
    available = [
        state
        for state in race.get("portfolio", [])
        if isinstance(state.get("current"), dict)
    ]
    by_score = sorted(
        available,
        key=lambda state: float(state["current"]["min_ratio"]),
        reverse=True,
    )
    by_gain = sorted(available, key=last_gain, reverse=True)
    selected: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for state in by_score[:top] + by_gain[:rising]:
        key = tuple(int(value) for value in state["key"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(state)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("race", type=Path)
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--rising", type=int, default=8)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from an existing output and skip recorded labels",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.top < 1
        or args.rising < 0
        or args.passes < 1
        or args.outer_rounds < 1
        or args.cuts_per_round < 1
        or args.violation_tolerance <= 0
        or args.positive_floor <= 0
        or args.gram_bound_factor <= 1
        or args.max_h_norm <= 0
    ):
        parser.error("invalid HiGHS refinement budget")

    race = json.loads(args.race.read_text())
    metric_path = Path(race["source_metric"])
    (
        _,
        _,
        _,
        _,
        _,
        source_evaluator,
    ) = _load_problem(metric_path, args.temperature, args.max_h_norm)
    source_parameters = np.asarray(
        race["source"]["parameters"], dtype=np.float64
    )
    source = source_evaluator.evaluate(
        source_parameters, with_witnesses=True
    )
    selected = select_candidates(race, args.top, args.rising)
    if not selected:
        raise RuntimeError("race contains no refinable candidates")

    started = time.perf_counter()
    elapsed_offset = 0.0
    if args.resume:
        if not args.output.exists():
            parser.error("--resume requires an existing output")
        payload = json.loads(args.output.read_text())
        if payload.get("source_race") != str(args.race):
            parser.error("resume output belongs to a different race")
        elapsed_offset = float(payload.get("elapsed_seconds", 0.0))
    else:
        payload: dict = {
            "method": (
                "convergence refinement of leading and rising HNF kernels "
                "with refreshed HiGHS PSD/eigenvector-cut passes"
            ),
            "source_race": str(args.race),
            "source_metric": str(metric_path),
            "n": race["n"],
            "dimension": race["dimension"],
            "source": source.as_json(),
            "settings": {
                "top": args.top,
                "rising": args.rising,
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
            "selected_candidates": len(selected),
            "refinements": [],
            "best_alternative": None,
            "best": source.as_json(),
            "valid_numerical_witness": False,
        }

    def save() -> None:
        payload["elapsed_seconds"] = (
            elapsed_offset + time.perf_counter() - started
        )
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    completed = {
        record.get("discrete_label")
        for record in payload["refinements"]
    }
    print(
        f"source={source.min_ratio:.12f} selected={len(selected)} "
        f"passes={args.passes} completed={len(completed)}",
        flush=True,
    )
    for index, state in enumerate(selected, start=1):
        if state.get("discrete_label") in completed:
            print(
                f"  {index:2d}/{len(selected)} SKIP "
                f"{state.get('discrete_label')}",
                flush=True,
            )
            continue
        kernel = np.asarray(
            state["kernel_basis_columns"], dtype=np.int64
        )
        try:
            optimized = optimize_kernel(
                source_evaluator.basis0,
                kernel,
                np.asarray(
                    state["current"]["parameters"], dtype=np.float64
                ),
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
            payload["refinements"].append(
                {
                    "rank": index - 1,
                    "discrete_label": state.get("discrete_label"),
                    "race_ratio": state["current"]["min_ratio"],
                    "race_last_gain": last_gain(state),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            save()
            print(
                f"  {index:2d}/{len(selected)} ERROR "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        record = {
            "rank": index - 1,
            "discrete_label": state.get("discrete_label"),
            "race_ratio": state["current"]["min_ratio"],
            "race_last_gain": last_gain(state),
            "optimization": optimized,
        }
        payload["refinements"].append(record)
        save()
        print(
            f"  {index:2d}/{len(selected)} "
            f"race={state['current']['min_ratio']:.12f} "
            f"best={optimized['best']['min_ratio']:.12f} "
            f"passes={len(optimized['history'])}",
            flush=True,
        )

    successful = [
        record
        for record in payload["refinements"]
        if isinstance(record.get("optimization"), dict)
    ]
    if not successful:
        raise RuntimeError("no candidate completed continuous refinement")
    best_alternative = max(
        successful,
        key=lambda record: float(
            record["optimization"]["best"]["min_ratio"]
        ),
    )
    payload["best_alternative"] = best_alternative
    if (
        best_alternative["optimization"]["best"]["min_ratio"]
        > source.min_ratio
    ):
        payload["best"] = best_alternative["optimization"]["best"]
    payload["valid_numerical_witness"] = (
        payload["best"]["min_ratio"] >= 1.0
    )
    save()
    print(
        f"FINAL source={source.min_ratio:.12f} "
        f"best-alternative="
        f"{best_alternative['optimization']['best']['min_ratio']:.12f} "
        f"best={payload['best']['min_ratio']:.12f} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
