"""Constrained active-set refinement inside one crossed D6 L-type halfspace.

``d6_ltype_wall_cross.py`` produces strict seeds in adjacent Voronoi/Delone
cones.  An unconstrained metric optimizer can immediately return through the
same wall to the known 0.980658 basin.  This second-stage optimizer therefore
keeps the source-oriented circuit functional strictly negative:

    1/2 <Q(p), A_wall> <= -margin.

At every iteration it combines the finite-difference bundle of active
coloring-distance constraints with a linearization of this wall inequality
in one trust-region LP.  A step is accepted only after the complete Voronoi
and coloring-distance oracle improves and the nonlinear wall inequality is
still satisfied.

This is a numerical adjacent-cone search.  It is neither an exhaustive search
of that cone nor a certificate at ratio one.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from chromatic_research.core.active_metric_refine import (
    _active_gradients,
    _load_problem,
)
from chromatic_research.campaigns.d6_ltype_wall_cross import (
    voronoi_geometry,
    wall_gradient,
    wall_slack,
)
from chromatic_research.core.metric_deform import MetricEvaluation
from chromatic_research.core.prime_radon import smith_diagonal


def constrained_max_min_step(
    values: np.ndarray,
    gradients: np.ndarray,
    radius: float,
    *,
    current_wall_slack: float,
    wall_gradient_values: np.ndarray,
    wall_margin: float,
) -> tuple[np.ndarray, float]:
    """Solve the local max-min LP while staying beyond the crossed wall."""
    dimension = gradients.shape[1]
    active_rows = np.column_stack(
        (-gradients, np.ones(len(values)))
    )
    wall_row = np.r_[
        np.asarray(wall_gradient_values, dtype=np.float64), 0.0
    ][None, :]
    a_ub = np.vstack((active_rows, wall_row))
    b_ub = np.r_[
        values,
        -float(wall_margin) - float(current_wall_slack),
    ]
    result = linprog(
        np.r_[np.zeros(dimension), -1.0],
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(-radius, radius)] * dimension + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"wall-constrained trust-region LP failed: {result.message}"
        )
    return np.asarray(result.x[:-1]), float(result.x[-1])


def _screen_wall(payload: dict, wall_index: int) -> dict:
    matches = [
        wall
        for wall in payload.get("walls", [])
        if int(wall.get("wall_index", -1)) == wall_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"screen contains {len(matches)} walls with index {wall_index}"
        )
    wall = matches[0]
    if wall.get("best_crossing") is None:
        raise ValueError("selected wall has no strict crossing seed")
    return wall


def _compact(evaluation: MetricEvaluation) -> dict:
    return {
        "min_ratio": evaluation.min_ratio,
        "soft_min": evaluation.soft_min,
        "diameter": evaluation.diameter,
        "parameters": evaluation.parameters.tolist(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("--wall-index", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--radius", type=float, default=5e-4)
    parser.add_argument("--min-radius", type=float, default=2e-8)
    parser.add_argument("--active-window", type=float, default=4e-3)
    parser.add_argument("--finite-difference", type=float, default=2e-6)
    parser.add_argument(
        "--wall-margin-fraction", type=float, default=0.1
    )
    parser.add_argument(
        "--absolute-seed-depth",
        type=float,
        default=0.0,
        help=(
            "if positive, start this far past the rooted wall along its "
            "crossing direction instead of using best_crossing"
        ),
    )
    parser.add_argument(
        "--minimum-wall-margin", type=float, default=1e-9
    )
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.wall_index < 0
        or args.iterations < 1
        or args.radius <= 0
        or args.min_radius <= 0
        or args.active_window <= 0
        or args.finite_difference <= 0
        or args.absolute_seed_depth < 0
        or args.minimum_wall_margin <= 0
    ):
        parser.error("invalid constrained-refinement budget")
    if not 0 < args.wall_margin_fraction <= 1:
        parser.error("--wall-margin-fraction must lie in (0,1]")

    screen_payload = json.loads(args.screen.read_text())
    wall_record = _screen_wall(screen_payload, args.wall_index)
    metric_path = Path(screen_payload["source_metric"])
    if not metric_path.is_absolute() and not metric_path.exists():
        metric_path = (
            args.screen.resolve().parent / metric_path.name
        )
    (
        metric_payload,
        source,
        base_metric,
        source_record,
        kernel,
        evaluator,
    ) = _load_problem(
        metric_path, args.temperature, args.max_h_norm
    )
    wall = np.asarray(
        wall_record["wall_matrix"], dtype=np.int64
    )
    source_center = np.asarray(
        metric_payload["best"]["parameters"], dtype=np.float64
    )
    if args.absolute_seed_depth > 0:
        center = source_center + (
            float(wall_record["parameter_root"])
            + args.absolute_seed_depth
        ) * np.asarray(
            wall_record["parameter_direction"], dtype=np.float64
        )
    else:
        center = np.asarray(
            wall_record["best_crossing"]["parameters"],
            dtype=np.float64,
        )
    source_slack = wall_slack(
        evaluator.basis0, source_center, wall
    )
    start_slack = wall_slack(evaluator.basis0, center, wall)
    if source_slack <= 0 or start_slack >= 0:
        raise RuntimeError(
            "wall orientation or crossing seed is inconsistent"
        )
    wall_margin = max(
        args.minimum_wall_margin,
        abs(start_slack) * args.wall_margin_fraction,
    )
    if start_slack > -wall_margin:
        raise RuntimeError("crossing seed does not satisfy wall margin")

    incumbent = evaluator.evaluate(center, with_witnesses=True)
    start_evaluation = incumbent
    start_geometry = voronoi_geometry(start_evaluation.basis)
    best = incumbent
    source_ratio = float(metric_payload["best"]["min_ratio"])
    source_signature = screen_payload["source"][
        "voronoi_signature"
    ]
    radius = float(args.radius)
    evaluations = 1
    history: list[dict] = []
    started = time.perf_counter()
    print(
        f"start wall={args.wall_index} "
        f"ratio={incumbent.min_ratio:.12f} "
        f"slack={start_slack:.6g} margin={wall_margin:.6g}",
        flush=True,
    )

    iteration = 0
    while iteration < args.iterations and radius >= args.min_radius:
        iteration += 1
        (
            local,
            coordinates,
            values,
            gradients,
            gradient_evaluations,
        ) = _active_gradients(
            evaluator,
            center,
            active_window=args.active_window,
            finite_difference=args.finite_difference,
        )
        evaluations += gradient_evaluations
        current_slack = wall_slack(
            evaluator.basis0, center, wall
        )
        wall_gradient_values = wall_gradient(
            evaluator.basis0,
            center,
            wall,
            args.finite_difference,
        )
        step, predicted = constrained_max_min_step(
            values,
            gradients,
            radius,
            current_wall_slack=current_slack,
            wall_gradient_values=wall_gradient_values,
            wall_margin=wall_margin,
        )
        accepted: MetricEvaluation | None = None
        accepted_scale = 0.0
        accepted_slack = current_slack
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_parameters = center + scale * step
            trial_slack = wall_slack(
                evaluator.basis0, trial_parameters, wall
            )
            if trial_slack > -0.999 * wall_margin:
                continue
            trial = evaluator.evaluate(trial_parameters)
            evaluations += 1
            if trial.min_ratio > incumbent.min_ratio + 2e-11:
                accepted = trial
                accepted_scale = scale
                accepted_slack = trial_slack
                break
        previous_ratio = incumbent.min_ratio
        if accepted is not None:
            center = accepted.parameters
            incumbent = evaluator.evaluate(
                center, with_witnesses=True
            )
            evaluations += 1
            if incumbent.min_ratio > best.min_ratio:
                best = incumbent
            radius = min(
                args.radius * 4.0,
                radius
                * (1.2 if accepted_scale == 1.0 else 0.9),
            )
            outcome = "accept"
        else:
            radius *= 0.45
            outcome = "shrink"
        history.append(
            {
                "iteration": iteration,
                "outcome": outcome,
                "active_constraints": len(coordinates),
                "radius": radius,
                "predicted_min": predicted,
                "accepted_scale": accepted_scale,
                "previous_min_ratio": previous_ratio,
                "incumbent_min_ratio": incumbent.min_ratio,
                "incumbent_wall_slack": (
                    accepted_slack
                    if accepted is not None
                    else current_slack
                ),
                "best_min_ratio": best.min_ratio,
            }
        )
        print(
            f"iter {iteration:3d}: {outcome:6s} "
            f"active={len(coordinates):2d} "
            f"min={incumbent.min_ratio:.12f} "
            f"best={best.min_ratio:.12f} "
            f"slack={wall_slack(evaluator.basis0, center, wall):.3g} "
            f"radius={radius:.3g}",
            flush=True,
        )

    best = evaluator.evaluate(best.parameters, with_witnesses=True)
    evaluations += 1
    best_slack = wall_slack(
        evaluator.basis0, best.parameters, wall
    )
    best_geometry = voronoi_geometry(best.basis)
    payload = {
        "method": (
            "wall-constrained active-set finite-difference max-min "
            "trust region"
        ),
        "source_screen": str(args.screen),
        "source_metric": str(metric_path),
        "source_campaign": str(source),
        "base_metric": (
            str(base_metric) if base_metric is not None else None
        ),
        "source_record": {
            "moduli": source_record["moduli"],
            "rows": source_record["rows"],
            "image_index": source_record["image_index"],
            "beta": source_record.get("beta"),
        },
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(
            int(round(np.linalg.det(kernel)))
        ),
        "kernel_smith": smith_diagonal(kernel),
        "wall_index": args.wall_index,
        "wall_matrix": wall.astype(int).tolist(),
        "source_reference_ratio": source_ratio,
        "source_wall_slack": source_slack,
        "start": {
            **_compact(start_evaluation),
            "wall_slack": start_slack,
            "voronoi_signature": start_geometry.signature,
        },
        "wall_margin": wall_margin,
        "iterations": iteration,
        "evaluations": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
        "final_radius": radius,
        "settings": {
            "iterations_budget": args.iterations,
            "initial_radius": args.radius,
            "minimum_radius": args.min_radius,
            "active_window": args.active_window,
            "finite_difference": args.finite_difference,
            "wall_margin_fraction": args.wall_margin_fraction,
            "absolute_seed_depth": args.absolute_seed_depth,
            "minimum_wall_margin": args.minimum_wall_margin,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
        },
        "history": history,
        "best": {
            **best.as_json(),
            "wall_slack": best_slack,
            "voronoi_signature": best_geometry.signature,
        },
        "remained_beyond_wall": (
            best_slack <= -0.999 * wall_margin
            and best_geometry.signature != source_signature
        ),
        "beats_source_basin": best.min_ratio > source_ratio,
        "valid_numerical_witness": best.min_ratio >= 1.0,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL wall={args.wall_index} "
        f"min={best.min_ratio:.12f} "
        f"slack={best_slack:.6g} "
        f"other-side={payload['remained_beyond_wall']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
