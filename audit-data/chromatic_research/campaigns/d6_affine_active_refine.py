"""HiGHS active-set refinement of an affine-coset metric checkpoint.

The black-box affine CMA search is effective for crossing Voronoi walls but
converges slowly once several same-color displacements exchange control of the
hard minimum.  This second stage freezes a local active bundle, estimates all
of its metric gradients, solves the max-min trust-region LP with open HiGHS,
and accepts a step only after reevaluation by the complete affine Voronoi
oracle.

This is numerical discovery.  A ratio crossing one would still require
rational reconstruction and an independent exact geometric certificate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.active_metric_refine import _active_gradients, _max_min_step
from chromatic_research.campaigns.d6_affine_metric_opt import (
    AffineMetricEvaluator,
    checkpoint_affine_cosets,
)
from chromatic_research.core.d6_cyclic_hole_search import primitive_cyclic_row
from chromatic_research.core.determinant_repair import exact_det, load_preset
from chromatic_research.core.prime_radon import hnf_columns, kernel_basis, smith_diagonal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--radius", type=float, default=1.5e-3)
    parser.add_argument("--min-radius", type=float, default=2e-8)
    parser.add_argument("--active-window", type=float, default=1.5e-2)
    parser.add_argument("--finite-difference", type=float, default=2e-6)
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--target-margin", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.iterations < 1
        or args.radius <= 0
        or args.min_radius <= 0
        or args.active_window <= 0
        or args.finite_difference <= 0
        or args.temperature <= 0
        or args.max_h_norm <= 0
    ):
        parser.error("invalid active-set budget or tolerance")

    try:
        metric_payload = json.loads(args.metric.read_text())
        row = np.asarray(metric_payload["cyclic_row"], dtype=np.int64)
        modulus = int(metric_payload["period_index"])
        center = np.asarray(
            metric_payload["best"]["parameters"],
            dtype=np.float64,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"invalid affine metric checkpoint: {error}")
    if not primitive_cyclic_row(row, modulus):
        parser.error("checkpoint cyclic row is not primitive")

    lattice, basis0, _, _, _ = load_preset("d6")
    period = hnf_columns(
        kernel_basis([row], [modulus], len(row))
    )
    if abs(exact_det(period)) != modulus:
        raise AssertionError("cyclic period determinant mismatch")
    try:
        (
            cosets,
            difference,
            block_size,
            difference_residues,
        ) = checkpoint_affine_cosets(metric_payload, row, modulus)
    except (TypeError, ValueError) as error:
        parser.error(f"invalid affine difference data: {error}")
    evaluator = AffineMetricEvaluator(
        basis0,
        period,
        cosets,
        softmin_temperature=args.temperature,
        max_h_norm=args.max_h_norm,
    )
    expected_parameters = len(row) * (len(row) + 1) // 2 - 1
    if center.shape != (expected_parameters,):
        parser.error("checkpoint metric parameters have the wrong shape")
    incumbent = evaluator.evaluate(center, with_witnesses=True)
    recorded_ratio = float(metric_payload["best"]["min_ratio"])
    tolerance = max(5e-8, 5e-7 * abs(recorded_ratio))
    if abs(incumbent.min_ratio - recorded_ratio) > tolerance:
        parser.error(
            "checkpoint metric mismatch: recomputed "
            f"{incumbent.min_ratio:.12g}, recorded {recorded_ratio:.12g}"
        )

    best = incumbent
    radius = float(args.radius)
    evaluations = 1
    history: list[dict] = []
    started = time.perf_counter()
    settings = {
        "iterations": args.iterations,
        "initial_radius": args.radius,
        "minimum_radius": args.min_radius,
        "active_window": args.active_window,
        "finite_difference": args.finite_difference,
        "temperature": args.temperature,
        "max_h_norm": args.max_h_norm,
    }

    def payload(iteration: int) -> dict:
        return {
            "method": (
                "affine-coset active-set finite-difference max-min "
                "trust region with HiGHS LP"
            ),
            "lattice": lattice,
            "source_metric": str(args.metric),
            "source_campaign": metric_payload.get("source_campaign"),
            "source_record_index": metric_payload.get(
                "source_record_index"
            ),
            "period_index": modulus,
            "target_colors": metric_payload.get("target_colors"),
            "target_difference": difference,
            "block_size": block_size,
            "difference_residues": (
                difference_residues.astype(int).tolist()
                if difference_residues is not None
                else None
            ),
            "cyclic_row": row.astype(int).tolist(),
            "period_basis_columns": period.astype(int).tolist(),
            "period_smith": smith_diagonal(period),
            "affine_coset_representatives": cosets.astype(int).tolist(),
            "settings": settings,
            "iteration": iteration,
            "evaluations": evaluations,
            "elapsed_seconds": time.perf_counter() - started,
            "final_radius": radius,
            "history": history,
            "best": best.as_json(),
            "valid_numerical_witness": best.min_ratio >= 1.0,
        }

    print(
        f"start min={incumbent.min_ratio:.12f} "
        f"radius={radius:g}",
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
        step, predicted = _max_min_step(values, gradients, radius)
        accepted = None
        accepted_scale = 0.0
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = evaluator.evaluate(center + scale * step)
            evaluations += 1
            if trial.min_ratio > incumbent.min_ratio + 2e-11:
                accepted = trial
                accepted_scale = scale
                break
        old_ratio = incumbent.min_ratio
        if accepted is not None:
            center = accepted.parameters
            incumbent = evaluator.evaluate(center, with_witnesses=True)
            evaluations += 1
            if incumbent.min_ratio > best.min_ratio:
                best = incumbent
            radius = min(
                args.radius * 4.0,
                radius * (1.2 if accepted_scale == 1.0 else 0.9),
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
                "local_min_ratio": local.min_ratio,
                "previous_min_ratio": old_ratio,
                "incumbent_min_ratio": incumbent.min_ratio,
                "best_min_ratio": best.min_ratio,
            }
        )
        print(
            f"iter {iteration:3d}: {outcome:6s} "
            f"active={len(coordinates):2d} "
            f"pred={predicted:.12f} "
            f"min={incumbent.min_ratio:.12f} "
            f"best={best.min_ratio:.12f} radius={radius:.3g}",
            flush=True,
        )
        best = evaluator.evaluate(best.parameters, with_witnesses=True)
        evaluations += 1
        args.output.write_text(
            json.dumps(payload(iteration), indent=2) + "\n"
        )
        if best.min_ratio >= 1.0 + args.target_margin:
            print("*** numerical affine target reached ***", flush=True)
            break
    best = evaluator.evaluate(best.parameters, with_witnesses=True)
    evaluations += 1
    args.output.write_text(json.dumps(payload(iteration), indent=2) + "\n")
    print(
        f"FINAL min={best.min_ratio:.12f} "
        f"D={best.min_distance:.12f} diam={best.diameter:.12f} "
        f"valid={best.min_ratio >= 1.0}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
