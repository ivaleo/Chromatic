"""Active-set trust-region refinement of a lattice-coloring metric.

``metric_deform.py`` uses a soft minimum so that CMA-ES can traverse the
piecewise-smooth geometric objective.  Close to feasibility, however, the
hard minimum is governed by a small set of sublattice-vector constraints and
the identity of the worst vector changes frequently.  This script performs a
bundle-like second stage:

1. enumerate every sublattice vector in the rigorous finite search radius;
2. retain all constraints within ``active_window`` of the hard minimum;
3. estimate their gradients by symmetric finite differences;
4. solve the local max-min linear program in an L-infinity trust region;
5. accept a step only after the complete Voronoi/separation oracle improves.

No numerical result produced here is a proof.  A candidate crossing one still
has to be rationalized and checked by ``verify_metric_candidate.py``.
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

from e7_abpr import M_E7
from metric_deform import MetricEvaluation, MetricEvaluator, select_record
from prime_radon import hnf_columns, kernel_basis, load_forbidden, smith_diagonal


def _resolve_source(metric_path: Path, source_text: str) -> Path:
    source = Path(source_text)
    if source.is_absolute() or source.exists():
        return source
    sibling = metric_path.resolve().parent / source.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"cannot resolve source campaign {source_text!r}")


def _load_problem(
    metric_path: Path, temperature: float, max_h_norm: float
) -> tuple[dict, Path, Path | None, dict, np.ndarray, MetricEvaluator]:
    metric_payload = json.loads(metric_path.read_text())
    source = _resolve_source(metric_path, metric_payload["source_campaign"])
    source_payload = json.loads(source.read_text())
    source_record = metric_payload["source_record"]
    if source_record.get("rows"):
        record = select_record(
            source_payload,
            source_record.get("moduli"),
            source_record.get("beta"),
            rows=source_record.get("rows"),
        )
        rows = [np.asarray(row, dtype=np.int64) for row in record["rows"]]
        n = int(source_payload.get("n", len(rows[0])))
        kernel = hnf_columns(kernel_basis(rows, record["moduli"], n))
    else:
        record = source_record
        kernel = hnf_columns(
            np.asarray(metric_payload["kernel_basis_columns"], dtype=np.int64)
        )
        n = len(kernel)
    base_text = metric_payload.get("base_metric")
    if base_text is None:
        base_text = metric_payload.get("optimizer", {}).get("base_metric")
    base_metric = (
        _resolve_source(metric_path, base_text) if base_text else None
    )
    lattice_name = source_payload["lattice"]
    if base_metric is not None:
        base_payload = json.loads(base_metric.read_text())
        basis0 = np.asarray(base_payload["best"]["basis"], dtype=np.float64)
        if basis0.shape != (n, n) or not np.all(np.isfinite(basis0)):
            raise ValueError(
                f"base metric basis has shape {basis0.shape}, expected {(n, n)}"
            )
    elif lattice_name == "E7*-ABPR":
        basis0 = np.linalg.cholesky(M_E7().T @ M_E7())
    else:
        basis0, _, _ = load_forbidden(lattice_name)
    evaluator = MetricEvaluator(
        basis0,
        kernel,
        softmin_temperature=temperature,
        max_h_norm=max_h_norm,
    )
    return metric_payload, source, base_metric, record, kernel, evaluator


def _ratio_map(evaluation: MetricEvaluation) -> dict[tuple[int, ...], float]:
    return {
        tuple(int(value) for value in witness["coordinate"]): float(
            witness["distance_ratio"]
        )
        for witness in evaluation.witnesses
    }


def _active_gradients(
    evaluator: MetricEvaluator,
    center: np.ndarray,
    *,
    active_window: float,
    finite_difference: float,
) -> tuple[MetricEvaluation, list[tuple[int, ...]], np.ndarray, np.ndarray, int]:
    current = evaluator.evaluate(
        center,
        with_witnesses=True,
        witness_window=active_window,
    )
    current_map = _ratio_map(current)
    # Distances are centrally symmetric, so retain one representative from
    # each +/- pair.  This removes duplicate rows without assuming any other
    # lattice symmetry.
    coordinates: list[tuple[int, ...]] = []
    for coordinate in current_map:
        negative = tuple(-value for value in coordinate)
        if negative in current_map and negative < coordinate:
            continue
        coordinates.append(coordinate)
    values = np.asarray([current_map[key] for key in coordinates])
    gradients = np.zeros((len(coordinates), len(center)), dtype=np.float64)
    evaluations = 1
    for index in range(len(center)):
        offset = np.zeros_like(center)
        offset[index] = finite_difference
        plus = evaluator.evaluate(
            center + offset,
            with_witnesses=True,
            witness_window=active_window + 10.0 * finite_difference,
        )
        minus = evaluator.evaluate(
            center - offset,
            with_witnesses=True,
            witness_window=active_window + 10.0 * finite_difference,
        )
        evaluations += 2
        plus_map = _ratio_map(plus)
        minus_map = _ratio_map(minus)
        for row, coordinate in enumerate(coordinates):
            if coordinate not in plus_map or coordinate not in minus_map:
                raise RuntimeError(
                    f"active coordinate {coordinate} left finite enumeration"
                )
            gradients[row, index] = (
                plus_map[coordinate] - minus_map[coordinate]
            ) / (2.0 * finite_difference)
    return current, coordinates, values, gradients, evaluations


def _max_min_step(
    values: np.ndarray, gradients: np.ndarray, radius: float
) -> tuple[np.ndarray, float]:
    dimension = gradients.shape[1]
    # t <= value_i + gradient_i . step
    a_ub = np.column_stack((-gradients, np.ones(len(values))))
    result = linprog(
        np.r_[np.zeros(dimension), -1.0],
        A_ub=a_ub,
        b_ub=values,
        bounds=[(-radius, radius)] * dimension + [(None, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"trust-region LP failed: {result.message}")
    return np.asarray(result.x[:-1]), float(result.x[-1])


def _payload(
    metric_path: Path,
    source: Path,
    base_metric: Path | None,
    record: dict,
    kernel: np.ndarray,
    evaluation: MetricEvaluation,
    *,
    elapsed: float,
    evaluations: int,
    iterations: int,
    radius: float,
    settings: dict,
    history: list[dict],
) -> dict:
    return {
        "method": "active-set finite-difference max-min trust region",
        "source_metric": str(metric_path),
        "source_campaign": str(source),
        "base_metric": str(base_metric) if base_metric is not None else None,
        "source_record": {
            "moduli": record["moduli"],
            "rows": record["rows"],
            "image_index": record["image_index"],
            "beta": record.get("beta"),
            "source_minimum_conflict_ratio": record.get(
                "minimum_conflict_ratio"
            ),
        },
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "iterations": iterations,
        "evaluations": evaluations,
        "elapsed_seconds": round(elapsed, 6),
        "final_radius": radius,
        "settings": settings,
        "history": history,
        "best": evaluation.as_json(),
        "valid_numerical_witness": evaluation.min_ratio >= 1.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--radius", type=float, default=8e-4)
    parser.add_argument("--min-radius", type=float, default=2e-8)
    parser.add_argument("--active-window", type=float, default=4e-3)
    parser.add_argument("--finite-difference", type=float, default=2e-6)
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--target-margin", type=float, default=2e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "active_metric_best.json",
    )
    args = parser.parse_args(argv)

    (
        metric_payload,
        source,
        base_metric,
        record,
        kernel,
        evaluator,
    ) = _load_problem(args.metric, args.temperature, args.max_h_norm)
    center = np.asarray(metric_payload["best"]["parameters"], dtype=np.float64)
    incumbent = evaluator.evaluate(center, with_witnesses=True)
    recorded_ratio = float(metric_payload["best"]["min_ratio"])
    consistency_tolerance = max(5e-8, 5e-7 * abs(recorded_ratio))
    if abs(incumbent.min_ratio - recorded_ratio) > consistency_tolerance:
        raise RuntimeError(
            "metric parameterization mismatch: recomputed start "
            f"{incumbent.min_ratio:.12g} != recorded {recorded_ratio:.12g}; "
            "check the base_metric chain"
        )
    best = incumbent
    radius = float(args.radius)
    total_evaluations = 1
    history: list[dict] = []
    start = time.perf_counter()
    settings = {
        "iterations_budget": args.iterations,
        "initial_radius": args.radius,
        "minimum_radius": args.min_radius,
        "active_window": args.active_window,
        "finite_difference": args.finite_difference,
        "temperature": args.temperature,
        "max_h_norm": args.max_h_norm,
    }
    print(
        f"start min={incumbent.min_ratio:.12f} "
        f"soft={incumbent.soft_min:.12f} radius={radius:g}",
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
        total_evaluations += gradient_evaluations
        step, predicted = _max_min_step(values, gradients, radius)
        accepted: MetricEvaluation | None = None
        accepted_scale = 0.0
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = evaluator.evaluate(center + scale * step)
            total_evaluations += 1
            if trial.min_ratio > incumbent.min_ratio + 2e-11:
                accepted = trial
                accepted_scale = scale
                break
        old_ratio = incumbent.min_ratio
        if accepted is not None:
            center = accepted.parameters
            incumbent = evaluator.evaluate(center, with_witnesses=True)
            total_evaluations += 1
            if incumbent.min_ratio > best.min_ratio:
                best = incumbent
            radius = min(args.radius * 4.0, radius * (1.2 if accepted_scale == 1 else 0.9))
            outcome = "accept"
        else:
            radius *= 0.45
            outcome = "shrink"
        record_history = {
            "iteration": iteration,
            "outcome": outcome,
            "active_constraints": len(coordinates),
            "radius": radius,
            "predicted_min": predicted,
            "accepted_scale": accepted_scale,
            "previous_min_ratio": old_ratio,
            "incumbent_min_ratio": incumbent.min_ratio,
            "best_min_ratio": best.min_ratio,
        }
        history.append(record_history)
        print(
            f"iter {iteration:3d}: {outcome:6s} active={len(coordinates):2d} "
            f"pred={predicted:.12f} min={incumbent.min_ratio:.12f} "
            f"best={best.min_ratio:.12f} radius={radius:.3g}",
            flush=True,
        )
        best_full = evaluator.evaluate(best.parameters, with_witnesses=True)
        total_evaluations += 1
        args.output.write_text(
            json.dumps(
                _payload(
                    args.metric,
                    source,
                    base_metric,
                    record,
                    kernel,
                    best_full,
                    elapsed=time.perf_counter() - start,
                    evaluations=total_evaluations,
                    iterations=iteration,
                    radius=radius,
                    settings=settings,
                    history=history,
                ),
                indent=2,
            )
            + "\n"
        )
        if best.min_ratio >= 1.0 + args.target_margin:
            print("*** numerical separation target reached ***", flush=True)
            break

    final = evaluator.evaluate(best.parameters, with_witnesses=True)
    total_evaluations += 1
    payload = _payload(
        args.metric,
        source,
        base_metric,
        record,
        kernel,
        final,
        elapsed=time.perf_counter() - start,
        evaluations=total_evaluations,
        iterations=iteration,
        radius=radius,
        settings=settings,
        history=history,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL min={final.min_ratio:.12f} D={final.min_distance:.12f} "
        f"diam={final.diameter:.12f} valid={final.min_ratio >= 1.0} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
