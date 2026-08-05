"""CEGAR metric deformation inside the A9-star permutohedral cone.

Generic halfspace vertex enumeration is too expensive for the 3,628,800
vertices of the A9-star Voronoi cell.  This optimizer combines:

* the explicit superbase/permutation covering-radius oracle;
* a sampled archive of Delone permutations inside the CMA-ES loop;
* periodic complete permutation scans that add the current worst vertex;
* all 1022 proper-subset facet inequalities for separation distances; and
* a final comparison with ``combigeo.relevant_facets`` to confirm that the
  optimized form remains in the same Delone secondary cone.

The sampled loop is only a search surrogate.  Every reported incumbent uses
all 10! Delone permutations.  As usual, a candidate reaching ratio one still
requires rationalization and an independent exact coloring certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from sympy import Matrix

import combigeo
from chromatic_research.core.lattices import Astar
from chromatic_research.core.metric_deform import (
    MetricEvaluation,
    cma,
    resolve_saved_path,
    span_stretch_seed,
    trace_free_matrix,
)
from chromatic_research.campaigns.permutohedral_cover import (
    covering_radius,
    covering_radius_for_orders,
    permutohedral_facet_coordinates,
    permutohedral_facets,
    superbase_from_astar_basis,
)
from chromatic_research.core.prime_radon import smith_diagonal


class SecondaryConeEvaluator:
    def __init__(
        self,
        basis0: np.ndarray,
        kernel_rows: np.ndarray,
        orders: np.ndarray,
        *,
        temperature: float,
        max_h_norm: float,
        cone_margin: float,
    ) -> None:
        self.basis0 = np.asarray(basis0, dtype=np.float64)
        self.kernel_rows = np.asarray(kernel_rows, dtype=np.int64)
        self.orders = np.asarray(orders, dtype=np.int64)
        self.n = len(self.basis0)
        self.temperature = float(temperature)
        self.max_h_norm = float(max_h_norm)
        self.cone_margin = float(cone_margin)

    def evaluate(
        self,
        parameters: Sequence[float],
        *,
        full_cover: bool = False,
        with_witnesses: bool = False,
        witness_window: float = 1e-7,
    ) -> tuple[MetricEvaluation, dict | None]:
        parameters_array = np.asarray(parameters, dtype=np.float64)
        deformation = trace_free_matrix(parameters_array, self.n)
        h_norm = float(np.linalg.norm(deformation))
        if not np.isfinite(h_norm) or h_norm > self.max_h_norm:
            excess = max(0.0, h_norm - self.max_h_norm)
            objective = 10.0 + excess * excess
            return (
                MetricEvaluation(
                    objective=objective,
                    soft_min=-objective,
                    min_ratio=-objective,
                    min_distance=0.0,
                    diameter=float("inf"),
                    facet_count=0,
                    vertex_count=0,
                    subvector_count=0,
                    h_norm=h_norm,
                    parameters=parameters_array,
                    basis=np.full_like(self.basis0, np.nan),
                    witnesses=[],
                ),
                None,
            )

        basis = self.basis0 @ expm(deformation)
        cone = obtuse_superbase_check(basis, margin=self.cone_margin)
        if not full_cover and not cone["feasible"]:
            # A nonpositive off-diagonal superbase Gram matrix is a cheap
            # certificate that all subset sums still describe the
            # permutohedral Voronoi cell.  Make every cone-feasible point
            # dominate every infeasible point, while retaining a graded
            # violation for CMA-ES populations that temporarily leave it.
            violation = float(cone["violation"])
            objective = 5.0 + 1000.0 * violation
            return (
                MetricEvaluation(
                    objective=objective,
                    soft_min=-objective,
                    min_ratio=-objective,
                    min_distance=0.0,
                    diameter=float("inf"),
                    facet_count=0,
                    vertex_count=0,
                    subvector_count=0,
                    h_norm=h_norm,
                    parameters=parameters_array,
                    basis=basis,
                    witnesses=[],
                ),
                None,
            )
        facets = permutohedral_facets(basis)
        if full_cover:
            radius, vertex_count, cover_witness = covering_radius(
                basis, with_witness=True
            )
        else:
            radius, cover_witness = covering_radius_for_orders(
                basis, self.orders, with_witness=True
            )
            vertex_count = len(self.orders)
        diameter = 2.0 * radius

        sub_basis = self.kernel_rows @ basis
        vectors = combigeo._vectors_near(
            sub_basis.tolist(),
            [0.0] * self.n,
            2.0 * diameter + 1e-8,
        )
        inverse = np.linalg.inv(basis)
        values: list[tuple[float, float, np.ndarray]] = []
        for raw_vector in vectors:
            vector = np.asarray(raw_vector, dtype=np.float64)
            if np.linalg.norm(vector) < 1e-10:
                continue
            distance = 2.0 * combigeo.dist_to_halfspaces(
                (0.5 * vector).tolist(), facets
            )
            coordinate = np.rint(vector @ inverse).astype(np.int64)
            values.append((distance / diameter, distance, coordinate))

        if values:
            ratios = np.asarray([item[0] for item in values], dtype=np.float64)
            min_ratio = float(ratios.min())
            min_distance = float(min(item[1] for item in values))
            shifted = np.exp(
                -self.temperature * (ratios - min_ratio)
            ).sum()
            soft_min = min_ratio - math.log(float(shifted)) / self.temperature
        else:
            min_ratio = 1.0
            min_distance = diameter
            soft_min = 1.0

        witnesses: list[dict] = []
        if with_witnesses:
            cutoff = min_ratio + float(witness_window)
            for ratio, distance, coordinate in sorted(
                values, key=lambda item: item[0]
            ):
                if ratio > cutoff:
                    break
                vector = coordinate @ basis
                witnesses.append(
                    {
                        "coordinate": coordinate.astype(int).tolist(),
                        "distance": float(distance),
                        "distance_ratio": float(ratio),
                        "norm_squared": float(vector @ vector),
                    }
                )

        return (
            MetricEvaluation(
                objective=-soft_min,
                soft_min=soft_min,
                min_ratio=min_ratio,
                min_distance=min_distance,
                diameter=diameter,
                facet_count=len(facets),
                vertex_count=vertex_count,
                subvector_count=max(0, len(vectors) - 1),
                h_norm=h_norm,
                parameters=parameters_array,
                basis=basis,
                witnesses=witnesses,
            ),
            cover_witness,
        )


_WORKER: SecondaryConeEvaluator | None = None


def _worker_init(
    basis0: np.ndarray,
    kernel_rows: np.ndarray,
    orders: np.ndarray,
    temperature: float,
    max_h_norm: float,
    cone_margin: float,
) -> None:
    global _WORKER
    _WORKER = SecondaryConeEvaluator(
        basis0,
        kernel_rows,
        orders,
        temperature=temperature,
        max_h_norm=max_h_norm,
        cone_margin=cone_margin,
    )


def _worker_evaluate(parameters: Sequence[float]) -> dict:
    assert _WORKER is not None
    try:
        evaluation, _ = _WORKER.evaluate(parameters)
        return {
            "objective": evaluation.objective,
            "soft_min": evaluation.soft_min,
            "min_ratio": evaluation.min_ratio,
            "diameter": evaluation.diameter,
            "parameters": evaluation.parameters.tolist(),
        }
    except Exception as error:
        return {
            "objective": 1e6,
            "soft_min": -1e6,
            "min_ratio": -1e6,
            "diameter": float("inf"),
            "parameters": list(parameters),
            "error": f"{type(error).__name__}: {error}",
        }


def sample_orders(
    n: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Random full permutations plus deterministic symmetry anchors."""
    keys = rng.random((count, n + 1))
    orders = np.argsort(keys, axis=1).astype(np.uint8)
    anchor_rows = [np.arange(n + 1), np.arange(n, -1, -1)]
    anchor_rows.extend(
        np.roll(np.arange(n + 1), shift) for shift in range(n + 1)
    )
    return np.unique(
        np.vstack((orders, np.asarray(anchor_rows, dtype=np.uint8))),
        axis=0,
    )


def append_order(orders: np.ndarray, order: Sequence[int]) -> tuple[np.ndarray, bool]:
    candidate = np.asarray(order, dtype=orders.dtype)
    if np.any(np.all(orders == candidate, axis=1)):
        return orders, False
    return np.vstack((orders, candidate)), True


def secondary_cone_check(basis: np.ndarray) -> dict:
    """Compare actual relevant facets with the A-star subset-sum catalogue."""
    basis = np.asarray(basis, dtype=np.float64)
    inverse = np.linalg.inv(basis)
    expected = {
        tuple(row)
        for row in permutohedral_facet_coordinates(len(basis)).tolist()
    }
    observed: set[tuple[int, ...]] = set()
    maximum_residual = 0.0
    for normal, _ in combigeo.relevant_facets(basis.tolist()):
        raw = np.asarray(normal) @ inverse
        rounded = np.rint(raw).astype(np.int64)
        maximum_residual = max(
            maximum_residual, float(np.max(np.abs(raw - rounded)))
        )
        observed.add(tuple(rounded.tolist()))
    obtuse = obtuse_superbase_check(basis)
    return {
        "stable": observed == expected and maximum_residual <= 2e-6,
        "expected_facets": len(expected),
        "observed_facets": len(observed),
        "missing_facets": len(expected - observed),
        "new_facets": len(observed - expected),
        "maximum_coordinate_residual": maximum_residual,
        "maximum_superbase_inner_product": obtuse[
            "maximum_off_diagonal_inner_product"
        ],
        "strict_obtuse_superbase": obtuse["feasible"],
    }


def obtuse_superbase_check(
    basis: np.ndarray, *, margin: float = 0.0
) -> dict:
    """Cheap membership test for the permutohedral secondary cone."""
    superbase = superbase_from_astar_basis(basis)
    gram = superbase @ superbase.T
    off_diagonal = gram[~np.eye(len(gram), dtype=bool)]
    maximum = float(np.max(off_diagonal))
    violation = max(0.0, maximum + float(margin))
    return {
        "feasible": violation == 0.0,
        "margin": float(margin),
        "maximum_off_diagonal_inner_product": maximum,
        "violation": violation,
    }


def campaign_records(source: dict) -> tuple[list[dict], str]:
    """Normalize determinant-repair and lazy-CEGAR campaign candidates."""
    records = source.get("best")
    if isinstance(records, list):
        return records, "best[]"

    results = source.get("results")
    if not isinstance(results, list):
        raise ValueError("campaign has neither best[] nor results[].best")
    normalized: list[dict] = []
    for result_index, result in enumerate(results):
        if not isinstance(result, dict) or not isinstance(result.get("best"), dict):
            continue
        record = dict(result["best"])
        separation = record.get("separation")
        if isinstance(separation, dict):
            record.setdefault(
                "distance_ratio",
                separation.get("minimum_distance_ratio"),
            )
            record.setdefault("conflicts", separation.get("conflicts", []))
        record.setdefault("smith", record.get("kernel_smith"))
        record["_campaign_result_index"] = result_index
        normalized.append(record)
    if not normalized:
        raise ValueError("campaign results[] contains no best candidate")
    return normalized, "results[].best"


def exact_lll_kernel_rows(kernel: np.ndarray) -> np.ndarray:
    """Return an exact LLL row basis for a column-HNF sublattice kernel."""
    kernel = np.asarray(kernel, dtype=np.int64)
    if kernel.shape != (9, 9):
        raise ValueError("this optimizer requires a 9 by 9 kernel")
    return np.asarray(Matrix(kernel.T.tolist()).lll().tolist(), dtype=np.int64)


def load_base_metric(
    campaign: Path,
    source: dict,
    explicit: Path | None,
) -> tuple[np.ndarray, str | None]:
    """Load an optional parent metric, including a lazy campaign reference."""
    saved: str | Path | None = explicit
    if saved is None and source.get("source_metric"):
        saved = str(source["source_metric"])
    if saved is None:
        return Astar(9), None

    path = resolve_saved_path(campaign, saved)
    payload = json.loads(path.read_text())
    try:
        basis = np.asarray(payload["best"]["basis"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid base metric JSON {path}: {error}") from error
    if basis.shape != (9, 9) or not np.all(np.isfinite(basis)):
        raise ValueError("base metric must contain a finite 9 by 9 best.basis")
    return basis, str(path)


def make_payload(
    *,
    source: Path,
    source_record: dict,
    source_container: str,
    base_metric: str | None,
    kernel: np.ndarray,
    best: MetricEvaluation,
    cover_witness: dict | None,
    cone_check: dict,
    optimizer: dict,
    generation: int,
    evaluations: int,
    full_evaluations: int,
    elapsed: float,
    adversarial_orders: list[list[int]],
) -> dict:
    return {
        "method": "A9-star secondary-cone permutation-archive CEGAR",
        "lattice": "A9*",
        "source_campaign": str(source),
        "base_metric": base_metric,
        "source_record": {
            "candidate_rank": optimizer["candidate_rank"],
            "candidate_container": source_container,
            "campaign_result_index": source_record.get(
                "_campaign_result_index"
            ),
            "distance_ratio": source_record.get("distance_ratio"),
            "smith": source_record.get("smith"),
        },
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "generation": generation,
        "surrogate_evaluations": evaluations,
        "complete_permutation_evaluations": full_evaluations,
        "elapsed_seconds": round(elapsed, 6),
        "optimizer": optimizer,
        "covering_oracle": {
            "secondary_cone": "A9-star permutohedral",
            "complete_vertex_count": math.factorial(10),
            "sampled_archive_size": optimizer["vertex_samples"],
            "adversarial_orders": adversarial_orders,
            "worst_vertex": cover_witness,
            "secondary_cone_check": cone_check,
        },
        "best": best.as_json(),
        "valid_numerical_witness": (
            best.min_ratio >= 1.0 and bool(cone_check["stable"])
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--base-metric",
        type=Path,
        help=(
            "metric JSON supplying best.basis; by default inherit "
            "source_metric from a lazy campaign"
        ),
    )
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=14)
    parser.add_argument("--sigma", type=float, default=0.025)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--initial-stretch", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--max-h-norm", type=float, default=0.45)
    parser.add_argument(
        "--cone-margin",
        type=float,
        default=1e-7,
        help="required negativity margin for off-diagonal superbase products",
    )
    parser.add_argument("--vertex-samples", type=int, default=40_000)
    parser.add_argument("--refresh-every", type=int, default=4)
    parser.add_argument("--full-checks", type=int, default=1)
    parser.add_argument("--target-margin", type=float, default=2e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(
        args.generations,
        args.population,
        args.workers,
        args.vertex_samples,
        args.refresh_every,
        args.full_checks,
    ) < 1:
        parser.error("all budgets must be positive")
    if args.cone_margin < 0:
        parser.error("cone margin must be nonnegative")

    source = json.loads(args.campaign.read_text())
    try:
        records, source_container = campaign_records(source)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.candidate_rank < len(records):
        parser.error("candidate rank is outside normalized campaign candidates")
    record = records[args.candidate_rank]
    kernel = np.asarray(record["kernel_basis_columns"], dtype=np.int64)
    if kernel.shape != (9, 9):
        parser.error("this optimizer requires a 9 by 9 kernel")
    kernel_rows = exact_lll_kernel_rows(kernel)
    kernel_determinant = abs(int(Matrix(kernel.tolist()).det()))
    row_determinant = abs(int(Matrix(kernel_rows.tolist()).det()))
    if row_determinant != kernel_determinant:
        parser.error("LLL row basis and HNF kernel have different index")
    try:
        basis0, base_metric = load_base_metric(
            args.campaign, source, args.base_metric
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    rng = np.random.default_rng(args.seed)
    orders = sample_orders(9, args.vertex_samples, rng)
    adversarial_orders: list[list[int]] = []
    full_evaluator = SecondaryConeEvaluator(
        basis0,
        kernel_rows,
        orders,
        temperature=args.temperature,
        max_h_norm=args.max_h_norm,
        cone_margin=args.cone_margin,
    )
    zero = np.zeros(44, dtype=np.float64)
    baseline, baseline_cover = full_evaluator.evaluate(
        zero, full_cover=True, with_witnesses=True
    )
    initial = span_stretch_seed(basis0, record, args.initial_stretch)
    seeded, seeded_cover = full_evaluator.evaluate(
        initial, full_cover=True, with_witnesses=True
    )
    baseline_cone = secondary_cone_check(baseline.basis)
    seeded_cone = secondary_cone_check(seeded.basis)
    if not baseline_cone["stable"]:
        raise AssertionError("the A9-star baseline left its own secondary cone")
    source_ratio = record.get("distance_ratio")
    if source_ratio is not None and not math.isclose(
        baseline.min_ratio,
        float(source_ratio),
        rel_tol=2e-8,
        abs_tol=2e-8,
    ):
        raise AssertionError(
            "baseline ratio does not reproduce the source candidate: "
            f"{baseline.min_ratio:.12g} != {float(source_ratio):.12g}"
        )
    best, best_cover = baseline, baseline_cover
    if seeded_cone["stable"] and seeded.min_ratio > baseline.min_ratio:
        best, best_cover = seeded, seeded_cover
    initial = best.parameters.copy()
    if best_cover is not None:
        orders, added = append_order(orders, best_cover["permutation"])
        if added:
            adversarial_orders.append(best_cover["permutation"])

    print(
        f"baseline full-min={baseline.min_ratio:.12f} "
        f"diam={baseline.diameter:.12f}; "
        f"seed full-min={seeded.min_ratio:.12f} "
        f"seed-cone={seeded_cone['stable']} "
        f"archive={len(orders)}",
        flush=True,
    )
    strategy = cma.CMAEvolutionStrategy(
        initial.tolist(),
        args.sigma,
        {
            "popsize": args.population,
            "maxiter": args.generations,
            "seed": args.seed,
            "verbose": -9,
            "tolfun": 1e-11,
            "tolx": 1e-9,
        },
    )
    optimizer = {
        "generations_budget": args.generations,
        "population": args.population,
        "sigma": args.sigma,
        "workers": args.workers,
        "seed": args.seed,
        "initial_stretch": args.initial_stretch,
        "temperature": args.temperature,
        "max_h_norm": args.max_h_norm,
        "cone_margin": args.cone_margin,
        "vertex_samples": args.vertex_samples,
        "refresh_every": args.refresh_every,
        "full_checks": args.full_checks,
        "candidate_rank": args.candidate_rank,
        "candidate_container": source_container,
        "base_metric": base_metric,
        "source_ratio_reproduced": (
            source_ratio is None
            or math.isclose(
                baseline.min_ratio,
                float(source_ratio),
                rel_tol=2e-8,
                abs_tol=2e-8,
            )
        ),
    }
    total_evaluations = 0
    full_evaluations = 2
    unstable_full_evaluations = 0
    best_unstable_ratio = -math.inf
    started = time.perf_counter()
    generation = 0

    target_reached = False
    while (
        not strategy.stop()
        and generation < args.generations
        and not target_reached
    ):
        epoch_start = generation + 1
        epoch_end = min(
            args.generations,
            epoch_start + args.refresh_every - 1,
        )
        with mp.Pool(
            max(1, args.workers),
            initializer=_worker_init,
            initargs=(
                basis0,
                kernel_rows,
                orders,
                args.temperature,
                args.max_h_norm,
                args.cone_margin,
            ),
        ) as pool:
            # Keep the pool for one CEGAR epoch.  It is deliberately rebuilt
            # after a complete scan adds an adversarial permutation.
            for current_generation in range(epoch_start, epoch_end + 1):
                if strategy.stop():
                    break
                generation = current_generation
                population = strategy.ask()
                summaries = pool.map(_worker_evaluate, population)
                objectives = [item["objective"] for item in summaries]
                strategy.tell(population, objectives)
                total_evaluations += len(population)
                surrogate_best = max(
                    item["min_ratio"] for item in summaries
                )
                print(
                    f"gen {generation:3d}/{args.generations}: "
                    f"surrogate-min={surrogate_best:.9f} "
                    f"full-best={best.min_ratio:.9f} "
                    f"archive={len(orders)} elapsed="
                    f"{time.perf_counter()-started:.1f}s",
                    flush=True,
                )
                if generation == epoch_end:
                    ranked = sorted(
                        summaries,
                        key=lambda item: item["min_ratio"],
                        reverse=True,
                    )[: args.full_checks]
                    for summary in ranked:
                        full_evaluator.orders = orders
                        candidate, cover = full_evaluator.evaluate(
                            summary["parameters"],
                            full_cover=True,
                            with_witnesses=True,
                        )
                        full_evaluations += 1
                        candidate_cone = secondary_cone_check(candidate.basis)
                        if cover is not None:
                            orders, added = append_order(
                                orders, cover["permutation"]
                            )
                            if added:
                                adversarial_orders.append(
                                    cover["permutation"]
                                )
                        if not candidate_cone["stable"]:
                            unstable_full_evaluations += 1
                            best_unstable_ratio = max(
                                best_unstable_ratio, candidate.min_ratio
                            )
                        elif candidate.min_ratio > best.min_ratio:
                            best, best_cover = candidate, cover
                            print(
                                f"  new complete best: "
                                f"min={best.min_ratio:.12f} "
                                f"diam={best.diameter:.12f} "
                                f"|H|={best.h_norm:.5f}",
                                flush=True,
                            )
                        print(
                            f"  complete candidate: "
                            f"min={candidate.min_ratio:.12f} "
                            f"cone-stable={candidate_cone['stable']}",
                            flush=True,
                        )
                    cone = secondary_cone_check(best.basis)
                    optimizer["unstable_full_evaluations"] = (
                        unstable_full_evaluations
                    )
                    optimizer["best_unstable_ratio"] = (
                        best_unstable_ratio
                        if math.isfinite(best_unstable_ratio)
                        else None
                    )
                    payload = make_payload(
                        source=args.campaign,
                        source_record=record,
                        source_container=source_container,
                        base_metric=base_metric,
                        kernel=kernel,
                        best=best,
                        cover_witness=best_cover,
                        cone_check=cone,
                        optimizer=optimizer,
                        generation=generation,
                        evaluations=total_evaluations,
                        full_evaluations=full_evaluations,
                        elapsed=time.perf_counter() - started,
                        adversarial_orders=adversarial_orders,
                    )
                    args.output.write_text(json.dumps(payload, indent=2) + "\n")
                    print(
                        f"  full scan: min={best.min_ratio:.12f} "
                        f"cone-stable={cone['stable']} "
                        f"adversarial={len(adversarial_orders)}",
                        flush=True,
                    )
                    if (
                        best.min_ratio >= 1.0 + args.target_margin
                        and cone["stable"]
                    ):
                        print(
                            "*** numerical separation target reached ***",
                            flush=True,
                        )
                        target_reached = True
                        break

    final, final_cover = full_evaluator.evaluate(
        best.parameters, full_cover=True, with_witnesses=True
    )
    full_evaluations += 1
    cone = secondary_cone_check(final.basis)
    payload = make_payload(
        source=args.campaign,
        source_record=record,
        source_container=source_container,
        base_metric=base_metric,
        kernel=kernel,
        best=final,
        cover_witness=final_cover,
        cone_check=cone,
        optimizer=optimizer,
        generation=generation,
        evaluations=total_evaluations,
        full_evaluations=full_evaluations,
        elapsed=time.perf_counter() - started,
        adversarial_orders=adversarial_orders,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL min={final.min_ratio:.12f} D={final.min_distance:.12f} "
        f"diam={final.diameter:.12f} cone-stable={cone['stable']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
