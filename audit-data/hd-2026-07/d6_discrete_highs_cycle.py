"""Alternate modular kernel best responses with the HiGHS PSD master.

``d6_highs_psd_outer.py`` can cheaply sample many Gram forms in distinct
L-type sign chambers.  A fixed kernel is locally optimal across those
chambers, but the changed geometry supplies different weights to the modular
search.  This script closes the loop:

1. select geometrically distinct probe metrics from one or more HiGHS
   chamber portfolios;
2. run complete-row weighted best responses for the quotient
   ``[7,4,4,3]`` on every probe;
3. deduplicate the resulting exact kernels by HNF;
4. optimize the strongest new kernels with repeated HiGHS
   PSD/eigenvector-cut passes;
5. check every continuous incumbent with the complete geometry oracle.

This is a numerical discovery loop, not an impossibility proof or a coloring
certificate.  Any ratio crossing one must still pass rational reconstruction
and independent exact verification.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import combigeo
from active_metric_refine import _load_problem
from block_row_metric_opt import candidate_record, parse_powers
from d6_highs_psd_outer import solve_psd_outer
from d6_sdp_hybrid import (
    circumradius_squared,
    determinant_rescale,
    exact_coordinate_separations,
    kernel_coordinates_within,
    parameters_for_gram,
    projection_certificate,
    relevant_coordinate_rows,
    triangulation_orbits,
)
from metric_deform import MetricEvaluator
from prime_radon import (
    PrimarySearch,
    hnf_columns,
    image_size,
    kernel_basis,
    smith_diagonal,
)
from prime_row_opt import _forbidden_with_weights


def kernel_key(kernel: np.ndarray) -> tuple[int, ...]:
    """Canonical hashable key for an HNF kernel basis."""
    canonical = hnf_columns(np.asarray(kernel, dtype=np.int64))
    return tuple(int(value) for value in canonical.flat)


def select_probes(
    portfolios: Sequence[tuple[Path, dict]],
    source_parameters: np.ndarray,
    count: int,
) -> list[dict]:
    """Select high-ratio and far-deformation probes from each portfolio."""
    if count < 1:
        raise ValueError("probe count must be positive")
    source_parameters = np.asarray(source_parameters, dtype=np.float64)
    raw: list[dict] = []
    for path, payload in portfolios:
        for branch in payload.get("branches", []):
            oracle = branch.get("oracle")
            if not isinstance(oracle, dict) or "parameters" not in oracle:
                continue
            parameters = np.asarray(
                oracle["parameters"], dtype=np.float64
            )
            if parameters.shape != source_parameters.shape:
                continue
            raw.append(
                {
                    "portfolio": str(path),
                    "label": branch.get("label"),
                    "voronoi_signature": branch.get("voronoi_signature"),
                    "min_ratio": float(oracle["min_ratio"]),
                    "parameters": parameters,
                    "parameter_distance": float(
                        np.linalg.norm(parameters - source_parameters)
                    ),
                }
            )
    if not raw:
        raise ValueError("HiGHS portfolios contain no usable probe metrics")

    # The same signature at two wall depths can have materially different
    # modular weights, so deduplicate only within one portfolio.
    unique: dict[tuple[str, str], dict] = {}
    for probe in raw:
        key = (
            probe["portfolio"],
            str(probe["voronoi_signature"]),
        )
        incumbent = unique.get(key)
        if (
            incumbent is None
            or probe["min_ratio"] > incumbent["min_ratio"]
        ):
            unique[key] = probe
    candidates = list(unique.values())
    chosen: list[dict] = []
    chosen_keys: set[tuple[str, str]] = set()

    def take(order: Sequence[dict], target: int) -> None:
        for probe in order:
            key = (
                probe["portfolio"],
                str(probe["voronoi_signature"]),
            )
            if key in chosen_keys:
                continue
            chosen.append(probe)
            chosen_keys.add(key)
            if len(chosen) >= target:
                return

    high_target = max(1, (count + 1) // 2)
    take(
        sorted(candidates, key=lambda item: item["min_ratio"], reverse=True),
        high_target,
    )
    take(
        sorted(
            candidates,
            key=lambda item: item["parameter_distance"],
            reverse=True,
        ),
        count,
    )
    take(
        sorted(candidates, key=lambda item: item["min_ratio"], reverse=True),
        count,
    )
    for probe in chosen:
        probe["parameters"] = probe["parameters"].tolist()
    return chosen[:count]


def _continuous_outer_pass(
    evaluator: MetricEvaluator,
    parameters: np.ndarray,
    *,
    projection_solver: str,
    outer_rounds: int,
    cuts_per_round: int,
    violation_tolerance: float,
    positive_floor: float,
    gram_bound_factor: float,
) -> tuple[dict, np.ndarray | None]:
    """Run one fixed-dual HiGHS/PSD pass for an arbitrary kernel."""
    current = evaluator.evaluate(parameters, with_witnesses=True)
    gram = current.basis @ current.basis.T
    simplices = triangulation_orbits(current.basis)
    facets = relevant_coordinate_rows(current.basis)
    coordinates = kernel_coordinates_within(
        current.basis,
        evaluator.kernel,
        2.0 * current.diameter + 1e-8,
    )
    distances, _ = exact_coordinate_separations(
        current.basis, coordinates
    )
    certificates: list[np.ndarray] = []
    certificate_values: list[float] = []
    for index in np.argsort(distances):
        coordinate = coordinates[int(index)]
        certificate, dual, primal = projection_certificate(
            gram,
            coordinate,
            facets,
            solver=projection_solver,
        )
        if abs(primal - distances[int(index)] ** 2) > max(
            2e-6, 3e-6 * primal
        ):
            raise RuntimeError(
                "projection QP and complete Voronoi distance disagree"
            )
        certificates.append(certificate)
        certificate_values.append(dual)
    minimum_certificate = min(certificate_values)
    warm_gram = gram / minimum_certificate
    warm_rho = max(
        circumradius_squared(warm_gram, rows) for rows in simplices
    )
    outer = solve_psd_outer(
        len(gram),
        simplices,
        certificates,
        warm_gram=warm_gram,
        warm_rho=warm_rho,
        positive_floor=positive_floor,
        max_rounds=outer_rounds,
        cuts_per_round=cuts_per_round,
        violation_tolerance=violation_tolerance,
        gram_bound_factor=gram_bound_factor,
    )
    record = {
        "start_ratio": current.min_ratio,
        "simplex_orbits": len(simplices),
        "separation_certificates": len(certificates),
        "success": outer["success"],
        "converged_outer_psd": outer["converged"],
        "highs_status": outer["status"],
        "outer_history": outer["history"],
        "outer_elapsed_seconds": outer["elapsed_seconds"],
        "linear_cuts": outer["linear_cuts"],
    }
    if not outer["success"]:
        return record, None
    target_determinant = float(
        np.linalg.det(evaluator.basis0 @ evaluator.basis0.T)
    )
    output_gram = determinant_rescale(
        outer["gram"], target_determinant
    )
    output_parameters = parameters_for_gram(
        evaluator.basis0, output_gram
    )
    result = evaluator.evaluate(
        output_parameters, with_witnesses=True
    )
    record.update(
        {
            "finite_outer_ratio": outer["finite_outer_ratio"],
            "result": result.as_json(),
        }
    )
    return record, output_parameters


def optimize_kernel(
    basis0: np.ndarray,
    kernel: np.ndarray,
    start_parameters: np.ndarray,
    *,
    passes: int,
    temperature: float,
    max_h_norm: float,
    projection_solver: str,
    outer_rounds: int,
    cuts_per_round: int,
    violation_tolerance: float,
    positive_floor: float,
    gram_bound_factor: float,
) -> dict:
    """Repeatedly refresh dual certificates and optimize one exact kernel."""
    evaluator = MetricEvaluator(
        basis0,
        kernel,
        softmin_temperature=temperature,
        max_h_norm=max_h_norm,
    )
    parameters = np.asarray(start_parameters, dtype=np.float64)
    initial = evaluator.evaluate(parameters, with_witnesses=True)
    best = initial
    best_parameters = parameters.copy()
    history: list[dict] = []
    for pass_number in range(1, passes + 1):
        record, candidate_parameters = _continuous_outer_pass(
            evaluator,
            parameters,
            projection_solver=projection_solver,
            outer_rounds=outer_rounds,
            cuts_per_round=cuts_per_round,
            violation_tolerance=violation_tolerance,
            positive_floor=positive_floor,
            gram_bound_factor=gram_bound_factor,
        )
        record["pass"] = pass_number
        history.append(record)
        if candidate_parameters is None:
            break
        candidate = evaluator.evaluate(
            candidate_parameters, with_witnesses=True
        )
        parameters = candidate_parameters
        if candidate.min_ratio > best.min_ratio:
            best = candidate
            best_parameters = candidate_parameters.copy()
        if (
            len(history) >= 2
            and abs(
                history[-1]["result"]["min_ratio"]
                - history[-2]["result"]["min_ratio"]
            )
            < 2e-9
        ):
            break
    best = evaluator.evaluate(best_parameters, with_witnesses=True)
    return {
        "kernel_basis_columns": np.asarray(
            kernel, dtype=np.int64
        ).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "initial": initial.as_json(),
        "history": history,
        "best": best.as_json(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("portfolios", nargs="+", type=Path)
    parser.add_argument("--probes", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=120)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--archive-per-search",
        type=int,
        default=1,
        help="number of HNF-distinct local optima retained per probe/power",
    )
    parser.add_argument(
        "--weight-powers",
        type=parse_powers,
        default=[2.0, 4.0, 8.0],
    )
    parser.add_argument("--weighted-pair-top", type=int, default=0)
    parser.add_argument("--continuous-candidates", type=int, default=12)
    parser.add_argument("--continuous-passes", type=int, default=3)
    parser.add_argument("--outer-rounds", type=int, default=40)
    parser.add_argument("--cuts-per-round", type=int, default=192)
    parser.add_argument("--violation-tolerance", type=float, default=2e-8)
    parser.add_argument("--positive-floor", type=float, default=1e-7)
    parser.add_argument("--gram-bound-factor", type=float, default=16.0)
    parser.add_argument(
        "--projection-solver",
        choices=("CLARABEL", "SCS"),
        default="CLARABEL",
    )
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=6339201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.probes < 1
        or args.restarts < 0
        or args.sweeps < 1
        or args.top < 1
        or args.archive_per_search < 1
        or args.weighted_pair_top < 0
        or args.continuous_candidates < 1
        or args.continuous_passes < 1
        or args.outer_rounds < 1
        or args.cuts_per_round < 1
        or args.violation_tolerance <= 0
        or args.positive_floor <= 0
        or args.gram_bound_factor <= 1
    ):
        parser.error("invalid discrete/continuous cycle budget")

    (
        metric_payload,
        source_path,
        base_metric,
        source_record,
        source_kernel,
        source_evaluator,
    ) = _load_problem(args.metric, args.temperature, args.max_h_norm)
    source_parameters = np.asarray(
        metric_payload["best"]["parameters"], dtype=np.float64
    )
    source_evaluation = source_evaluator.evaluate(
        source_parameters, with_witnesses=True
    )
    rows = [
        np.asarray(row, dtype=np.int64)
        for row in source_record["rows"]
    ]
    moduli = [int(value) for value in source_record["moduli"]]
    if image_size(rows, moduli, len(source_kernel)) != math.prod(moduli):
        raise RuntimeError("source rows do not have the requested image")

    portfolio_payloads = [
        (path, json.loads(path.read_text())) for path in args.portfolios
    ]
    probes = select_probes(
        portfolio_payloads, source_parameters, args.probes
    )
    source_payload = json.loads(source_path.read_text())
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "L-type chamber metric probes, modular weighted best responses, "
            "and repeated HiGHS PSD outer optimization"
        ),
        "lattice": source_payload["lattice"],
        "n": len(source_kernel),
        "dimension": len(source_kernel),
        "source_metric": str(args.metric),
        "source_campaign": str(source_path),
        "base_metric": (
            str(base_metric) if base_metric is not None else None
        ),
        "portfolios": [str(path) for path in args.portfolios],
        "source_record": {
            "moduli": moduli,
            "rows": [row.astype(int).tolist() for row in rows],
            "image_index": math.prod(moduli),
            "beta": source_record.get("beta"),
        },
        "kernel_basis_columns": source_kernel.astype(int).tolist(),
        "source": source_evaluation.as_json(),
        "settings": {
            "probes": args.probes,
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "archive_per_search": args.archive_per_search,
            "weight_powers": args.weight_powers,
            "weighted_pair_top": args.weighted_pair_top,
            "continuous_candidates": args.continuous_candidates,
            "continuous_passes": args.continuous_passes,
            "outer_rounds": args.outer_rounds,
            "cuts_per_round": args.cuts_per_round,
            "violation_tolerance": args.violation_tolerance,
            "positive_floor": args.positive_floor,
            "gram_bound_factor": args.gram_bound_factor,
            "projection_solver": args.projection_solver,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
            "seed": args.seed,
        },
        "probes": probes,
        "discrete_candidates": [],
        "continuous_candidates": [],
        "best_alternative": None,
        "best": source_evaluation.as_json(),
        "valid_numerical_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"source={source_evaluation.min_ratio:.12f} "
        f"probes={len(probes)} powers={args.weight_powers}",
        flush=True,
    )
    for probe_index, probe in enumerate(probes):
        parameters = np.asarray(probe["parameters"], dtype=np.float64)
        evaluation = source_evaluator.evaluate(parameters)
        basis = evaluation.basis
        diameter = evaluation.diameter
        facets = combigeo.relevant_facets(basis.tolist())
        forbidden, ratios, _ = _forbidden_with_weights(
            basis, diameter
        )
        deficits = np.maximum(0.0, 1.0 - ratios)
        probe["reproduced_ratio"] = evaluation.min_ratio
        probe["forbidden_projective_pairs"] = len(forbidden)
        print(
            f"probe {probe_index + 1:2d}/{len(probes)} "
            f"{probe['label']} ratio={evaluation.min_ratio:.12f} "
            f"forbidden={len(forbidden)}",
            flush=True,
        )
        for power_index, power in enumerate(args.weight_powers):
            weights = np.power(deficits, power)
            search_started = time.perf_counter()
            search = PrimarySearch(
                forbidden,
                moduli,
                seed=(
                    args.seed
                    + 1009 * probe_index
                    + 104729 * (power_index + 1)
                ),
            )
            archive = search.run_weighted_archive(
                weights,
                archive_size=args.archive_per_search,
                restarts=args.restarts,
                max_sweeps=args.sweeps,
                top=args.top,
                progress_every=max(1, (args.restarts + 1) // 3),
                initial_rows=rows,
            )
            records: list[dict] = []
            for archive_rank, result in enumerate(archive):
                record = candidate_record(
                    label=(
                        f"probe-{probe_index}-"
                        f"{probe['label']}-power-{power:g}-"
                        f"archive-{archive_rank}"
                    ),
                    beta=float(power),
                    rows=result.rows,
                    moduli=moduli,
                    forbidden=forbidden,
                    ratios=ratios,
                    weights=weights,
                    basis=basis,
                    diameter=diameter,
                    facets=facets,
                    search_seconds=time.perf_counter() - search_started,
                    search_metadata=result.as_json(),
                )
                record["probe_index"] = probe_index
                record["probe_label"] = probe["label"]
                record["probe_portfolio"] = probe["portfolio"]
                record["archive_rank"] = archive_rank
                payload["discrete_candidates"].append(record)
                records.append(record)
            save()
            record = records[0]
            print(
                f"  p={power:g} archive={len(records)} "
                f"killed={record['killed']} "
                f"loss={record['weighted_loss']:.6g} "
                f"min={record['minimum_conflict_ratio']}",
                flush=True,
            )
            if args.weighted_pair_top:
                pair_started = time.perf_counter()
                _, pair_rows = search.pair_polish_weighted(
                    archive[0].rows,
                    weights,
                    first_top=args.weighted_pair_top,
                )
                pair_record = candidate_record(
                    label=record["label"] + "-pair",
                    beta=float(power),
                    rows=pair_rows,
                    moduli=moduli,
                    forbidden=forbidden,
                    ratios=ratios,
                    weights=weights,
                    basis=basis,
                    diameter=diameter,
                    facets=facets,
                    search_seconds=time.perf_counter() - pair_started,
                    search_metadata={
                        "weighted_pair_top": args.weighted_pair_top,
                        "source": record["label"],
                    },
                )
                pair_record["probe_index"] = probe_index
                pair_record["probe_label"] = probe["label"]
                pair_record["probe_portfolio"] = probe["portfolio"]
                payload["discrete_candidates"].append(pair_record)
                save()

    source_key = kernel_key(source_kernel)
    unique: dict[tuple[int, ...], dict] = {}
    for record in payload["discrete_candidates"]:
        raw_kernel = record.get("kernel_basis_columns")
        if raw_kernel is None:
            continue
        key = kernel_key(np.asarray(raw_kernel, dtype=np.int64))
        if key == source_key:
            continue
        incumbent = unique.get(key)
        candidate_ratio = float(
            record.get("minimum_conflict_ratio") or -1.0
        )
        if (
            incumbent is None
            or candidate_ratio
            > float(incumbent.get("minimum_conflict_ratio") or -1.0)
        ):
            unique[key] = record
    ranked = sorted(
        unique.values(),
        key=lambda record: (
            float(record.get("minimum_conflict_ratio") or -1.0),
            -int(record.get("killed", 0)),
        ),
        reverse=True,
    )[: args.continuous_candidates]
    payload["unique_alternative_kernels"] = len(unique)
    save()
    print(
        f"unique alternative kernels={len(unique)} "
        f"continuous queue={len(ranked)}",
        flush=True,
    )

    best_alternative: dict | None = None
    for candidate_index, record in enumerate(ranked):
        kernel = hnf_columns(
            np.asarray(
                record["kernel_basis_columns"], dtype=np.int64
            )
        )
        probe = probes[int(record["probe_index"])]
        optimized = optimize_kernel(
            source_evaluator.basis0,
            kernel,
            np.asarray(probe["parameters"], dtype=np.float64),
            passes=args.continuous_passes,
            temperature=args.temperature,
            max_h_norm=args.max_h_norm,
            projection_solver=args.projection_solver,
            outer_rounds=args.outer_rounds,
            cuts_per_round=args.cuts_per_round,
            violation_tolerance=args.violation_tolerance,
            positive_floor=args.positive_floor,
            gram_bound_factor=args.gram_bound_factor,
        )
        optimized["candidate_rank"] = candidate_index
        optimized["source_discrete_record"] = {
            "label": record["label"],
            "beta": record["beta"],
            "moduli": record["moduli"],
            "rows": record["rows"],
            "image_index": record["image_index"],
            "minimum_conflict_ratio": record[
                "minimum_conflict_ratio"
            ],
            "killed": record["killed"],
            "probe_index": record["probe_index"],
        }
        payload["continuous_candidates"].append(optimized)
        if (
            best_alternative is None
            or optimized["best"]["min_ratio"]
            > best_alternative["best"]["min_ratio"]
        ):
            best_alternative = optimized
        save()
        print(
            f"continuous {candidate_index + 1:2d}/{len(ranked)} "
            f"start={optimized['initial']['min_ratio']:.12f} "
            f"best={optimized['best']['min_ratio']:.12f}",
            flush=True,
        )

    if best_alternative is not None:
        payload["best_alternative"] = best_alternative
        if (
            best_alternative["best"]["min_ratio"]
            > source_evaluation.min_ratio
        ):
            payload["best"] = best_alternative["best"]
            payload["source_record"] = {
                **best_alternative["source_discrete_record"],
            }
            payload["kernel_basis_columns"] = best_alternative[
                "kernel_basis_columns"
            ]
    payload["valid_numerical_witness"] = (
        payload["best"]["min_ratio"] >= 1.0
    )
    save()
    print(
        f"FINAL source={source_evaluation.min_ratio:.12f} "
        f"best={payload['best']['min_ratio']:.12f} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
