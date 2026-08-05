"""Sequential SDP/cutting-plane optimization for affine color differences.

This is the affine-coset analogue of ``d6_sdp_hybrid.py``.  A fixed
triangulation gives rigorous linear-matrix-inequality upper bounds on the
covering radius.  Frozen projection duals give linear lower bounds on the
separation of every currently relevant displacement in

    {z : row*z is 0,+y,-y modulo N}.

After every conic master solve, complete affine enumeration adds newly
violating displacements and the full simplex catalogue adds missing covering
constraints.  The final candidate is always checked by the complete Voronoi
oracle.  Results remain numerical until rationalized independently.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from d6_affine_metric_opt import (
    AffineMetricEvaluator,
    affine_coordinates_within,
    checkpoint_affine_cosets,
)
from d6_cyclic_hole_search import primitive_cyclic_row
from d6_sdp_hybrid import (
    circumradius_squared,
    determinant_rescale,
    exact_coordinate_separations,
    parameters_for_gram,
    projection_certificate,
    relevant_coordinate_rows,
    solve_sdp,
    triangulation_orbits,
)
from determinant_repair import exact_det, load_preset
from prime_radon import hnf_columns, kernel_basis, smith_diagonal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--initial-simplices", type=int, default=48)
    parser.add_argument("--add-simplices", type=int, default=48)
    parser.add_argument("--add-affine-vectors", type=int, default=24)
    parser.add_argument(
        "--solver",
        choices=("CLARABEL", "SCS"),
        default="CLARABEL",
    )
    parser.add_argument("--solver-tolerance", type=float, default=2e-7)
    parser.add_argument("--solver-iterations", type=int, default=300)
    parser.add_argument("--positive-floor", type=float, default=1e-7)
    parser.add_argument("--violation-tolerance", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.rounds < 1
        or args.initial_simplices < 1
        or args.add_simplices < 1
        or args.add_affine_vectors < 1
        or args.solver_tolerance <= 0
        or args.solver_iterations < 1
        or args.positive_floor <= 0
        or args.violation_tolerance <= 0
        or args.temperature <= 0
        or args.max_h_norm <= 0
    ):
        parser.error("all SDP and cutting-plane budgets must be positive")

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
    source_evaluation = evaluator.evaluate(center, with_witnesses=True)
    recorded_ratio = float(metric_payload["best"]["min_ratio"])
    tolerance = max(5e-8, 5e-7 * abs(recorded_ratio))
    if abs(source_evaluation.min_ratio - recorded_ratio) > tolerance:
        parser.error(
            "source metric mismatch: recomputed "
            f"{source_evaluation.min_ratio:.12g}, recorded "
            f"{recorded_ratio:.12g}"
        )

    source_gram = (
        source_evaluation.basis @ source_evaluation.basis.T
    )
    simplices = triangulation_orbits(source_evaluation.basis)
    source_radii = np.asarray(
        [
            circumradius_squared(source_gram, rows)
            for rows in simplices
        ]
    )
    source_facets = relevant_coordinate_rows(source_evaluation.basis)
    source_coordinates = affine_coordinates_within(
        source_evaluation.basis,
        period,
        cosets,
        2.0 * source_evaluation.diameter + 1e-8,
    )
    source_distances, _ = exact_coordinate_separations(
        source_evaluation.basis,
        source_coordinates,
    )
    source_order = np.argsort(source_distances)
    certificates: dict[tuple[int, ...], np.ndarray] = {}
    certificate_values: dict[tuple[int, ...], float] = {}
    for index in source_order:
        coordinate = source_coordinates[int(index)]
        certificate, dual, primal = projection_certificate(
            source_gram,
            coordinate,
            source_facets,
            solver=args.solver,
        )
        if abs(primal - source_distances[int(index)] ** 2) > max(
            2e-6,
            3e-6 * primal,
        ):
            raise RuntimeError(
                "projection QP and complete affine distance disagree"
            )
        key = tuple(int(value) for value in coordinate)
        certificates[key] = certificate
        certificate_values[key] = dual
    if not certificates:
        raise RuntimeError("source affine enumeration is unexpectedly empty")

    minimum_certificate = min(certificate_values.values())
    gram = source_gram / minimum_certificate
    all_radii = source_radii / minimum_certificate
    rho = float(all_radii.max())
    selected_simplex_ids = set(
        int(index)
        for index in np.argsort(all_radii)[
            -min(args.initial_simplices, len(all_radii)) :
        ]
    )
    history: list[dict] = []
    converged = False
    started = time.perf_counter()
    print(
        f"source ratio={source_evaluation.min_ratio:.12f} "
        f"simplices={len(simplices)} affine-vectors={len(certificates)} "
        f"rho={rho:.12g}",
        flush=True,
    )

    for round_number in range(1, args.rounds + 1):
        gram, master_rho, status, solve_seconds = solve_sdp(
            len(gram),
            [simplices[index] for index in sorted(selected_simplex_ids)],
            list(certificates.values()),
            solver=args.solver,
            positive_floor=args.positive_floor,
            solver_tolerance=args.solver_tolerance,
            max_iterations=args.solver_iterations,
            warm_gram=gram,
            warm_rho=rho,
        )
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[0] <= 0:
            raise RuntimeError("SDP returned a non-positive Gram form")
        basis = np.linalg.cholesky(gram)
        simplex_radii = np.asarray(
            [
                circumradius_squared(gram, rows)
                for rows in simplices
            ]
        )
        covering_rho = float(simplex_radii.max())
        simplex_violations = [
            int(index)
            for index in np.argsort(simplex_radii)[::-1]
            if int(index) not in selected_simplex_ids
            and simplex_radii[int(index)]
            > master_rho * (1.0 + args.violation_tolerance)
        ]

        safe_norm_radius = (
            1.0 + 2.0 * math.sqrt(covering_rho) + 2e-6
        )
        coordinates = affine_coordinates_within(
            basis,
            period,
            cosets,
            safe_norm_radius,
        )
        distances, _ = exact_coordinate_separations(
            basis,
            coordinates,
        )
        violating_ids = [
            int(index)
            for index in np.argsort(distances)
            if distances[int(index)] ** 2
            < 1.0 - args.violation_tolerance
            and tuple(int(value) for value in coordinates[int(index)])
            not in certificates
        ]
        new_simplex_ids = simplex_violations[: args.add_simplices]
        selected_simplex_ids.update(new_simplex_ids)
        new_coordinates = [
            coordinates[index]
            for index in violating_ids[: args.add_affine_vectors]
        ]
        if new_coordinates:
            current_facets = relevant_coordinate_rows(basis)
            for coordinate in new_coordinates:
                certificate, dual, _ = projection_certificate(
                    gram,
                    coordinate,
                    current_facets,
                    solver=args.solver,
                )
                key = tuple(int(value) for value in coordinate)
                certificates[key] = certificate
                certificate_values[key] = dual

        minimum_distance = (
            float(distances.min()) if len(distances) else float("inf")
        )
        bound_ratio = 1.0 / (2.0 * math.sqrt(covering_rho))
        history.append(
            {
                "round": round_number,
                "status": status,
                "solve_seconds": solve_seconds,
                "master_rho": master_rho,
                "covering_rho": covering_rho,
                "certified_ratio_if_separated": bound_ratio,
                "minimum_gram_eigenvalue": float(eigenvalues[0]),
                "selected_simplices": len(selected_simplex_ids),
                "separation_certificates": len(certificates),
                "enumerated_affine_vectors": len(coordinates),
                "minimum_affine_separation": minimum_distance,
                "new_simplex_constraints": len(new_simplex_ids),
                "new_affine_constraints": len(new_coordinates),
                "remaining_simplex_violations": len(simplex_violations),
                "remaining_new_affine_violations": len(violating_ids),
            }
        )
        print(
            f"round={round_number:2d} status={status} "
            f"rho(master/all)={master_rho:.9g}/{covering_rho:.9g} "
            f"ratio-bound={bound_ratio:.12f} "
            f"min-sep={minimum_distance:.9g} "
            f"+simp={len(new_simplex_ids)} "
            f"+affine={len(new_coordinates)}",
            flush=True,
        )
        rho = covering_rho
        if not simplex_violations and not violating_ids:
            converged = True
            break

    target_determinant = float(
        np.linalg.det(evaluator.basis0 @ evaluator.basis0.T)
    )
    output_gram = determinant_rescale(gram, target_determinant)
    output_parameters = parameters_for_gram(
        evaluator.basis0,
        output_gram,
    )
    evaluation = evaluator.evaluate(
        output_parameters,
        with_witnesses=True,
    )
    final_radii = np.asarray(
        [
            circumradius_squared(output_gram, rows)
            for rows in simplices
        ]
    )
    scaled_minimum_certificate = min(
        float(np.sum(output_gram * matrix))
        for matrix in certificates.values()
    )
    certified_diameter = 2.0 * math.sqrt(float(final_radii.max()))
    certified_ratio = (
        math.sqrt(max(0.0, scaled_minimum_certificate))
        / certified_diameter
    )
    payload = {
        "method": (
            "sequential semidefinite covering optimization with "
            "fixed-dual affine separation certificates"
        ),
        "lattice": lattice,
        "source_metric": str(args.metric),
        "source_campaign": metric_payload.get("source_campaign"),
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
        "settings": {
            "rounds": args.rounds,
            "initial_simplices": args.initial_simplices,
            "add_simplices": args.add_simplices,
            "add_affine_vectors": args.add_affine_vectors,
            "solver": args.solver,
            "solver_tolerance": args.solver_tolerance,
            "solver_iterations": args.solver_iterations,
            "positive_floor": args.positive_floor,
            "violation_tolerance": args.violation_tolerance,
        },
        "triangulation_translation_orbits": len(simplices),
        "selected_simplex_constraints": len(selected_simplex_ids),
        "separation_certificates": len(certificates),
        "converged_cutting_planes": converged,
        "history": history,
        "sdp_certificate": {
            "minimum_linear_separation_squared": (
                scaled_minimum_certificate
            ),
            "triangulation_covering_radius_squared": float(
                final_radii.max()
            ),
            "triangulation_diameter": certified_diameter,
            "certified_ratio": certified_ratio,
        },
        "best": evaluation.as_json(),
        "elapsed_seconds": time.perf_counter() - started,
        "valid_numerical_witness": evaluation.min_ratio >= 1.0,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL oracle-ratio={evaluation.min_ratio:.12f} "
        f"SDP-bound={certified_ratio:.12f} converged={converged}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
