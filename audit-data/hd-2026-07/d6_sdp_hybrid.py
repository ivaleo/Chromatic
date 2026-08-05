"""Sequential SDP/cutting-plane optimization for a fixed lattice-color kernel.

The metric search used elsewhere in this audit is deliberately black-box: it
recomputes the exact Voronoi cell and the exact kernel separation after every
trial deformation.  This module supplies a complementary convex step.

Fix a periodic unimodular triangulation ``T`` of the parent coordinate lattice.
For a simplex ``{0, v_1, ..., v_n}``, put ``V = (v_i)`` and let ``Q`` be the
parent Gram form.  Its squared circumradius is at most ``rho`` exactly when

    [[V Q V^T, diag(V Q V^T)/2],
     [diag(V Q V^T)^T/2, rho]] >= 0.

This is a linear matrix inequality in ``Q`` and ``rho``.  The triangulation
need not remain Delaunay: every point of a simplex is within its circumradius
of at least one vertex, so the maximum simplex radius is still a valid upper
bound for the lattice covering radius.

Separation of a kernel vector ``k`` from twice the Voronoi cell also has a
useful dual.  For any nonnegative multiplier vector ``lambda`` on lattice
halfspaces ``z`` it gives the lower bound

    dist_Q(k, 2 Vor(Q))^2 >= <Q, C(k, z, lambda)>.

Once the multipliers are frozen, this is linear in ``Q``.  We obtain tight
multipliers at the current metric by a small convex projection QP, solve the
resulting SDP, and then use the complete geometric oracle to add newly
violating kernel vectors and triangulation simplices.  Thus the method is a
hybrid of:

* SDP for the continuous Gram form;
* cutting planes for the finite active model;
* exact finite enumeration for every accepted numerical incumbent.

The output remains numerical.  A ratio crossing one must still be rationalized
and independently certified by the repository's exact verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import cvxpy as cp
import numpy as np
from scipy.linalg import expm, logm
from sympy import Matrix

import combigeo
from active_metric_refine import _load_problem
from d6_ltype_wall_cross import voronoi_geometry
from metric_deform import matrix_parameters
from prime_radon import smith_diagonal


def canonical_projective_rows(rows: Iterable[Sequence[int]]) -> np.ndarray:
    """Deduplicate nonzero integer rows modulo multiplication by ``-1``."""
    unique: dict[tuple[int, ...], tuple[int, ...]] = {}
    width = 0
    for raw in rows:
        row = tuple(int(value) for value in raw)
        width = len(row)
        if not any(row):
            continue
        negative = tuple(-value for value in row)
        key = min(row, negative)
        unique.setdefault(key, key)
    if not unique:
        return np.empty((0, width), dtype=np.int64)
    return np.asarray(sorted(unique), dtype=np.int64)


def canonical_simplex(
    nonzero_rows: Sequence[Sequence[int]] | np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    """Canonicalize a lattice simplex under translation of its anchor."""
    rows = np.asarray(nonzero_rows, dtype=np.int64)
    n = rows.shape[1]
    points = np.vstack((np.zeros((1, n), dtype=np.int64), rows))
    candidates: list[tuple[tuple[int, ...], ...]] = []
    for anchor in points:
        shifted = points - anchor
        candidates.append(
            tuple(
                sorted(
                    tuple(int(value) for value in row)
                    for row in shifted
                )
            )
        )
    return min(candidates)


def simplex_rows(
    canonical: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return the nonzero rows of a canonical translated simplex."""
    rows = np.asarray(canonical, dtype=np.int64)
    return rows[np.any(rows, axis=1)]


def triangulation_orbits(basis: np.ndarray) -> list[np.ndarray]:
    """Recover one coordinate representative of every translation orbit."""
    geometry = voronoi_geometry(basis)
    grouped: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for active in geometry.active_facets:
        rows = geometry.facet_coordinates[active]
        key = canonical_simplex(rows)
        grouped.setdefault(key, simplex_rows(key))
    result = list(grouped.values())
    result.sort(
        key=lambda rows: tuple(int(value) for value in rows.flat)
    )
    return result


def circumradius_squared(gram: np.ndarray, rows: np.ndarray) -> float:
    """Squared circumradius of ``{0, rows}`` under a coordinate Gram form."""
    gram = np.asarray(gram, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    simplex_gram = rows @ gram @ rows.T
    diagonal = np.diag(simplex_gram)
    return 0.25 * float(
        diagonal @ np.linalg.solve(simplex_gram, diagonal)
    )


def covering_lmi_numeric(
    gram: np.ndarray, rows: np.ndarray, rho: float
) -> np.ndarray:
    """Numerical Schur block used by the covering-radius SDP constraint."""
    rows = np.asarray(rows, dtype=np.float64)
    simplex_gram = rows @ np.asarray(gram, dtype=np.float64) @ rows.T
    diagonal = np.diag(simplex_gram)
    return np.block(
        [
            [simplex_gram, 0.5 * diagonal[:, None]],
            [0.5 * diagonal[None, :], np.asarray([[rho]])],
        ]
    )


def covering_lmi_expression(
    gram: cp.Expression, rows: np.ndarray, rho: cp.Expression
) -> cp.Expression:
    rows = np.asarray(rows, dtype=np.float64)
    simplex_gram = rows @ gram @ rows.T
    diagonal = cp.reshape(
        cp.diag(simplex_gram), (len(rows), 1), order="F"
    )
    return cp.bmat(
        [
            [simplex_gram, 0.5 * diagonal],
            [0.5 * diagonal.T, cp.reshape(rho, (1, 1), order="F")],
        ]
    )


def separation_dual_matrix(
    coordinate: Sequence[int] | np.ndarray,
    halfspace_rows: np.ndarray,
    multipliers: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return ``C`` in the valid lower bound ``dist^2 >= <Q,C>``."""
    coordinate = np.asarray(coordinate, dtype=np.float64)
    rows = np.asarray(halfspace_rows, dtype=np.float64)
    multipliers = np.asarray(multipliers, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] != len(multipliers):
        raise ValueError("halfspace rows and multipliers have incompatible shapes")
    if np.min(multipliers, initial=0.0) < -1e-9:
        raise ValueError("dual multipliers must be nonnegative")
    multipliers = np.maximum(multipliers, 0.0)
    weighted_rows = multipliers @ rows
    raw = np.zeros((len(coordinate), len(coordinate)), dtype=np.float64)
    for multiplier, row in zip(multipliers, rows):
        raw += multiplier * np.outer(row, coordinate - row)
    symmetric = 0.5 * (raw + raw.T)
    return symmetric - 0.25 * np.outer(weighted_rows, weighted_rows)


def projection_certificate(
    gram: np.ndarray,
    coordinate: Sequence[int] | np.ndarray,
    halfspace_rows: np.ndarray,
    *,
    solver: str = "CLARABEL",
) -> tuple[np.ndarray, float, float]:
    """Solve the fixed-metric projection QP and return its linear dual bound."""
    gram = np.asarray(gram, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    rows = np.asarray(halfspace_rows, dtype=np.float64)
    n = len(coordinate)
    point = cp.Variable(n)
    normals = rows @ gram
    offsets = np.einsum("ij,ij->i", normals, rows)
    halfspaces = normals @ point <= offsets
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(coordinate - point, gram)),
        [halfspaces],
    )
    problem.solve(solver=solver, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"projection QP failed with status {problem.status!r}"
        )
    multipliers = np.asarray(
        halfspaces.dual_value, dtype=np.float64
    ).reshape(-1)
    multipliers[np.abs(multipliers) < 1e-9] = 0.0
    if np.min(multipliers, initial=0.0) < -2e-7:
        raise RuntimeError("projection QP returned a negative dual multiplier")
    multipliers = np.maximum(multipliers, 0.0)
    certificate = separation_dual_matrix(
        coordinate, rows, multipliers
    )
    dual_value = float(np.sum(gram * certificate))
    primal_value = float(problem.value)
    tolerance = max(2e-7, 2e-6 * abs(primal_value))
    if dual_value > primal_value + tolerance:
        raise RuntimeError(
            "projection dual exceeds the primal value: "
            f"{dual_value:.12g} > {primal_value:.12g}"
        )
    return certificate, dual_value, primal_value


def relevant_coordinate_rows(basis: np.ndarray) -> np.ndarray:
    """Return all current Voronoi facet normals in parent coordinates."""
    facets = combigeo.relevant_facets(
        np.asarray(basis, dtype=np.float64).tolist()
    )
    physical = np.asarray([facet[0] for facet in facets], dtype=np.float64)
    raw = physical @ np.linalg.inv(basis)
    rows = np.rint(raw).astype(np.int64)
    residual = float(np.max(np.abs(raw - rows)))
    if residual > 2e-7:
        raise RuntimeError(
            f"could not recover facet coordinates: residual={residual:g}"
        )
    return np.unique(rows, axis=0)


def kernel_coordinates_within(
    basis: np.ndarray,
    kernel_columns: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Enumerate kernel coordinates modulo sign within a physical radius."""
    coordinate_rows = np.asarray(
        Matrix(
            np.asarray(kernel_columns, dtype=np.int64).T.tolist()
        ).lll().tolist(),
        dtype=np.int64,
    )
    sub_basis = coordinate_rows @ np.asarray(basis, dtype=np.float64)
    physical = combigeo._vectors_near(
        sub_basis.tolist(),
        [0.0] * len(basis),
        float(radius),
    )
    if not physical:
        return np.empty((0, len(basis)), dtype=np.int64)
    raw = np.asarray(physical, dtype=np.float64) @ np.linalg.inv(basis)
    coordinates = np.rint(raw).astype(np.int64)
    residual = float(np.max(np.abs(raw - coordinates)))
    if residual > 2e-6:
        raise RuntimeError(
            f"could not recover kernel coordinates: residual={residual:g}"
        )
    return canonical_projective_rows(coordinates)


def exact_coordinate_separations(
    basis: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, list[tuple[list[float], float]]]:
    """Evaluate exact Voronoi-cell separation for supplied coordinates."""
    facets = combigeo.relevant_facets(
        np.asarray(basis, dtype=np.float64).tolist()
    )
    distances = np.asarray(
        [
            2.0
            * combigeo.dist_to_halfspaces(
                (0.5 * (coordinate @ basis)).tolist(), facets
            )
            for coordinate in coordinates
        ],
        dtype=np.float64,
    )
    return distances, facets


def determinant_rescale(
    gram: np.ndarray, target_determinant: float
) -> np.ndarray:
    """Rescale an SPD Gram form without changing any distance ratio."""
    gram = 0.5 * (
        np.asarray(gram, dtype=np.float64)
        + np.asarray(gram, dtype=np.float64).T
    )
    determinant = float(np.linalg.det(gram))
    if determinant <= 0 or target_determinant <= 0:
        raise ValueError("Gram determinants must be positive")
    factor = (target_determinant / determinant) ** (1.0 / len(gram))
    return gram * factor


def parameters_for_gram(
    basis0: np.ndarray, gram: np.ndarray
) -> np.ndarray:
    """Map a determinant-matched Gram form to the repository parameterization."""
    inverse = np.linalg.inv(np.asarray(basis0, dtype=np.float64))
    relative = inverse @ np.asarray(gram, dtype=np.float64) @ inverse.T
    relative = 0.5 * (relative + relative.T)
    deformation = 0.5 * np.real_if_close(logm(relative)).astype(np.float64)
    deformation = 0.5 * (deformation + deformation.T)
    deformation -= np.trace(deformation) / len(deformation) * np.eye(
        len(deformation)
    )
    reconstructed = basis0 @ expm(deformation)
    error = float(
        np.linalg.norm(
            reconstructed @ reconstructed.T - gram, ord="fro"
        )
        / max(1.0, np.linalg.norm(gram, ord="fro"))
    )
    if error > 2e-6:
        raise RuntimeError(
            f"could not parameterize SDP Gram form: relative error={error:g}"
        )
    return matrix_parameters(deformation)


def solve_sdp(
    n: int,
    simplex_rows_list: Sequence[np.ndarray],
    separation_matrices: Sequence[np.ndarray],
    *,
    solver: str,
    positive_floor: float,
    solver_tolerance: float,
    max_iterations: int,
    warm_gram: np.ndarray,
    warm_rho: float,
) -> tuple[np.ndarray, float, str, float]:
    """Solve one finite SDP master problem."""
    gram = cp.Variable((n, n), symmetric=True)
    rho = cp.Variable(nonneg=True)
    constraints: list[cp.Constraint] = [
        gram >> positive_floor * np.eye(n)
    ]
    constraints.extend(
        covering_lmi_expression(gram, rows, rho) >> 0
        for rows in simplex_rows_list
    )
    constraints.extend(
        cp.sum(cp.multiply(matrix, gram)) >= 1.0
        for matrix in separation_matrices
    )
    problem = cp.Problem(cp.Minimize(rho), constraints)
    gram.value = np.asarray(warm_gram, dtype=np.float64)
    rho.value = float(warm_rho)
    options: dict[str, float | int | bool] = {
        "solver": solver,
        "verbose": False,
        "warm_start": True,
    }
    if solver == "SCS":
        options.update(
            {
                "eps": solver_tolerance,
                "max_iters": max_iterations,
                "acceleration_lookback": 10,
            }
        )
    elif solver == "CLARABEL":
        options.update(
            {
                "tol_gap_abs": solver_tolerance,
                "tol_gap_rel": solver_tolerance,
                "tol_feas": solver_tolerance,
                "max_iter": max_iterations,
            }
        )
    started = time.perf_counter()
    problem.solve(**options)
    elapsed = time.perf_counter() - started
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SDP failed with status {problem.status!r}")
    solution = np.asarray(gram.value, dtype=np.float64)
    solution = 0.5 * (solution + solution.T)
    return solution, float(rho.value), str(problem.status), elapsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--initial-simplices", type=int, default=48)
    parser.add_argument("--add-simplices", type=int, default=48)
    parser.add_argument("--add-kernel-vectors", type=int, default=16)
    parser.add_argument(
        "--solver", choices=("CLARABEL", "SCS"), default="CLARABEL"
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
        or args.add_kernel_vectors < 1
        or args.solver_tolerance <= 0
        or args.solver_iterations < 1
        or args.positive_floor <= 0
        or args.violation_tolerance <= 0
    ):
        parser.error("all SDP and cutting-plane budgets must be positive")

    (
        metric_payload,
        source,
        base_metric,
        source_record,
        kernel,
        evaluator,
    ) = _load_problem(args.metric, args.temperature, args.max_h_norm)
    center = np.asarray(
        metric_payload["best"]["parameters"], dtype=np.float64
    )
    source_evaluation = evaluator.evaluate(center, with_witnesses=True)
    recorded_ratio = float(metric_payload["best"]["min_ratio"])
    if abs(source_evaluation.min_ratio - recorded_ratio) > 5e-7:
        raise RuntimeError("source metric does not reproduce before SDP")
    source_gram = source_evaluation.basis @ source_evaluation.basis.T
    simplices = triangulation_orbits(source_evaluation.basis)
    source_radii = np.asarray(
        [
            circumradius_squared(source_gram, rows)
            for rows in simplices
        ]
    )
    source_facets = relevant_coordinate_rows(source_evaluation.basis)

    initial_radius = 2.0 * source_evaluation.diameter + 1e-8
    source_coordinates = kernel_coordinates_within(
        source_evaluation.basis, kernel, initial_radius
    )
    source_distances, _ = exact_coordinate_separations(
        source_evaluation.basis, source_coordinates
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
            2e-6, 3e-6 * primal
        ):
            raise RuntimeError(
                "projection QP and complete Voronoi distance disagree"
            )
        key = tuple(int(value) for value in coordinate)
        certificates[key] = certificate
        certificate_values[key] = dual

    source_minimum_certificate = min(certificate_values.values())
    gram = source_gram / source_minimum_certificate
    all_radii = source_radii / source_minimum_certificate
    rho = float(all_radii.max())
    selected_simplex_ids = set(
        int(index)
        for index in np.argsort(all_radii)[-args.initial_simplices :]
    )
    started = time.perf_counter()
    history: list[dict] = []
    converged = False

    print(
        f"source ratio={source_evaluation.min_ratio:.12f} "
        f"simplices={len(simplices)} "
        f"kernel-vectors={len(certificates)} "
        f"rho={rho:.12g}",
        flush=True,
    )

    for round_number in range(1, args.rounds + 1):
        separation_matrices = list(certificates.values())
        gram, master_rho, status, solve_seconds = solve_sdp(
            len(gram),
            [simplices[index] for index in sorted(selected_simplex_ids)],
            separation_matrices,
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

        safe_norm_radius = 1.0 + 2.0 * math.sqrt(covering_rho) + 2e-6
        coordinates = kernel_coordinates_within(
            basis, kernel, safe_norm_radius
        )
        distances, _ = exact_coordinate_separations(basis, coordinates)
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
            for index in violating_ids[: args.add_kernel_vectors]
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
        certified_ratio = 1.0 / (2.0 * math.sqrt(covering_rho))
        history.append(
            {
                "round": round_number,
                "status": status,
                "solve_seconds": solve_seconds,
                "master_rho": master_rho,
                "covering_rho": covering_rho,
                "certified_ratio_if_separated": certified_ratio,
                "minimum_gram_eigenvalue": float(eigenvalues[0]),
                "selected_simplices": len(selected_simplex_ids),
                "separation_certificates": len(certificates),
                "enumerated_kernel_vectors": len(coordinates),
                "minimum_kernel_separation": minimum_distance,
                "new_simplex_constraints": len(new_simplex_ids),
                "new_kernel_constraints": len(new_coordinates),
                "remaining_simplex_violations": len(simplex_violations),
                "remaining_new_kernel_violations": len(violating_ids),
            }
        )
        print(
            f"round={round_number:2d} status={status} "
            f"rho(master/all)={master_rho:.9g}/{covering_rho:.9g} "
            f"ratio-bound={certified_ratio:.12f} "
            f"min-sep={minimum_distance:.9g} "
            f"+simp={len(new_simplex_ids)} "
            f"+kernel={len(new_coordinates)}",
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
        evaluator.basis0, output_gram
    )
    evaluation = evaluator.evaluate(
        output_parameters, with_witnesses=True
    )
    final_radii = np.asarray(
        [
            circumradius_squared(output_gram, rows)
            for rows in simplices
        ]
    )
    scale = float(
        np.trace(output_gram) / np.trace(gram)
    )
    scaled_minimum_certificate = min(
        float(np.sum(output_gram * matrix))
        for matrix in certificates.values()
    )
    certified_diameter = 2.0 * math.sqrt(float(final_radii.max()))
    certified_ratio = math.sqrt(
        max(0.0, scaled_minimum_certificate)
    ) / certified_diameter
    payload = {
        "method": (
            "sequential semidefinite covering optimization with fixed-dual "
            "kernel separation certificates and complete cutting planes"
        ),
        "source_metric": str(args.metric),
        "source_campaign": str(source),
        "base_metric": str(base_metric) if base_metric is not None else None,
        "source_record": {
            "moduli": source_record["moduli"],
            "rows": source_record["rows"],
            "image_index": source_record["image_index"],
            "beta": source_record.get("beta"),
        },
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "settings": {
            "rounds": args.rounds,
            "initial_simplices": args.initial_simplices,
            "add_simplices": args.add_simplices,
            "add_kernel_vectors": args.add_kernel_vectors,
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
            "gram_rescale": scale,
        },
        "best": evaluation.as_json(),
        "elapsed_seconds": time.perf_counter() - started,
        "valid_numerical_witness": evaluation.min_ratio >= 1.0,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL oracle-ratio={evaluation.min_ratio:.12f} "
        f"SDP-bound={certified_ratio:.12f} "
        f"converged={converged} saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
