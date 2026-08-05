"""HiGHS outer approximation of the D6 covering SDP.

The SDP in :mod:`d6_sdp_hybrid` has only 22 scalar variables in dimension
six (21 entries of a symmetric Gram form and one covering-radius variable),
but it has hundreds of small positive-semidefinite constraints.  Every PSD
constraint ``M(x) >= 0`` implies the linear inequality

    u^T M(x) u >= 0

for every fixed vector ``u``.  This script solves the resulting LP with
HiGHS, separates a negative eigenvector of every violated matrix, and repeats.
The finite LP is an outer approximation: its objective is a lower bound on
the corresponding finite SDP until all eigenvalue violations have vanished.

The same master can impose exact circuit-wall signs.  In particular, when a
metric lies at a high-codimension L-type intersection, we can enumerate sign
chambers for several almost-active walls instead of crossing only one wall at
a time.  Every LP candidate is mapped back to the determinant-preserving
metric parameterization and checked by the complete Voronoi/coloring oracle.

All results remain numerical.  A ratio above one would still require the
repository's rational and independent exact certification pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog

from active_metric_refine import _load_problem
from d6_ltype_wall_cross import voronoi_geometry, wall_slack_from_gram
from d6_sdp_hybrid import (
    canonical_projective_rows,
    circumradius_squared,
    covering_lmi_numeric,
    determinant_rescale,
    exact_coordinate_separations,
    kernel_coordinates_within,
    parameters_for_gram,
    projection_certificate,
    relevant_coordinate_rows,
    triangulation_orbits,
)
from prime_radon import smith_diagonal


def symmetric_pairs(n: int) -> list[tuple[int, int]]:
    """Return the upper-triangular variable order for a symmetric matrix."""
    return [(row, col) for row in range(n) for col in range(row, n)]


def symmetric_coefficients(matrix: np.ndarray) -> np.ndarray:
    """Coefficients of ``<matrix, Q>`` in upper-triangular coordinates."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("linear-form matrix must be square")
    matrix = 0.5 * (matrix + matrix.T)
    return np.asarray(
        [
            matrix[row, col]
            if row == col
            else 2.0 * matrix[row, col]
            for row, col in symmetric_pairs(len(matrix))
        ],
        dtype=np.float64,
    )


def pack_gram(gram: np.ndarray) -> np.ndarray:
    """Pack a symmetric Gram matrix in :func:`symmetric_pairs` order."""
    gram = np.asarray(gram, dtype=np.float64)
    return np.asarray(
        [gram[row, col] for row, col in symmetric_pairs(len(gram))],
        dtype=np.float64,
    )


def unpack_gram(values: Sequence[float], n: int) -> np.ndarray:
    """Unpack upper-triangular variables into a symmetric Gram matrix."""
    values = np.asarray(values, dtype=np.float64)
    pairs = symmetric_pairs(n)
    if values.shape != (len(pairs),):
        raise ValueError(
            f"expected {len(pairs)} Gram variables, got {values.shape}"
        )
    gram = np.zeros((n, n), dtype=np.float64)
    for value, (row, col) in zip(values, pairs):
        gram[row, col] = gram[col, row] = float(value)
    return gram


def covering_cut_coefficients(
    rows: np.ndarray, vector: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Return ``(A, a)`` with ``u^T M(Q,rho)u=<A,Q>+a*rho``."""
    rows = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    if rows.ndim != 2 or vector.shape != (len(rows) + 1,):
        raise ValueError("simplex rows and PSD-cut vector are incompatible")
    head = vector[:-1]
    tail = float(vector[-1])
    projected = rows.T @ head
    matrix = np.outer(projected, projected)
    for coefficient, row in zip(tail * head, rows):
        matrix += float(coefficient) * np.outer(row, row)
    return matrix, tail * tail


def parse_indices(text: str, upper: int) -> list[int]:
    """Parse comma-separated indices and inclusive ranges."""
    if upper < 0:
        raise ValueError("upper index bound must be nonnegative")
    if not text.strip():
        return list(range(upper))
    result: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start = int(start_text)
            stop = int(stop_text)
            if stop < start:
                raise ValueError(f"descending range {token!r}")
            result.update(range(start, stop + 1))
        else:
            result.add(int(token))
    if not result or min(result) < 0 or max(result) >= upper:
        raise ValueError(
            f"wall indices must lie in [0,{max(0, upper - 1)}]"
        )
    return sorted(result)


def _cut_key(coefficients: np.ndarray, right_hand_side: float) -> tuple:
    vector = np.r_[np.asarray(coefficients, dtype=np.float64), right_hand_side]
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        raise ValueError("zero linear cut")
    return tuple(np.round(vector / scale, decimals=11))


def solve_psd_outer(
    n: int,
    simplex_rows_list: Sequence[np.ndarray],
    separation_matrices: Sequence[np.ndarray],
    *,
    warm_gram: np.ndarray,
    warm_rho: float,
    signed_walls: Sequence[tuple[np.ndarray, int]] = (),
    wall_depth: float = 1e-7,
    positive_floor: float = 1e-7,
    max_rounds: int = 40,
    cuts_per_round: int = 192,
    violation_tolerance: float = 2e-8,
    gram_bound_factor: float = 16.0,
) -> dict:
    """Solve a finite PSD master by HiGHS eigenvector cutting planes."""
    if (
        n < 1
        or max_rounds < 1
        or cuts_per_round < 1
        or wall_depth < 0
        or positive_floor <= 0
        or violation_tolerance <= 0
        or gram_bound_factor <= 1
    ):
        raise ValueError("invalid PSD outer-approximation settings")
    warm_gram = np.asarray(warm_gram, dtype=np.float64)
    if warm_gram.shape != (n, n):
        raise ValueError("warm Gram matrix has the wrong shape")
    variable_count = n * (n + 1) // 2 + 1
    rho_index = variable_count - 1
    identity = np.eye(n)
    trace_coefficients = symmetric_coefficients(identity)

    lower_rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    lower_labels: list[str] = []
    known_cuts: set[tuple] = set()

    def add_lower(
        gram_matrix: np.ndarray,
        rho_coefficient: float,
        right_hand_side: float,
        label: str,
    ) -> bool:
        row = np.r_[
            symmetric_coefficients(gram_matrix),
            float(rho_coefficient),
        ]
        key = _cut_key(row, float(right_hand_side))
        if key in known_cuts:
            return False
        known_cuts.add(key)
        lower_rows.append(row)
        lower_bounds.append(float(right_hand_side))
        lower_labels.append(label)
        return True

    for matrix in separation_matrices:
        add_lower(matrix, 0.0, 1.0, "kernel-separation")

    # Coordinate and pair directions give HiGHS a bounded, well-scaled first
    # relaxation before adaptive eigenvector cuts start.
    directions: list[np.ndarray] = []
    for index in range(n):
        direction = np.zeros(n)
        direction[index] = 1.0
        directions.append(direction)
    for row in range(n):
        for col in range(row + 1, n):
            for sign in (-1.0, 1.0):
                direction = np.zeros(n)
                direction[row] = 1.0 / math.sqrt(2.0)
                direction[col] = sign / math.sqrt(2.0)
                directions.append(direction)
    for direction in directions:
        add_lower(
            np.outer(direction, direction),
            0.0,
            positive_floor * float(direction @ direction),
            "gram-psd-initial",
        )

    for simplex_index, rows in enumerate(simplex_rows_list):
        lmi = covering_lmi_numeric(warm_gram, rows, warm_rho)
        _, eigenvectors = np.linalg.eigh(lmi)
        matrix, rho_coefficient = covering_cut_coefficients(
            rows, eigenvectors[:, 0]
        )
        add_lower(
            matrix,
            rho_coefficient,
            0.0,
            f"simplex-{simplex_index}-initial",
        )

    for wall_index, (wall, side) in enumerate(signed_walls):
        if side not in {-1, 1}:
            raise ValueError("wall side must be -1 or +1")
        # side * <W,Q>/2 >= wall_depth * trace(Q)
        matrix = (
            0.5 * float(side) * np.asarray(wall, dtype=np.float64)
            - wall_depth * identity
        )
        add_lower(matrix, 0.0, 0.0, f"wall-{wall_index}-side-{side:+d}")

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[rho_index] = 1.0
    entry_bound = gram_bound_factor * max(
        1.0, float(np.max(np.abs(warm_gram)))
    )
    bounds: list[tuple[float | None, float | None]] = []
    for row, col in symmetric_pairs(n):
        if row == col:
            bounds.append((positive_floor, entry_bound))
        else:
            bounds.append((-entry_bound, entry_bound))
    bounds.append((0.0, None))

    history: list[dict] = []
    solution: np.ndarray | None = None
    status = "not_solved"
    converged = False
    started = time.perf_counter()

    for round_number in range(1, max_rounds + 1):
        a_ub = -np.asarray(lower_rows, dtype=np.float64)
        b_ub = -np.asarray(lower_bounds, dtype=np.float64)
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        status = str(result.message)
        if not result.success:
            return {
                "success": False,
                "converged": False,
                "status": status,
                "highs_status": int(result.status),
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
                "linear_cuts": len(lower_rows),
            }
        solution = np.asarray(result.x, dtype=np.float64)
        gram = unpack_gram(solution[:-1], n)
        rho = float(solution[-1])

        violations: list[
            tuple[float, str, int, np.ndarray, np.ndarray, float]
        ] = []
        gram_shift = gram - positive_floor * identity
        eigenvalues, eigenvectors = np.linalg.eigh(gram_shift)
        gram_scale = max(1.0, float(np.linalg.norm(gram_shift, ord=2)))
        for eigenvalue, vector in zip(eigenvalues, eigenvectors.T):
            relative = float(eigenvalue) / gram_scale
            if relative < -violation_tolerance:
                violations.append(
                    (
                        relative,
                        "gram",
                        -1,
                        vector,
                        np.outer(vector, vector),
                        positive_floor * float(vector @ vector),
                    )
                )

        worst_simplex_value = math.inf
        for simplex_index, rows in enumerate(simplex_rows_list):
            lmi = covering_lmi_numeric(gram, rows, rho)
            eigenvalues, eigenvectors = np.linalg.eigh(lmi)
            matrix_scale = max(1.0, float(np.linalg.norm(lmi, ord=2)))
            worst_simplex_value = min(
                worst_simplex_value, float(eigenvalues[0])
            )
            for eigenvalue, vector in zip(eigenvalues, eigenvectors.T):
                relative = float(eigenvalue) / matrix_scale
                if relative >= -violation_tolerance:
                    continue
                matrix, rho_coefficient = covering_cut_coefficients(
                    rows, vector
                )
                violations.append(
                    (
                        relative,
                        "simplex",
                        simplex_index,
                        vector,
                        matrix,
                        rho_coefficient,
                    )
                )

        violations.sort(key=lambda item: item[0])
        added = 0
        for _, kind, index, _, matrix, last_value in violations[
            :cuts_per_round
        ]:
            if kind == "gram":
                was_added = add_lower(
                    matrix,
                    0.0,
                    last_value,
                    "gram-psd-separated",
                )
            else:
                was_added = add_lower(
                    matrix,
                    last_value,
                    0.0,
                    f"simplex-{index}-separated",
                )
            added += int(was_added)

        packed = pack_gram(gram)
        bound_activity = float(
            np.max(np.abs(packed)) / entry_bound
        )
        history.append(
            {
                "round": round_number,
                "rho": rho,
                "linear_cuts": len(lower_rows),
                "violated_psd_eigenvalues": len(violations),
                "new_eigenvector_cuts": added,
                "minimum_gram_eigenvalue": float(
                    np.linalg.eigvalsh(gram)[0]
                ),
                "minimum_covering_lmi_eigenvalue": (
                    worst_simplex_value
                ),
                "gram_bound_activity": bound_activity,
            }
        )
        if not violations:
            converged = True
            break
        if added == 0:
            status = "stalled: all violated eigenvector cuts duplicated"
            break

    if solution is None:
        raise RuntimeError("HiGHS outer approximation produced no solution")
    gram = unpack_gram(solution[:-1], n)
    rho = float(solution[-1])
    wall_depths = [
        {
            "side": int(side),
            "achieved_relative_depth": (
                float(side)
                * wall_slack_from_gram(gram, wall)
                / max(float(np.trace(gram)), 1e-300)
            ),
        }
        for wall, side in signed_walls
    ]
    return {
        "success": True,
        "converged": converged,
        "status": status,
        "gram": gram,
        "rho": rho,
        "finite_outer_ratio": (
            1.0 / (2.0 * math.sqrt(rho)) if rho > 0 else math.inf
        ),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "linear_cuts": len(lower_rows),
        "gram_entry_bound": entry_bound,
        "wall_depths": wall_depths,
    }


def wall_branches(
    wall_records: Sequence[dict],
    selected_indices: Sequence[int],
    *,
    intersection_walls: int,
    max_combination_width: int,
) -> list[dict]:
    """Build individual crossings and sign chambers near an intersection."""
    selected = set(int(index) for index in selected_indices)
    branches: list[dict] = []
    for wall_index in sorted(selected):
        branches.append(
            {
                "label": f"wall-{wall_index}",
                "crossed_walls": [wall_index],
                "signed_walls": [(wall_index, -1)],
                "kind": "individual",
            }
        )

    control = [
        index
        for index in range(min(intersection_walls, len(wall_records)))
        if index in selected
    ]
    width = min(max_combination_width, len(control))
    for size in range(1, width + 1):
        for crossed_tuple in itertools.combinations(control, size):
            crossed = set(crossed_tuple)
            signs = [
                (index, -1 if index in crossed else 1)
                for index in control
            ]
            label = "chamber-neg-" + "-".join(
                str(index) for index in sorted(crossed)
            )
            branches.append(
                {
                    "label": label,
                    "crossed_walls": sorted(crossed),
                    "signed_walls": signs,
                    "kind": "intersection-chamber",
                }
            )
    unique: dict[tuple, dict] = {}
    for branch in branches:
        key = tuple(branch["signed_walls"])
        unique.setdefault(key, branch)
    return list(unique.values())


def _resolve_metric(screen_path: Path, payload: dict) -> Path:
    metric = Path(payload["source_metric"])
    if not metric.is_absolute() and not metric.exists():
        metric = screen_path.resolve().parent / metric.name
    return metric


def _compact_evaluation(evaluation) -> dict:
    return {
        "min_ratio": evaluation.min_ratio,
        "min_distance": evaluation.min_distance,
        "diameter": evaluation.diameter,
        "soft_min": evaluation.soft_min,
        "parameters": evaluation.parameters.tolist(),
        "facet_count": evaluation.facet_count,
        "vertex_count": evaluation.vertex_count,
        "subvector_count": evaluation.subvector_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument(
        "--wall-indices",
        default="",
        help="comma-separated wall indices/ranges; empty means all",
    )
    parser.add_argument("--intersection-walls", type=int, default=0)
    parser.add_argument("--max-combination-width", type=int, default=3)
    parser.add_argument("--max-branches", type=int, default=0)
    parser.add_argument("--outer-rounds", type=int, default=40)
    parser.add_argument("--cuts-per-round", type=int, default=192)
    parser.add_argument("--violation-tolerance", type=float, default=2e-8)
    parser.add_argument("--positive-floor", type=float, default=1e-7)
    parser.add_argument("--wall-depth", type=float, default=1e-7)
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
        args.intersection_walls < 0
        or args.max_combination_width < 1
        or args.max_branches < 0
        or args.outer_rounds < 1
        or args.cuts_per_round < 1
        or args.violation_tolerance <= 0
        or args.positive_floor <= 0
        or args.wall_depth < 0
        or args.gram_bound_factor <= 1
    ):
        parser.error("invalid HiGHS/PSD outer-approximation budget")

    screen_payload = json.loads(args.screen.read_text())
    wall_records = screen_payload.get("walls", [])
    if not wall_records:
        raise ValueError("wall screen contains no selected wall records")
    selected_indices = parse_indices(args.wall_indices, len(wall_records))
    branches = wall_branches(
        wall_records,
        selected_indices,
        intersection_walls=args.intersection_walls,
        max_combination_width=args.max_combination_width,
    )
    if args.max_branches:
        branches = branches[: args.max_branches]

    metric_path = _resolve_metric(args.screen, screen_payload)
    (
        metric_payload,
        source,
        base_metric,
        source_record,
        kernel,
        evaluator,
    ) = _load_problem(metric_path, args.temperature, args.max_h_norm)
    center = np.asarray(
        metric_payload["best"]["parameters"], dtype=np.float64
    )
    source_evaluation = evaluator.evaluate(center, with_witnesses=True)
    recorded_ratio = float(metric_payload["best"]["min_ratio"])
    if abs(source_evaluation.min_ratio - recorded_ratio) > 5e-7:
        raise RuntimeError("source metric does not reproduce before LP screen")
    source_gram = source_evaluation.basis @ source_evaluation.basis.T
    simplices = triangulation_orbits(source_evaluation.basis)
    source_facets = relevant_coordinate_rows(source_evaluation.basis)
    source_coordinates = kernel_coordinates_within(
        source_evaluation.basis,
        kernel,
        2.0 * source_evaluation.diameter + 1e-8,
    )
    source_coordinates = canonical_projective_rows(source_coordinates)
    source_distances, _ = exact_coordinate_separations(
        source_evaluation.basis, source_coordinates
    )
    certificates: dict[tuple[int, ...], np.ndarray] = {}
    certificate_values: dict[tuple[int, ...], float] = {}
    for index in np.argsort(source_distances):
        coordinate = source_coordinates[int(index)]
        certificate, dual, primal = projection_certificate(
            source_gram,
            coordinate,
            source_facets,
            solver=args.projection_solver,
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
    minimum_certificate = min(certificate_values.values())
    warm_gram = source_gram / minimum_certificate
    warm_rho = max(
        circumradius_squared(warm_gram, rows) for rows in simplices
    )
    target_determinant = float(
        np.linalg.det(evaluator.basis0 @ evaluator.basis0.T)
    )

    payload: dict = {
        "method": (
            "HiGHS LP outer approximation of covering PSD constraints "
            "with negative-eigenvector cuts and L-type sign chambers"
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
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "source": {
            **source_evaluation.as_json(),
            "voronoi_signature": voronoi_geometry(
                source_evaluation.basis
            ).signature,
        },
        "settings": {
            "wall_indices": selected_indices,
            "intersection_walls": args.intersection_walls,
            "max_combination_width": args.max_combination_width,
            "max_branches": args.max_branches,
            "outer_rounds": args.outer_rounds,
            "cuts_per_round": args.cuts_per_round,
            "violation_tolerance": args.violation_tolerance,
            "positive_floor": args.positive_floor,
            "wall_depth": args.wall_depth,
            "gram_bound_factor": args.gram_bound_factor,
            "projection_solver": args.projection_solver,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
        },
        "triangulation_translation_orbits": len(simplices),
        "separation_certificates": len(certificates),
        "branch_count": len(branches),
        "branches": [],
        "best": None,
        "valid_numerical_witness": None,
    }
    started = time.perf_counter()
    best_parameters: np.ndarray | None = None
    best_ratio = -math.inf

    print(
        f"source={source_evaluation.min_ratio:.12f} "
        f"simplices={len(simplices)} certs={len(certificates)} "
        f"branches={len(branches)}",
        flush=True,
    )
    for branch_number, branch in enumerate(branches, start=1):
        signed_walls = [
            (
                np.asarray(
                    wall_records[wall_index]["wall_matrix"],
                    dtype=np.int64,
                ),
                side,
            )
            for wall_index, side in branch["signed_walls"]
        ]
        outer = solve_psd_outer(
            len(warm_gram),
            simplices,
            list(certificates.values()),
            warm_gram=warm_gram,
            warm_rho=warm_rho,
            signed_walls=signed_walls,
            wall_depth=args.wall_depth,
            positive_floor=args.positive_floor,
            max_rounds=args.outer_rounds,
            cuts_per_round=args.cuts_per_round,
            violation_tolerance=args.violation_tolerance,
            gram_bound_factor=args.gram_bound_factor,
        )
        record: dict = {
            **branch,
            "success": outer["success"],
            "converged_outer_psd": outer["converged"],
            "highs_status": outer["status"],
            "outer_history": outer["history"],
            "outer_elapsed_seconds": outer["elapsed_seconds"],
            "linear_cuts": outer["linear_cuts"],
        }
        if outer["success"]:
            output_gram = determinant_rescale(
                outer["gram"], target_determinant
            )
            try:
                parameters = parameters_for_gram(
                    evaluator.basis0, output_gram
                )
                evaluation = evaluator.evaluate(parameters)
                geometry = voronoi_geometry(evaluation.basis)
                sign_checks = []
                for (wall_index, side), (wall, _) in zip(
                    branch["signed_walls"], signed_walls
                ):
                    slack = wall_slack_from_gram(output_gram, wall)
                    sign_checks.append(
                        {
                            "wall_index": wall_index,
                            "requested_side": side,
                            "wall_slack": slack,
                            "side_satisfied": side * slack > 0,
                        }
                    )
                record.update(
                    {
                        "finite_outer_ratio": outer[
                            "finite_outer_ratio"
                        ],
                        "gram_entry_bound": outer["gram_entry_bound"],
                        "wall_depths": outer["wall_depths"],
                        "sign_checks": sign_checks,
                        "voronoi_signature": geometry.signature,
                        "changed_from_source": (
                            geometry.signature
                            != payload["source"]["voronoi_signature"]
                        ),
                        "oracle": _compact_evaluation(evaluation),
                    }
                )
                if evaluation.min_ratio > best_ratio:
                    best_ratio = evaluation.min_ratio
                    best_parameters = parameters.copy()
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                record["conversion_error"] = str(error)
        payload["branches"].append(record)
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        ratio_text = (
            f"{record['oracle']['min_ratio']:.12f}"
            if "oracle" in record
            else "n/a"
        )
        print(
            f"branch {branch_number:3d}/{len(branches)} "
            f"{branch['label']}: status={outer['status']} "
            f"converged={outer['converged']} oracle={ratio_text}",
            flush=True,
        )

    if best_parameters is not None:
        best = evaluator.evaluate(best_parameters, with_witnesses=True)
        best_geometry = voronoi_geometry(best.basis)
        payload["best"] = {
            **best.as_json(),
            "voronoi_signature": best_geometry.signature,
        }
        payload["valid_numerical_witness"] = best.min_ratio >= 1.0
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL best={best_ratio:.12f} saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
