"""Screen adjacent L-type cones around a six-dimensional coloring metric.

For a simple Voronoi vertex with active lattice-coordinate rows
``v_1, ..., v_n``, the points ``0, v_1, ..., v_n`` form a Delone simplex.
Adding another lattice point ``w`` gives a circuit.  Since the active matrix
is unimodular for the current D6/336 form, write

    w = sum_i lambda_i v_i.

The power slack of ``w`` at the simplex circumcenter is the *linear* Gram-form
functional

    slack_Q(V, w)
      = 1/2 <Q, w w^T - sum_i lambda_i v_i v_i^T>.

An L-type wall is reached when this value vanishes.  This script extracts the
nearest distinct circuit functionals from every Voronoi vertex, follows the
steepest parameter-space direction to each wall, verifies that the full
Voronoi incidence signature changes only after the computed root, and checks
several strict overshoots with the complete coloring-distance oracle.

The result is a reproducible adjacent-cone screen, not a proof of local or
global optimality.  A crossing whose ratio reaches one must still be
rationalized and independently certified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.spatial import HalfspaceIntersection

import combigeo
from chromatic_research.core.active_metric_refine import _load_problem
from chromatic_research.core.metric_deform import MetricEvaluation, trace_free_matrix
from chromatic_research.core.prime_radon import smith_diagonal


@dataclass
class VoronoiGeometry:
    facet_coordinates: np.ndarray
    vertices: np.ndarray
    active_facets: list[list[int]]
    radius: float
    signature: str


def basis_from_parameters(
    basis0: np.ndarray, parameters: Sequence[float]
) -> np.ndarray:
    parameters_array = np.asarray(parameters, dtype=np.float64)
    return np.asarray(basis0, dtype=np.float64) @ expm(
        trace_free_matrix(parameters_array, len(basis0))
    )


def _vertex_signature(
    facet_coordinates: np.ndarray,
    active_facets: Sequence[Sequence[int]],
) -> str:
    """Order-independent hash of all Delone simplices at the origin."""
    simplices = [
        tuple(
            sorted(
                tuple(int(value) for value in facet_coordinates[index])
                for index in active
            )
        )
        for active in active_facets
    ]
    encoded = json.dumps(
        sorted(simplices), separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def voronoi_geometry(basis: np.ndarray) -> VoronoiGeometry:
    basis = np.asarray(basis, dtype=np.float64)
    facets = combigeo.relevant_facets(basis.tolist())
    normals = np.asarray([facet[0] for facet in facets], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    offsets = np.asarray([facet[1] for facet in facets], dtype=np.float64)
    halfspaces = np.column_stack(
        (normals / lengths[:, None], -offsets)
    )
    hull = HalfspaceIntersection(
        halfspaces, np.zeros(len(basis)), qhull_options="Qx"
    )
    inverse = np.linalg.inv(basis)
    raw_coordinates = normals @ inverse
    facet_coordinates = np.rint(raw_coordinates).astype(np.int64)
    residual = float(
        np.max(np.abs(raw_coordinates - facet_coordinates))
    )
    if residual > 2e-7:
        raise RuntimeError(
            f"could not recover integer facet coordinates: {residual:g}"
        )
    active_facets = [
        [int(index) for index in active]
        for active in hull.dual_facets
    ]
    if any(len(active) != len(basis) for active in active_facets):
        raise RuntimeError("source Voronoi cell is not simple")
    vertices = np.asarray(hull.intersections, dtype=np.float64)
    return VoronoiGeometry(
        facet_coordinates=facet_coordinates,
        vertices=vertices,
        active_facets=active_facets,
        radius=float(np.linalg.norm(vertices, axis=1).max()),
        signature=_vertex_signature(facet_coordinates, active_facets),
    )


def circuit_matrix(
    active_rows: Sequence[Sequence[int]] | np.ndarray,
    candidate: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the primitive integer wall matrix and circuit coefficients."""
    active = np.asarray(active_rows, dtype=np.int64)
    candidate_array = np.asarray(candidate, dtype=np.int64)
    n = len(candidate_array)
    if active.shape != (n, n):
        raise ValueError("active rows must be a square full-rank matrix")
    determinant = int(round(np.linalg.det(active)))
    if abs(determinant) != 1:
        raise ValueError(
            "current wall extractor requires unimodular Delone simplices"
        )
    raw_coefficients = np.linalg.solve(
        active.T.astype(np.float64), candidate_array.astype(np.float64)
    )
    coefficients = np.rint(raw_coefficients).astype(np.int64)
    if (
        np.max(np.abs(raw_coefficients - coefficients)) > 1e-9
        or not np.array_equal(coefficients @ active, candidate_array)
    ):
        raise RuntimeError("failed to recover the exact circuit relation")
    wall = np.outer(candidate_array, candidate_array)
    for coefficient, row in zip(coefficients, active):
        wall -= int(coefficient) * np.outer(row, row)
    if not np.array_equal(wall, wall.T) or not np.any(wall):
        raise RuntimeError("degenerate circuit wall")
    divisor = reduce(
        math.gcd,
        (abs(int(value)) for value in wall.flat if value),
    )
    wall //= divisor
    return wall, coefficients


def wall_slack_from_gram(gram: np.ndarray, wall: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            np.asarray(gram, dtype=np.float64)
            * np.asarray(wall, dtype=np.float64)
        )
    )


def oriented_wall_key(
    wall: np.ndarray, gram: np.ndarray
) -> tuple[tuple[int, ...], np.ndarray, float]:
    """Orient a primitive wall so that the source metric has positive slack."""
    oriented = np.asarray(wall, dtype=np.int64).copy()
    slack = wall_slack_from_gram(gram, oriented)
    if abs(slack) <= 1e-12:
        raise RuntimeError("source metric lies numerically on a wall")
    if slack < 0:
        oriented *= -1
        slack *= -1
    upper = oriented[np.triu_indices(len(oriented))]
    return tuple(int(value) for value in upper), oriented, float(slack)


def candidate_coordinates(
    basis: np.ndarray, radius: float
) -> np.ndarray:
    raw = np.asarray(
        combigeo._vectors_near(
            np.asarray(basis, dtype=np.float64).tolist(),
            [0.0] * len(basis),
            float(radius),
        ),
        dtype=np.float64,
    )
    inverse = np.linalg.inv(basis)
    coordinates = np.unique(
        np.rint(raw @ inverse).astype(np.int64), axis=0
    )
    return coordinates[np.any(coordinates, axis=1)]


def discover_wall_orbits(
    basis: np.ndarray,
    *,
    candidate_margin: float,
    neighbors_per_simplex: int,
) -> tuple[VoronoiGeometry, list[dict], int]:
    geometry = voronoi_geometry(basis)
    candidates = candidate_coordinates(
        basis, 2.0 * geometry.radius * (1.0 + candidate_margin)
    )
    physical = candidates @ basis
    offsets = 0.5 * np.einsum(
        "ij,ij->i", physical, physical
    )
    gram = basis @ basis.T
    grouped: dict[tuple[int, ...], dict] = {}
    for vertex_index, (vertex, active_ids) in enumerate(
        zip(geometry.vertices, geometry.active_facets)
    ):
        active = geometry.facet_coordinates[active_ids]
        active_keys = {tuple(int(value) for value in row) for row in active}
        slacks = offsets - physical @ vertex
        selected = 0
        for candidate_index in np.argsort(slacks):
            candidate = candidates[int(candidate_index)]
            if tuple(int(value) for value in candidate) in active_keys:
                continue
            numerical_slack = float(slacks[int(candidate_index)])
            if numerical_slack <= 1e-10:
                continue
            wall_raw, coefficients = circuit_matrix(active, candidate)
            key, wall, exact_slack = oriented_wall_key(wall_raw, gram)
            formula_error = abs(exact_slack - numerical_slack)
            if formula_error > 2e-8:
                raise RuntimeError(
                    "circuit and circumcenter slacks disagree: "
                    f"{formula_error:g}"
                )
            record = grouped.setdefault(
                key,
                {
                    "wall_matrix": wall,
                    "source_slack": exact_slack,
                    "representative_active_rows": active.copy(),
                    "candidate_row": candidate.copy(),
                    "circuit_coefficients": coefficients.copy(),
                    "vertex_indices": set(),
                },
            )
            record["vertex_indices"].add(int(vertex_index))
            if exact_slack < record["source_slack"]:
                record["source_slack"] = exact_slack
                record["representative_active_rows"] = active.copy()
                record["candidate_row"] = candidate.copy()
                record["circuit_coefficients"] = coefficients.copy()
            selected += 1
            if selected >= neighbors_per_simplex:
                break
    records: list[dict] = []
    for record in grouped.values():
        wall = record["wall_matrix"]
        records.append(
            {
                "wall_matrix": wall,
                "source_slack": float(record["source_slack"]),
                "wall_frobenius_norm": float(np.linalg.norm(wall)),
                "normalized_gram_gap": float(
                    record["source_slack"] / np.linalg.norm(wall)
                ),
                "representative_active_rows": record[
                    "representative_active_rows"
                ],
                "candidate_row": record["candidate_row"],
                "circuit_coefficients": record[
                    "circuit_coefficients"
                ],
                "source_vertex_multiplicity": len(
                    record["vertex_indices"]
                ),
            }
        )
    records.sort(key=lambda item: item["normalized_gram_gap"])
    return geometry, records, int(len(candidates))


def wall_slack(
    basis0: np.ndarray,
    parameters: Sequence[float],
    wall: np.ndarray,
) -> float:
    basis = basis_from_parameters(basis0, parameters)
    return wall_slack_from_gram(basis @ basis.T, wall)


def wall_gradient(
    basis0: np.ndarray,
    parameters: Sequence[float],
    wall: np.ndarray,
    finite_difference: float,
) -> np.ndarray:
    center = np.asarray(parameters, dtype=np.float64)
    gradient = np.empty_like(center)
    for index in range(len(center)):
        offset = np.zeros_like(center)
        offset[index] = finite_difference
        gradient[index] = (
            wall_slack(basis0, center + offset, wall)
            - wall_slack(basis0, center - offset, wall)
        ) / (2.0 * finite_difference)
    return gradient


def wall_root(
    basis0: np.ndarray,
    parameters: Sequence[float],
    wall: np.ndarray,
    *,
    finite_difference: float,
    maximum_step: float,
) -> tuple[float, np.ndarray, float]:
    center = np.asarray(parameters, dtype=np.float64)
    source_slack = wall_slack(basis0, center, wall)
    gradient = wall_gradient(
        basis0, center, wall, finite_difference
    )
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= 1e-12:
        raise RuntimeError("wall has zero parameter gradient")
    direction = -gradient / gradient_norm
    estimate = source_slack / gradient_norm
    upper = max(2.0 * estimate, 1e-5)
    while (
        upper < maximum_step
        and wall_slack(basis0, center + upper * direction, wall) > 0
    ):
        upper *= 2.0
    upper = min(upper, maximum_step)
    if wall_slack(basis0, center + upper * direction, wall) > 0:
        raise RuntimeError("wall root exceeds the maximum step")
    root = brentq(
        lambda step: wall_slack(
            basis0, center + step * direction, wall
        ),
        0.0,
        upper,
        xtol=1e-13,
        rtol=1e-13,
    )
    return float(root), direction, gradient_norm


def compact_evaluation(evaluation: MetricEvaluation) -> dict:
    return {
        "min_ratio": evaluation.min_ratio,
        "soft_min": evaluation.soft_min,
        "min_distance": evaluation.min_distance,
        "diameter": evaluation.diameter,
        "facet_count": evaluation.facet_count,
        "vertex_count": evaluation.vertex_count,
        "subvector_count": evaluation.subvector_count,
        "deformation_frobenius_norm": evaluation.h_norm,
        "parameters": evaluation.parameters.tolist(),
    }


def parse_overshoots(text: str) -> list[float]:
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise argparse.ArgumentTypeError(
            "overshoots must be a non-empty JSON list"
        )
    result = [float(value) for value in values]
    if any(
        not math.isfinite(value) or value <= 0 for value in result
    ):
        raise argparse.ArgumentTypeError(
            "overshoots must be finite and positive"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--walls", type=int, default=24)
    parser.add_argument("--neighbors-per-simplex", type=int, default=2)
    parser.add_argument("--candidate-margin", type=float, default=0.05)
    parser.add_argument("--finite-difference", type=float, default=2e-6)
    parser.add_argument("--maximum-step", type=float, default=0.08)
    parser.add_argument(
        "--overshoots",
        type=parse_overshoots,
        default=[0.01, 0.05, 0.2, 0.5, 1.0, 2.0],
    )
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.walls < 1 or args.neighbors_per_simplex < 1:
        parser.error("wall and neighbor counts must be positive")
    if (
        not 0 <= args.candidate_margin <= 0.5
        or args.finite_difference <= 0
        or args.maximum_step <= 0
    ):
        parser.error("invalid wall-screen scale")

    (
        metric_payload,
        source,
        base_metric,
        record,
        kernel,
        evaluator,
    ) = _load_problem(
        args.metric, args.temperature, args.max_h_norm
    )
    center = np.asarray(
        metric_payload["best"]["parameters"], dtype=np.float64
    )
    source_evaluation = evaluator.evaluate(
        center, with_witnesses=True
    )
    recorded_ratio = float(metric_payload["best"]["min_ratio"])
    if abs(source_evaluation.min_ratio - recorded_ratio) > 5e-7:
        raise RuntimeError(
            "metric parameterization mismatch at the source"
        )
    started = time.perf_counter()
    source_geometry, discovered, candidate_count = discover_wall_orbits(
        source_evaluation.basis,
        candidate_margin=args.candidate_margin,
        neighbors_per_simplex=args.neighbors_per_simplex,
    )
    print(
        f"source ratio={source_evaluation.min_ratio:.12f} "
        f"signature={source_geometry.signature[:12]} "
        f"walls={len(discovered)} candidates={candidate_count}",
        flush=True,
    )

    rooted: list[dict] = []
    for record_index, wall_record in enumerate(discovered):
        try:
            root, direction, gradient_norm = wall_root(
                evaluator.basis0,
                center,
                wall_record["wall_matrix"],
                finite_difference=args.finite_difference,
                maximum_step=args.maximum_step,
            )
        except RuntimeError:
            continue
        rooted.append(
            {
                **wall_record,
                "discovery_rank": record_index,
                "parameter_root": root,
                "parameter_gradient_norm": gradient_norm,
                "parameter_direction": direction,
            }
        )
    rooted.sort(key=lambda item: item["parameter_root"])
    selected = rooted[: args.walls]

    payload: dict = {
        "method": (
            "exact circuit Gram walls with parameter-space root finding "
            "and full adjacent Voronoi/coloring checks"
        ),
        "source_metric": str(args.metric),
        "source_campaign": str(source),
        "base_metric": (
            str(base_metric) if base_metric is not None else None
        ),
        "source_record": {
            "moduli": record["moduli"],
            "rows": record["rows"],
            "image_index": record["image_index"],
            "beta": record.get("beta"),
        },
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(
            int(round(np.linalg.det(kernel)))
        ),
        "kernel_smith": smith_diagonal(kernel),
        "source": {
            **source_evaluation.as_json(),
            "voronoi_signature": source_geometry.signature,
        },
        "settings": {
            "walls": args.walls,
            "neighbors_per_simplex": args.neighbors_per_simplex,
            "candidate_margin": args.candidate_margin,
            "finite_difference": args.finite_difference,
            "maximum_step": args.maximum_step,
            "overshoots": args.overshoots,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
        },
        "discovered_wall_orbits": len(discovered),
        "rooted_wall_orbits": len(rooted),
        "candidate_coordinate_count": candidate_count,
        "walls": [],
        "neighbor_signatures": [],
        "best_crossing": None,
        "valid_numerical_witness": None,
    }
    best_ratio = -math.inf
    neighbor_signatures: set[str] = set()

    for wall_index, wall_record in enumerate(selected):
        wall = wall_record["wall_matrix"]
        root = float(wall_record["parameter_root"])
        direction = wall_record["parameter_direction"]
        before_parameters = center + 0.99 * root * direction
        after_parameters = center + 1.01 * root * direction
        before_geometry = voronoi_geometry(
            basis_from_parameters(
                evaluator.basis0, before_parameters
            )
        )
        after_geometry = voronoi_geometry(
            basis_from_parameters(
                evaluator.basis0, after_parameters
            )
        )
        adjacent_verified = (
            before_geometry.signature == source_geometry.signature
            and after_geometry.signature != source_geometry.signature
        )
        crossing_records: list[dict] = []
        wall_best: MetricEvaluation | None = None
        wall_best_signature: str | None = None
        wall_best_overshoot: float | None = None
        for overshoot in args.overshoots:
            parameters = center + root * (1.0 + overshoot) * direction
            evaluation = evaluator.evaluate(parameters)
            geometry = voronoi_geometry(evaluation.basis)
            signed_slack = wall_slack(
                evaluator.basis0, parameters, wall
            )
            crossing_records.append(
                {
                    "overshoot": overshoot,
                    "signed_wall_slack": signed_slack,
                    "voronoi_signature": geometry.signature,
                    "changed_from_source": (
                        geometry.signature
                        != source_geometry.signature
                    ),
                    **compact_evaluation(evaluation),
                }
            )
            if (
                signed_slack < 0
                and geometry.signature != source_geometry.signature
                and (
                    wall_best is None
                    or evaluation.min_ratio > wall_best.min_ratio
                )
            ):
                wall_best = evaluation
                wall_best_signature = geometry.signature
                wall_best_overshoot = overshoot

        result = {
            "wall_index": wall_index,
            "discovery_rank": wall_record["discovery_rank"],
            "source_slack": wall_record["source_slack"],
            "wall_frobenius_norm": wall_record[
                "wall_frobenius_norm"
            ],
            "normalized_gram_gap": wall_record[
                "normalized_gram_gap"
            ],
            "parameter_root": root,
            "parameter_gradient_norm": wall_record[
                "parameter_gradient_norm"
            ],
            "parameter_direction": direction.tolist(),
            "wall_matrix": wall.astype(int).tolist(),
            "representative_active_rows": wall_record[
                "representative_active_rows"
            ].astype(int).tolist(),
            "candidate_row": wall_record["candidate_row"].astype(
                int
            ).tolist(),
            "circuit_coefficients": wall_record[
                "circuit_coefficients"
            ].astype(int).tolist(),
            "source_vertex_multiplicity": wall_record[
                "source_vertex_multiplicity"
            ],
            "signature_before_root": before_geometry.signature,
            "signature_after_root": after_geometry.signature,
            "adjacent_verified": adjacent_verified,
            "crossings": crossing_records,
            "best_crossing": None,
        }
        if wall_best is not None and wall_best_signature is not None:
            wall_best = evaluator.evaluate(
                wall_best.parameters, with_witnesses=True
            )
            best_payload = {
                **wall_best.as_json(),
                "overshoot": wall_best_overshoot,
                "signed_wall_slack": wall_slack(
                    evaluator.basis0, wall_best.parameters, wall
                ),
                "voronoi_signature": wall_best_signature,
            }
            result["best_crossing"] = best_payload
            neighbor_signatures.add(wall_best_signature)
            if wall_best.min_ratio > best_ratio:
                best_ratio = wall_best.min_ratio
                payload["best_crossing"] = {
                    "wall_index": wall_index,
                    "wall_matrix": wall.astype(int).tolist(),
                    **best_payload,
                }
        payload["walls"].append(result)
        payload["neighbor_signatures"] = sorted(neighbor_signatures)
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"wall={wall_index:2d} root={root:.6g} "
            f"adjacent={adjacent_verified} "
            f"best={wall_best.min_ratio if wall_best else None} "
            f"signature={wall_best_signature[:12] if wall_best_signature else None}",
            flush=True,
        )

    payload["valid_numerical_witness"] = (
        payload["best_crossing"] is not None
        and payload["best_crossing"]["min_ratio"] >= 1.0
    )
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL best="
        f"{payload['best_crossing']['min_ratio'] if payload['best_crossing'] else None} "
        f"neighbors={len(neighbor_signatures)} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
