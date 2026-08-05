"""Search independent valid periods with sparse finite conflict graphs.

The prime-primary search normally stops as soon as no forbidden displacement
lies in the kernel.  For a non-coset coloring that is only the first gate:
the image of the forbidden set in the quotient should also omit many nonzero
group elements.  Those omitted elements are the compatible differences from
which larger color classes can be formed.

This campaign samples exact valid quotient maps, canonicalizes their kernels,
and ranks them by the number of distinct signed forbidden images.  Periods
nested in the known 343-color kernel are labelled separately.  For every
record connection count, CP-SAT tests the minimum independence number needed
to improve the Euclidean upper bound.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sympy import Matrix

from determinant_repair import exact_det, load_preset
from prime_radon import (
    PrimarySearch,
    hnf_columns,
    image_size,
    kernel_basis,
    killed_mask,
    smith_diagonal,
)
from prime_row_opt import _forbidden_with_weights


def parse_moduli(text: str) -> list[int]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("expected a nonempty modulus list")
    return [int(value) for value in raw]


def signed_connection_images(
    forbidden: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
) -> np.ndarray:
    """Return sorted distinct nonzero signed quotient images."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    rows_array = [np.asarray(row, dtype=np.int64) for row in rows]
    if len(rows_array) != len(moduli):
        raise ValueError("one modulus is required per quotient row")
    images = np.column_stack(
        [
            (forbidden @ row) % int(modulus)
            for row, modulus in zip(rows_array, moduli)
        ]
    )
    modulus_array = np.asarray(moduli, dtype=np.int64)
    signed = np.vstack([images, (-images) % modulus_array])
    unique = np.unique(signed, axis=0)
    nonzero = unique[np.any(unique != 0, axis=1)]
    return nonzero.astype(np.int64)


def is_nested_sublattice(
    child_columns: np.ndarray,
    parent_columns: np.ndarray,
) -> bool:
    """Whether ``child Z^n`` is contained in ``parent Z^n``."""
    exact = Matrix(
        np.asarray(parent_columns, dtype=np.int64).tolist()
    ).inv() * Matrix(np.asarray(child_columns, dtype=np.int64).tolist())
    return all(value.q == 1 for value in exact)


def quotient_independent_set_target(
    connection_images: np.ndarray,
    moduli: Sequence[int],
    target_size: int,
    *,
    time_limit: float = 30.0,
    workers: int = 8,
) -> dict:
    """Target-independent-set decision in direct-product coordinates.

    Translation fixes the zero vertex.  Its possible companions are precisely
    the nonzero quotient elements absent from the signed connection set, which
    is usually tiny even when the full quotient has thousands of elements.
    """
    moduli_array = np.asarray(moduli, dtype=np.int64)
    connection_images = np.asarray(connection_images, dtype=np.int64)
    if connection_images.ndim != 2 or connection_images.shape[1] != len(
        moduli_array
    ):
        raise ValueError("connection images and moduli are incompatible")
    if target_size < 1:
        raise ValueError("target size must be positive")
    connection_set = {
        tuple(int(value) for value in row)
        for row in connection_images
    }
    elements = [
        tuple(int(value) for value in values)
        for values in np.ndindex(*(int(modulus) for modulus in moduli_array))
    ]
    zero = tuple(0 for _ in moduli_array)
    candidates = [
        element
        for element in elements
        if element == zero or element not in connection_set
    ]
    if len(candidates) < target_size:
        return {
            "status": "INFEASIBLE",
            "feasible": False,
            "proven_infeasible": True,
            "vertices": [],
            "candidate_vertices": len(candidates),
            "edge_constraints": 0,
            "elapsed_seconds": 0.0,
            "solver": "direct-product cardinality precheck",
        }
    edges: list[tuple[int, int]] = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            difference = tuple(
                (
                    candidates[right][coordinate]
                    - candidates[left][coordinate]
                )
                % int(moduli_array[coordinate])
                for coordinate in range(len(moduli_array))
            )
            if difference in connection_set:
                edges.append((left, right))
    model = cp_model.CpModel()
    variables = [
        model.NewBoolVar(f"x_{index}") for index in range(len(candidates))
    ]
    model.Add(variables[candidates.index(zero)] == 1)
    for left, right in edges:
        model.Add(variables[left] + variables[right] <= 1)
    model.Add(sum(variables) >= target_size)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    started = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - started
    feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    selected = (
        [
            list(candidates[index])
            for index, variable in enumerate(variables)
            if solver.Value(variable)
        ]
        if feasible
        else []
    )
    labels = {
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.OPTIMAL: "OPTIMAL",
    }
    return {
        "status": labels.get(status, str(status)),
        "feasible": feasible,
        "proven_infeasible": status == cp_model.INFEASIBLE,
        "vertices": selected,
        "candidate_vertices": len(candidates),
        "edge_constraints": len(edges),
        "elapsed_seconds": elapsed,
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
        "solver": "OR-Tools CP-SAT on direct-product quotient residues",
    }


def quotient_matching_coloring(
    connection_images: np.ndarray,
    moduli: Sequence[int],
    target_colors: int,
    *,
    time_limit: float = 60.0,
) -> dict:
    """Merge compatible residue pairs using a HiGHS maximum-matching MIP."""
    moduli_array = np.asarray(moduli, dtype=np.int64)
    connection_set = {
        tuple(int(value) for value in row)
        for row in np.asarray(connection_images, dtype=np.int64)
    }
    elements = [
        tuple(int(value) for value in values)
        for values in np.ndindex(*(int(modulus) for modulus in moduli_array))
    ]
    vertex_count = len(elements)
    required_pairs = max(0, vertex_count - int(target_colors))
    if required_pairs == 0:
        return {
            "success": True,
            "optimal": True,
            "status": 0,
            "message": "target has at least one color per quotient element",
            "matching_size": 0,
            "required_pairs": 0,
            "color_count": vertex_count,
            "colors": [[list(element)] for element in elements],
            "solver": "closed form",
        }
    index = {element: number for number, element in enumerate(elements)}
    zero = tuple(0 for _ in moduli_array)
    missing = [
        element
        for element in elements
        if element != zero and element not in connection_set
    ]
    edges: set[tuple[int, int]] = set()
    for left, element in enumerate(elements):
        for difference in missing:
            right_element = tuple(
                (
                    element[coordinate] + difference[coordinate]
                )
                % int(moduli_array[coordinate])
                for coordinate in range(len(moduli_array))
            )
            right = index[right_element]
            if left < right:
                edges.add((left, right))
            elif right < left:
                edges.add((right, left))
    edge_list = sorted(edges)
    if len(edge_list) < required_pairs:
        return {
            "success": False,
            "optimal": True,
            "status": "edge-count precheck",
            "matching_size": 0,
            "required_pairs": required_pairs,
            "compatible_edges": len(edge_list),
            "solver": "closed form",
        }
    edge_array = np.asarray(edge_list, dtype=np.int64)
    columns = np.repeat(np.arange(len(edge_array), dtype=np.int64), 2)
    rows = edge_array.reshape(-1)
    incidence = coo_matrix(
        (
            np.ones(2 * len(edge_array), dtype=np.float64),
            (rows, columns),
        ),
        shape=(vertex_count, len(edge_array)),
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        c=-np.ones(len(edge_array), dtype=np.float64),
        integrality=np.ones(len(edge_array), dtype=np.int8),
        bounds=Bounds(
            np.zeros(len(edge_array)), np.ones(len(edge_array))
        ),
        constraints=LinearConstraint(
            incidence,
            lb=np.full(vertex_count, -np.inf),
            ub=np.ones(vertex_count),
        ),
        options={"time_limit": float(time_limit), "presolve": True},
    )
    selected_edges = (
        edge_array[np.flatnonzero(result.x > 0.5)]
        if result.x is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    matching_size = len(selected_edges)
    success = matching_size >= required_pairs
    colors: list[list[list[int]]] = []
    if success:
        used: set[int] = set()
        for left, right in selected_edges[:required_pairs]:
            left = int(left)
            right = int(right)
            if left in used or right in used:
                raise AssertionError("HiGHS matching repeats a vertex")
            used.add(left)
            used.add(right)
            colors.append([list(elements[left]), list(elements[right])])
        colors.extend(
            [[list(element)]]
            for vertex, element in enumerate(elements)
            if vertex not in used
        )
        if len(colors) != vertex_count - required_pairs:
            raise AssertionError("decoded matching has the wrong color count")
        for color in colors:
            if len(color) == 2:
                difference = tuple(
                    (color[1][coordinate] - color[0][coordinate])
                    % int(moduli_array[coordinate])
                    for coordinate in range(len(moduli_array))
                )
                if difference in connection_set:
                    raise AssertionError("matching uses a conflicting pair")
    return {
        "success": success,
        "optimal": bool(result.status == 0),
        "status": int(result.status),
        "message": str(result.message),
        "matching_size": matching_size,
        "required_pairs": required_pairs,
        "compatible_edges": len(edge_array),
        "color_count": len(colors) if success else None,
        "colors": colors if success else None,
        "elapsed_seconds": time.perf_counter() - started,
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "solver": "HiGHS maximum-matching MIP via scipy.optimize.milp",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--moduli", type=parse_moduli, default=[7, 7, 7, 2]
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--keep", type=int, default=40)
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--decision-time-limit", type=float, default=30.0)
    parser.add_argument("--decision-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.sweeps < 1 or args.top < 1 or args.keep < 1:
        parser.error("search budgets must be positive")
    if args.target_colors < 1 or args.decision_time_limit <= 0:
        parser.error("target and time limit must be positive")

    lattice, basis, diameter, _, source_kernel = load_preset("d6")
    forbidden, _, _ = _forbidden_with_weights(basis, diameter)
    target_index = math.prod(args.moduli)
    necessary_alpha = math.ceil(target_index / args.target_colors)
    search = PrimarySearch(forbidden, args.moduli, seed=args.seed)
    started = time.perf_counter()
    archive: dict[tuple[int, ...], dict] = {}
    valid_samples = 0
    invalid_samples = 0
    best_connections = target_index
    record_graphs: list[dict] = []
    target_status_counts: dict[str, int] = {}
    promising_periods: list[dict] = []

    payload: dict = {
        "method": (
            "random primary-coordinate descent for valid periods, ranked by "
            "distinct signed forbidden quotient images"
        ),
        "lattice": lattice,
        "dimension": len(source_kernel),
        "source_index": abs(exact_det(source_kernel)),
        "source_kernel_basis_columns": source_kernel.astype(int).tolist(),
        "moduli": args.moduli,
        "target_period_index": target_index,
        "target_colors": args.target_colors,
        "necessary_independence_number_for_target": necessary_alpha,
        "forbidden_projective_pairs": len(forbidden),
        "settings": {
            "samples": args.samples,
            "sweeps": args.sweeps,
            "top": args.top,
            "keep": args.keep,
            "decision_time_limit": args.decision_time_limit,
            "decision_workers": args.decision_workers,
            "seed": args.seed,
        },
        "valid_samples": 0,
        "invalid_samples": 0,
        "unique_valid_kernels": 0,
        "best_connection_count": None,
        "records": [],
        "promising_periods": [],
    }

    def save() -> None:
        ranked = sorted(
            archive.values(),
            key=lambda record: (
                record["connection_keys"],
                record["nested_in_source_kernel"],
                record["sample"],
            ),
        )[: args.keep]
        payload["valid_samples"] = valid_samples
        payload["invalid_samples"] = invalid_samples
        payload["unique_valid_kernels"] = len(archive)
        payload["best_connection_count"] = (
            min(
                (record["connection_keys"] for record in archive.values()),
                default=None,
            )
        )
        payload["records"] = ranked
        payload["record_graphs"] = record_graphs
        payload["target_status_counts"] = target_status_counts
        payload["promising_periods"] = promising_periods
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"period={target_index} moduli={args.moduli} "
        f"target={args.target_colors} need-alpha={necessary_alpha}",
        flush=True,
    )
    for sample in range(args.samples):
        score, rows, sweeps = search.descend(
            max_sweeps=args.sweeps,
            top=args.top,
            kick_probability=0.15,
            temperature=0.35,
        )
        exact_killed = int(
            killed_mask(forbidden, rows, args.moduli).sum()
        )
        if exact_killed != int(round(score)):
            raise AssertionError("primary search score mismatch")
        if exact_killed:
            invalid_samples += 1
            continue
        if image_size(rows, args.moduli, len(source_kernel)) != target_index:
            invalid_samples += 1
            continue
        valid_samples += 1
        kernel = hnf_columns(
            kernel_basis(rows, args.moduli, len(source_kernel))
        )
        if abs(exact_det(kernel)) != target_index:
            raise AssertionError("period determinant mismatch")
        key = tuple(int(value) for value in kernel.flat)
        if key in archive:
            continue
        connections = signed_connection_images(
            forbidden, rows, args.moduli
        )
        connection_count = len(connections)
        record = {
            "sample": sample,
            "sweeps": sweeps,
            "rows": [
                np.asarray(row, dtype=np.int64).astype(int).tolist()
                for row in rows
            ],
            "kernel_basis_columns": kernel.astype(int).tolist(),
            "kernel_smith": smith_diagonal(kernel),
            "determinant": target_index,
            "connection_keys": connection_count,
            "missing_nonzero_quotient_classes": (
                target_index - 1 - connection_count
            ),
            "nested_in_source_kernel": is_nested_sublattice(
                kernel, source_kernel
            ),
        }
        decision = quotient_independent_set_target(
            connections,
            args.moduli,
            necessary_alpha,
            time_limit=args.decision_time_limit,
            workers=args.decision_workers,
        )
        record["target_independent_set"] = decision
        target_status_counts[decision["status"]] = (
            target_status_counts.get(decision["status"], 0) + 1
        )
        if decision["feasible"]:
            matching = quotient_matching_coloring(
                connections,
                args.moduli,
                args.target_colors,
                time_limit=args.decision_time_limit,
            )
            record["matching_coloring"] = matching
            promising_periods.append(
                {
                    "sample": sample,
                    "kernel_basis_columns": kernel.astype(int).tolist(),
                    "rows": record["rows"],
                    "connection_keys": connection_count,
                    "nested_in_source_kernel": record[
                        "nested_in_source_kernel"
                    ],
                    "target_independent_set": decision,
                    "matching_coloring": matching,
                }
            )
        archive[key] = record
        if connection_count < best_connections:
            best_connections = connection_count
            graph_record = {
                "sample": sample,
                "connection_keys": connection_count,
                "kernel_basis_columns": kernel.astype(int).tolist(),
                "nested_in_source_kernel": record[
                    "nested_in_source_kernel"
                ],
                "target_independent_set": decision,
            }
            if decision["proven_infeasible"]:
                graph_record["independence_number_upper_bound"] = (
                    necessary_alpha - 1
                )
            elif decision["feasible"]:
                graph_record["independence_number_lower_bound"] = len(
                    decision["vertices"]
                )
            record_graphs.append(graph_record)
            print(
                f"sample={sample} valid={valid_samples} unique={len(archive)} "
                f"connections={connection_count}/{target_index-1} "
                f"nested={record['nested_in_source_kernel']} "
                f"target-status={decision['status']}",
                flush=True,
            )
            save()
        if (sample + 1) % 100 == 0:
            print(
                f"progress {sample+1}/{args.samples} valid={valid_samples} "
                f"unique={len(archive)} best-connections={best_connections}",
                flush=True,
            )
            save()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
