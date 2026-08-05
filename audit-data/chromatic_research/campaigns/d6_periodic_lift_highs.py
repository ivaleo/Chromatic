"""Search non-coset periodic colorings above a near-feasible lattice kernel.

Let ``K`` be an index-336 sublattice that is close to, but not itself, a
valid color class.  For a prime ``p`` and a character ``a`` on the coordinates
of ``K``, refine the period to

    P = K * ker(a : Z^6 -> F_p).

The quotient ``Z^6/P`` has ``p`` layers of 336 vertices.  A 336-coloring can
use every color once per layer, provided the selected ``p`` vertices of every
color form an independent set in the finite conflict Cayley graph.

For a fixed order of layers this script builds those transversals greedily.
Each new layer is an assignment problem: match its 336 vertices to the 336
partial color classes, allowing only nonconflicting pairs.  The assignment LP
is totally unimodular and is solved by open HiGHS through SciPy.  Multiple
random LP objectives explore different perfect matchings.

This remains a numerical discovery calculation.  A successful coloring must
subsequently be rationalized and checked by an independent exact verifier.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Sequence

import combigeo
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sympy import Matrix, isprime

from chromatic_research.core.active_metric_refine import _load_problem
from chromatic_research.core.prime_radon import (
    hnf_columns,
    kernel_basis,
    projective_forms,
)
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


def quotient_map(
    columns: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return determinant and exact adjugate map for Z^n / columns Z^n."""
    matrix = Matrix(np.asarray(columns, dtype=np.int64).tolist())
    determinant = abs(int(matrix.det()))
    if determinant < 1:
        raise ValueError("quotient columns must be nonsingular")
    adjugate = np.asarray(matrix.adjugate().tolist(), dtype=object)
    return determinant, adjugate


def quotient_key(
    vector: Sequence[int] | np.ndarray,
    adjugate: np.ndarray,
    determinant: int,
) -> tuple[int, ...]:
    values = adjugate @ np.asarray(vector, dtype=object)
    return tuple(int(value) % determinant for value in values)


def quotient_representatives(
    columns: np.ndarray,
) -> tuple[list[np.ndarray], dict[tuple[int, ...], int]]:
    """Enumerate exact coordinate representatives by generator BFS."""
    determinant, adjugate = quotient_map(columns)
    n = len(columns)
    zero = tuple(0 for _ in range(n))
    generator_keys = [
        quotient_key(np.eye(n, dtype=np.int64)[index], adjugate, determinant)
        for index in range(n)
    ]
    keys = [zero]
    representatives = [np.zeros(n, dtype=np.int64)]
    index_by_key = {zero: 0}
    cursor = 0
    while cursor < len(keys):
        key = keys[cursor]
        representative = representatives[cursor]
        cursor += 1
        for coordinate, generator in enumerate(generator_keys):
            next_key = tuple(
                (key[position] + generator[position]) % determinant
                for position in range(n)
            )
            if next_key in index_by_key:
                continue
            next_representative = representative.copy()
            next_representative[coordinate] += 1
            index_by_key[next_key] = len(keys)
            keys.append(next_key)
            representatives.append(next_representative)
    if len(keys) != determinant:
        raise AssertionError(
            f"quotient BFS found {len(keys)} elements, expected {determinant}"
        )
    return representatives, index_by_key


def solve_assignment(
    allowed: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, dict]:
    """Solve one perfect matching LP with HiGHS."""
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.ndim != 2 or allowed.shape[0] != allowed.shape[1]:
        raise ValueError("assignment mask must be square")
    n = len(allowed)
    if np.any(allowed.sum(axis=1) == 0) or np.any(
        allowed.sum(axis=0) == 0
    ):
        return None, {
            "success": False,
            "status": "empty row or column",
            "allowed_edges": int(allowed.sum()),
        }
    edges = np.argwhere(allowed)
    edge_count = len(edges)
    columns = np.arange(edge_count, dtype=np.int64)
    equality_rows = np.concatenate(
        [edges[:, 0], n + edges[:, 1]]
    )
    equality_columns = np.concatenate([columns, columns])
    matrix = coo_matrix(
        (
            np.ones(2 * edge_count, dtype=np.float64),
            (equality_rows, equality_columns),
        ),
        shape=(2 * n, edge_count),
    ).tocsr()
    objective = rng.random(edge_count) * 1e-7
    started = time.perf_counter()
    result = linprog(
        objective,
        A_eq=matrix,
        b_eq=np.ones(2 * n, dtype=np.float64),
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    metadata = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "allowed_edges": int(edge_count),
        "elapsed_seconds": time.perf_counter() - started,
        "objective": float(result.fun) if result.success else None,
        "highs_crossover_iterations": int(
            getattr(result, "crossover_nit", 0)
        ),
    }
    if not result.success:
        return None, metadata
    selected = np.flatnonzero(result.x > 0.5)
    if len(selected) != n:
        metadata["success"] = False
        metadata["message"] = (
            f"HiGHS assignment was not integral: {len(selected)} selected"
        )
        return None, metadata
    assignment = np.full(n, -1, dtype=np.int64)
    for edge_index in selected:
        color, vertex = edges[int(edge_index)]
        if assignment[color] != -1:
            raise AssertionError("assignment repeats a color")
        assignment[color] = int(vertex)
    if np.any(assignment < 0) or len(set(assignment.tolist())) != n:
        raise AssertionError("decoded HiGHS assignment is not a permutation")
    if not np.all(allowed[np.arange(n), assignment]):
        raise AssertionError("decoded HiGHS assignment uses a forbidden edge")
    return assignment, metadata


def pair_conflict_matrices(
    layer_keys: Sequence[np.ndarray],
    connection_keys: set[tuple[int, ...]],
    determinant: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Precompute every cross-layer conflict matrix."""
    matrices: dict[tuple[int, int], np.ndarray] = {}
    layer_count = len(layer_keys)
    size = len(layer_keys[0])
    for left in range(layer_count):
        for right in range(left + 1, layer_count):
            matrix = np.zeros((size, size), dtype=bool)
            left_keys = layer_keys[left]
            right_keys = layer_keys[right]
            for left_index in range(size):
                base = left_keys[left_index]
                for right_index in range(size):
                    difference = tuple(
                        int(value)
                        for value in (
                            right_keys[right_index] - base
                        )
                        % determinant
                    )
                    matrix[left_index, right_index] = (
                        difference in connection_keys
                    )
            matrices[(left, right)] = matrix
    return matrices


def conflict_row(
    matrices: dict[tuple[int, int], np.ndarray],
    old_layer: int,
    old_vertex: int,
    new_layer: int,
) -> np.ndarray:
    if old_layer < new_layer:
        return matrices[(old_layer, new_layer)][old_vertex]
    return matrices[(new_layer, old_layer)][:, old_vertex]


def sequential_coloring(
    matrices: dict[tuple[int, int], np.ndarray],
    layer_order: Sequence[int],
    size: int,
    rng: np.random.Generator,
) -> tuple[list[list[tuple[int, int]]] | None, list[dict]]:
    """Build independent transversals with successive HiGHS assignments."""
    colors: list[list[tuple[int, int]]] = [
        [(0, color)] for color in range(size)
    ]
    history: list[dict] = []
    for layer in layer_order:
        allowed = np.ones((size, size), dtype=bool)
        for color, assigned in enumerate(colors):
            for old_layer, old_vertex in assigned:
                allowed[color] &= ~conflict_row(
                    matrices,
                    old_layer,
                    old_vertex,
                    int(layer),
                )
        assignment, metadata = solve_assignment(allowed, rng)
        metadata["layer"] = int(layer)
        metadata["minimum_allowed_per_color"] = int(
            allowed.sum(axis=1).min()
        )
        metadata["minimum_allowed_per_vertex"] = int(
            allowed.sum(axis=0).min()
        )
        history.append(metadata)
        if assignment is None:
            return None, history
        for color, vertex in enumerate(assignment):
            colors[color].append((int(layer), int(vertex)))
    return colors, history


def minimum_coloring_ratio(
    colors: Sequence[Sequence[np.ndarray]],
    basis: np.ndarray,
    diameter: float,
    facets: Sequence[tuple[Sequence[float], float]],
) -> tuple[float, list[int], int]:
    """Check every same-color displacement with the full Voronoi facets."""
    best_ratio = math.inf
    best_difference: list[int] = []
    unique: set[tuple[int, ...]] = set()
    for color in colors:
        for left, right in itertools.combinations(color, 2):
            difference = np.asarray(right, dtype=np.int64) - np.asarray(
                left, dtype=np.int64
            )
            first_nonzero = next(
                (int(value) for value in difference if value),
                1,
            )
            if first_nonzero < 0:
                difference = -difference
            key = tuple(int(value) for value in difference)
            if key in unique:
                continue
            unique.add(key)
            physical = difference @ basis
            distance = 2.0 * combigeo.dist_to_halfspaces(
                (0.5 * physical).tolist(), facets
            )
            ratio = float(distance / diameter)
            if ratio < best_ratio:
                best_ratio = ratio
                best_difference = list(key)
    return best_ratio, best_difference, len(unique)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--prime", type=int, default=7)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--max-characters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=6339801)
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prime < 2 or not isprime(args.prime):
        parser.error("--prime must be prime")
    if args.trials < 1 or args.max_characters < 0:
        parser.error("invalid periodic-lift budget")

    metric = json.loads(args.metric.read_text())
    (
        _,
        _,
        _,
        _,
        source_kernel,
        source_evaluator,
    ) = _load_problem(args.metric, args.temperature, args.max_h_norm)
    parameters = np.asarray(
        metric["best"]["parameters"], dtype=np.float64
    )
    evaluation = source_evaluator.evaluate(
        parameters, with_witnesses=True
    )
    basis = evaluation.basis
    diameter = evaluation.diameter
    facets = combigeo.relevant_facets(basis.tolist())
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    source_determinant = abs(
        int(Matrix(source_kernel.astype(int).tolist()).det())
    )
    base_representatives, _ = quotient_representatives(source_kernel)
    if len(base_representatives) != source_determinant:
        raise AssertionError("source quotient size mismatch")

    source_det, source_adjugate = quotient_map(source_kernel)
    zero_source = tuple(0 for _ in range(len(source_kernel)))
    inverse_source = Matrix(source_kernel.astype(int).tolist()).inv()
    conflict_coordinates: list[np.ndarray] = []
    for vector in forbidden:
        if (
            quotient_key(vector, source_adjugate, source_det)
            != zero_source
        ):
            continue
        exact = inverse_source * Matrix(
            np.asarray(vector, dtype=np.int64).tolist()
        )
        if any(value.q != 1 for value in exact):
            raise AssertionError("source-kernel coordinate is not integral")
        conflict_coordinates.append(
            np.asarray([int(value) for value in exact], dtype=np.int64)
        )
    conflict_array = np.asarray(conflict_coordinates, dtype=np.int64)
    characters = projective_forms(len(source_kernel), args.prime)
    good_mask = np.all(
        (conflict_array @ characters.T) % args.prime != 0,
        axis=0,
    )
    good_characters = characters[good_mask]
    if args.max_characters:
        good_characters = good_characters[: args.max_characters]

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    payload: dict = {
        "method": (
            "prime-index period lift and non-coset layer coloring by "
            "successive totally-unimodular HiGHS assignment LPs"
        ),
        "source_metric": str(args.metric),
        "n": len(source_kernel),
        "dimension": len(source_kernel),
        "source_index": source_determinant,
        "prime": args.prime,
        "period_index": source_determinant * args.prime,
        "target_colors": source_determinant,
        "forbidden_projective_pairs": len(forbidden),
        "source_kernel_conflicts": len(conflict_coordinates),
        "projective_characters": len(characters),
        "loop_free_characters": int(good_mask.sum()),
        "settings": {
            "trials": args.trials,
            "max_characters": args.max_characters,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
        },
        "character_summaries": [],
        "attempts": [],
        "coloring": None,
        "valid_numerical_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"source={source_determinant} conflicts={len(conflict_coordinates)} "
        f"p={args.prime} characters={len(characters)} "
        f"loop-free={int(good_mask.sum())}",
        flush=True,
    )
    for character_index, character in enumerate(good_characters):
        lift = hnf_columns(
            kernel_basis(
                [np.asarray(character, dtype=np.int64)],
                [args.prime],
                len(source_kernel),
            )
        )
        period = hnf_columns(source_kernel @ lift)
        period_determinant, period_adjugate = quotient_map(period)
        if period_determinant != source_determinant * args.prime:
            raise AssertionError("lifted period has the wrong determinant")
        zero_period = tuple(0 for _ in range(len(period)))
        connection_keys: set[tuple[int, ...]] = set()
        for vector in forbidden:
            connection_keys.add(
                quotient_key(vector, period_adjugate, period_determinant)
            )
            connection_keys.add(
                quotient_key(-vector, period_adjugate, period_determinant)
            )
        if zero_period in connection_keys:
            raise AssertionError("screened character still has a loop")

        pivot = next(
            index
            for index, value in enumerate(character)
            if int(value) % args.prime
        )
        lift_coordinate = np.zeros(len(character), dtype=np.int64)
        lift_coordinate[pivot] = pow(
            int(character[pivot]), -1, args.prime
        )
        translation = source_kernel @ lift_coordinate
        layer_representatives = [
            [
                representative + layer * translation
                for representative in base_representatives
            ]
            for layer in range(args.prime)
        ]
        layer_keys = [
            np.asarray(
                [
                    quotient_key(
                        representative,
                        period_adjugate,
                        period_determinant,
                    )
                    for representative in layer
                ],
                dtype=np.int64,
            )
            for layer in layer_representatives
        ]
        all_keys = {
            tuple(int(value) for value in key)
            for layer in layer_keys
            for key in layer
        }
        if len(all_keys) != period_determinant:
            raise AssertionError("layer representatives do not cover quotient")
        matrices = pair_conflict_matrices(
            layer_keys, connection_keys, period_determinant
        )
        payload["character_summaries"].append(
            {
                "character_index": character_index,
                "character": character.astype(int).tolist(),
                "connection_keys": len(connection_keys),
            }
        )
        save()
        print(
            f"character {character_index + 1}/{len(good_characters)} "
            f"{character.tolist()} conn={len(connection_keys)}",
            flush=True,
        )
        for trial in range(args.trials):
            if trial < args.prime - 1:
                first_layer = trial + 1
                remaining = [
                    layer
                    for layer in range(1, args.prime)
                    if layer != first_layer
                ]
                order = [
                    first_layer,
                    *rng.permutation(
                        np.asarray(remaining, dtype=np.int64)
                    ).tolist(),
                ]
            else:
                order = rng.permutation(
                    np.arange(1, args.prime, dtype=np.int64)
                ).tolist()
            colors, history = sequential_coloring(
                matrices,
                order,
                source_determinant,
                rng,
            )
            attempt = {
                "character_index": character_index,
                "character": character.astype(int).tolist(),
                "trial": trial,
                "layer_order": order,
                "success": colors is not None,
                "assignment_history": history,
            }
            payload["attempts"].append(attempt)
            save()
            print(
                f"  trial {trial + 1}/{args.trials} order={order} "
                f"layers={len(history)} success={colors is not None}",
                flush=True,
            )
            if colors is None:
                continue
            coordinate_colors = [
                [
                    layer_representatives[layer][vertex]
                    for layer, vertex in color
                ]
                for color in colors
            ]
            minimum_ratio, witness, unique_differences = (
                minimum_coloring_ratio(
                    coordinate_colors,
                    basis,
                    diameter,
                    facets,
                )
            )
            payload["coloring"] = {
                "character": character.astype(int).tolist(),
                "period_basis_columns": period.astype(int).tolist(),
                "period_determinant": period_determinant,
                "colors": [
                    [
                        np.asarray(vertex, dtype=np.int64)
                        .astype(int)
                        .tolist()
                        for vertex in color
                    ]
                    for color in coordinate_colors
                ],
                "color_count": len(coordinate_colors),
                "vertices_per_color": args.prime,
                "covered_quotient_vertices": sum(
                    len(color) for color in coordinate_colors
                ),
                "unique_same_color_differences": unique_differences,
                "minimum_distance_ratio": minimum_ratio,
                "minimum_witness_difference": witness,
            }
            payload["valid_numerical_witness"] = minimum_ratio >= 1.0
            save()
            print(
                f"FOUND colors={len(coordinate_colors)} "
                f"minimum-ratio={minimum_ratio:.12f}",
                flush=True,
            )
            return 0
    save()
    print("no periodic coloring found", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
