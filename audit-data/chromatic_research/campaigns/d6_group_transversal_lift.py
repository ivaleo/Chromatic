"""Coprime group-transversal lifts above a near-feasible lattice kernel.

Let ``K`` be an index-``C`` lattice kernel and refine it to

    P = K * ker(a : Z^n -> F_p),

where ``p`` is prime and coprime to ``C``.  The finite quotient splits as

    Z^n / P = H x F_p,       |H| = C.

Choose one point ``(y_l,l)`` in every layer and translate this block by all
``h in H``.  The resulting ``C`` blocks partition the quotient, so they are
color classes exactly when no block difference is a forbidden quotient
element.

For every projective character this script first computes the exact forbidden
difference sets in the ``p`` layers.  This gives an exhaustive pairwise
screen without constructing dense ``C x C`` matrices.  Surviving phase CSPs
are sent to a lazy binary MIP: HiGHS chooses one phase in ``H`` per layer,
the exact verifier returns every conflicting selected pair, and those
two-variable no-goods are added to the next master.

All finite-group and coloring arithmetic is exact.  The forbidden catalogue
comes from a floating metric checkpoint, so a found coloring is still a
numerical discovery until the metric is rationalized and audited separately.
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
from sympy import Matrix, isprime

from chromatic_research.core.active_metric_refine import _load_problem
from chromatic_research.campaigns.d6_periodic_lift_highs import (
    quotient_key,
    quotient_map,
    quotient_representatives,
    sequential_coloring,
)
from chromatic_research.core.prime_radon import hnf_columns, kernel_basis, projective_forms
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


def quotient_group_tables(
    columns: np.ndarray,
) -> tuple[
    list[np.ndarray],
    list[tuple[int, ...]],
    np.ndarray,
    np.ndarray,
]:
    """Return representatives, keys, inverses, and the subtraction table."""
    representatives, index_by_key = quotient_representatives(columns)
    determinant, adjugate = quotient_map(columns)
    keys = [
        quotient_key(representative, adjugate, determinant)
        for representative in representatives
    ]
    inverse = np.empty(determinant, dtype=np.int64)
    for index, key in enumerate(keys):
        negative = tuple((-value) % determinant for value in key)
        inverse[index] = int(index_by_key[negative])
    subtraction = np.empty(
        (determinant, determinant),
        dtype=np.int64,
    )
    for left, left_key in enumerate(keys):
        for right, right_key in enumerate(keys):
            difference = tuple(
                (left_key[position] - right_key[position]) % determinant
                for position in range(len(left_key))
            )
            subtraction[left, right] = int(index_by_key[difference])
    return representatives, keys, inverse, subtraction


def split_forbidden_coordinates(
    source_kernel: np.ndarray,
    forbidden: np.ndarray,
    prime: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Express forbidden vectors in the canonical ``H x F_p`` splitting."""
    source_kernel = np.asarray(source_kernel, dtype=np.int64)
    forbidden = np.asarray(forbidden, dtype=np.int64)
    source_order, source_adjugate = quotient_map(source_kernel)
    if math.gcd(source_order, prime) != 1:
        raise ValueError("the lift prime must be coprime to the source index")
    representatives, index_by_key = quotient_representatives(source_kernel)
    representative_keys = [
        quotient_key(value, source_adjugate, source_order)
        for value in representatives
    ]
    index_by_key = {
        key: int(index)
        for index, key in enumerate(representative_keys)
    }
    inverse_indices = np.empty(source_order, dtype=np.int64)
    for index, key in enumerate(representative_keys):
        negative = tuple((-value) % source_order for value in key)
        inverse_indices[index] = index_by_key[negative]

    # This scalar is 1 on H and 0 on the p-primary factor.
    complement_scalar = prime * pow(prime, -1, source_order)
    quotient_indices = np.empty(len(forbidden), dtype=np.int64)
    layer_coordinates = np.empty(
        (len(forbidden), len(source_kernel)),
        dtype=np.int64,
    )
    for vector_index, vector in enumerate(forbidden):
        key = quotient_key(vector, source_adjugate, source_order)
        quotient_index = index_by_key[key]
        quotient_indices[vector_index] = quotient_index
        remainder = (
            np.asarray(vector, dtype=object)
            - complement_scalar
            * np.asarray(representatives[quotient_index], dtype=object)
        )
        numerators = source_adjugate @ remainder
        coordinates: list[int] = []
        for numerator in numerators:
            numerator = int(numerator)
            if numerator % source_order:
                raise AssertionError(
                    "canonical complement did not leave a kernel vector"
                )
            coordinates.append(numerator // source_order)
        layer_coordinates[vector_index] = np.asarray(
            coordinates,
            dtype=np.int64,
        )
    return (
        quotient_indices,
        inverse_indices[quotient_indices],
        layer_coordinates,
        np.asarray(representatives, dtype=np.int64),
    )


def connection_table(
    layer_values: Sequence[int],
    quotient_indices: Sequence[int],
    inverse_indices: Sequence[int],
    prime: int,
    source_order: int,
) -> np.ndarray:
    """Build the symmetric forbidden subset of ``H x F_p``."""
    layers = np.remainder(
        np.asarray(layer_values, dtype=np.int64),
        prime,
    )
    quotient = np.asarray(quotient_indices, dtype=np.int64)
    inverse = np.asarray(inverse_indices, dtype=np.int64)
    if not (
        layers.shape == quotient.shape == inverse.shape
        and np.all((0 <= quotient) & (quotient < source_order))
        and np.all((0 <= inverse) & (inverse < source_order))
    ):
        raise ValueError("incompatible split forbidden coordinates")
    table = np.zeros((prime, source_order), dtype=bool)
    table[layers, quotient] = True
    table[np.remainder(-layers, prime), inverse] = True
    return table


def sampled_projective_forms(
    dimension: int,
    prime: int,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample distinct canonically normalized projective forms."""
    total = (prime**dimension - 1) // (prime - 1)
    target = min(int(count), total)
    if dimension < 1 or prime < 2 or target < 1:
        raise ValueError("invalid projective sample")
    rng = np.random.default_rng(seed)
    unique: dict[tuple[int, ...], tuple[int, ...]] = {}
    while len(unique) < target:
        batch_size = min(
            max(256, 2 * (target - len(unique))),
            65536,
        )
        batch = rng.integers(
            0,
            prime,
            size=(batch_size, dimension),
            dtype=np.int64,
        )
        for vector in batch:
            nonzero = np.flatnonzero(vector)
            if not len(nonzero):
                continue
            pivot = int(nonzero[0])
            inverse = pow(int(vector[pivot]), -1, prime)
            canonical = tuple(
                int(value)
                for value in np.remainder(vector * inverse, prime)
            )
            unique.setdefault(canonical, canonical)
            if len(unique) >= target:
                break
    return np.asarray(list(unique.values()), dtype=np.int64)


def transversal_conflicts(
    phases: Sequence[int],
    connections: np.ndarray,
    subtraction: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """List conflicting layer pairs of a group-transversal block."""
    phases = np.asarray(phases, dtype=np.int64)
    connections = np.asarray(connections, dtype=bool)
    subtraction = np.asarray(subtraction, dtype=np.int64)
    prime, source_order = connections.shape
    if (
        phases.shape != (prime,)
        or subtraction.shape != (source_order, source_order)
        or np.any((phases < 0) | (phases >= source_order))
    ):
        raise ValueError("invalid group-transversal state")
    conflicts: list[tuple[int, int, int, int]] = []
    for left in range(prime):
        for right in range(left + 1, prime):
            layer = (right - left) % prime
            difference = int(
                subtraction[int(phases[right]), int(phases[left])]
            )
            if connections[layer, difference]:
                conflicts.append(
                    (
                        left,
                        right,
                        int(phases[left]),
                        int(phases[right]),
                    )
                )
    return conflicts


def phase_coordinate_descent(
    connections: np.ndarray,
    subtraction: np.ndarray,
    *,
    restarts: int,
    sweeps: int,
    seed: int,
) -> dict:
    """Heuristically minimize the exact number of conflicting layer pairs."""
    connections = np.asarray(connections, dtype=bool)
    subtraction = np.asarray(subtraction, dtype=np.int64)
    prime, source_order = connections.shape
    if (
        prime < 2
        or subtraction.shape != (source_order, source_order)
        or restarts < 1
        or sweeps < 1
    ):
        raise ValueError("invalid phase-descent instance")
    started = time.perf_counter()
    domains = [np.asarray([0], dtype=np.int64)]
    for layer in range(1, prime):
        allowed = np.flatnonzero(~connections[layer]).astype(np.int64)
        if not len(allowed):
            return {
                "status": "PAIR_INFEASIBLE",
                "feasible": False,
                "proven_infeasible": True,
                "phases": None,
                "exact_conflicts": None,
                "restart": None,
                "sweep": None,
                "elapsed_seconds": time.perf_counter() - started,
                "solver": "exact-coordinate phase descent",
            }
        domains.append(allowed)
    rng = np.random.default_rng(seed)
    best_phases = None
    best_conflicts = prime * (prime - 1) // 2 + 1
    best_restart = -1
    best_sweep = -1
    evaluations = 0
    for restart in range(restarts):
        phases = np.zeros(prime, dtype=np.int64)
        for layer in range(1, prime):
            phases[layer] = int(rng.choice(domains[layer]))
        for sweep in range(sweeps):
            changed = False
            for layer in rng.permutation(
                np.arange(1, prime, dtype=np.int64)
            ):
                layer = int(layer)
                candidates = domains[layer]
                scores = np.zeros(len(candidates), dtype=np.int32)
                for other in range(prime):
                    if other == layer:
                        continue
                    if other < layer:
                        differences = subtraction[
                            candidates,
                            int(phases[other]),
                        ]
                        scores += connections[
                            (layer - other) % prime,
                            differences,
                        ]
                    else:
                        differences = subtraction[
                            int(phases[other]),
                            candidates,
                        ]
                        scores += connections[
                            (other - layer) % prime,
                            differences,
                        ]
                evaluations += len(candidates)
                minimum = int(scores.min())
                choices = np.flatnonzero(scores == minimum)
                chosen = int(candidates[int(rng.choice(choices))])
                changed |= chosen != int(phases[layer])
                phases[layer] = chosen
            conflicts = len(
                transversal_conflicts(
                    phases,
                    connections,
                    subtraction,
                )
            )
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_phases = phases.copy()
                best_restart = restart
                best_sweep = sweep
            if conflicts == 0:
                break
            if not changed or rng.random() < 0.18:
                perturbations = max(1, (prime - 1) // 6)
                for layer in rng.choice(
                    np.arange(1, prime, dtype=np.int64),
                    size=perturbations,
                    replace=False,
                ):
                    phases[int(layer)] = int(
                        rng.choice(domains[int(layer)])
                    )
        if best_conflicts == 0:
            break
    if best_phases is None:
        raise AssertionError("phase descent produced no incumbent")
    exact_conflicts = transversal_conflicts(
        best_phases,
        connections,
        subtraction,
    )
    if len(exact_conflicts) != best_conflicts:
        raise AssertionError("phase descent score fails exact recount")
    return {
        "status": "FEASIBLE" if not exact_conflicts else "HEURISTIC",
        "feasible": not exact_conflicts,
        "proven_infeasible": False,
        "phases": best_phases.astype(int).tolist(),
        "exact_conflicts": len(exact_conflicts),
        "conflicting_layer_pairs": [
            list(values) for values in exact_conflicts
        ],
        "restart": best_restart,
        "sweep": best_sweep,
        "evaluations": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
        "solver": "exact-coordinate phase descent",
    }


def canonical_conflict_matrices(
    connections: np.ndarray,
    subtraction: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Build cross-layer matrices for the canonical ``H`` cosets."""
    connections = np.asarray(connections, dtype=bool)
    subtraction = np.asarray(subtraction, dtype=np.int64)
    prime, source_order = connections.shape
    if subtraction.shape != (source_order, source_order):
        raise ValueError("invalid canonical layer instance")
    matrices: dict[tuple[int, int], np.ndarray] = {}
    # Rows index the left H-label and columns the right H-label.
    right_minus_left = subtraction.T
    for left in range(prime):
        for right in range(left + 1, prime):
            matrices[(left, right)] = connections[
                (right - left) % prime,
                right_minus_left,
            ]
    return matrices


def canonical_matching_coloring(
    connections: np.ndarray,
    subtraction: np.ndarray,
    *,
    trials: int,
    seed: int,
) -> dict:
    """Build arbitrary cross-layer color permutations with HiGHS LPs."""
    connections = np.asarray(connections, dtype=bool)
    prime, source_order = connections.shape
    if trials < 1:
        raise ValueError("matching trials must be positive")
    matrices = canonical_conflict_matrices(
        connections,
        subtraction,
    )
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    attempts: list[dict] = []
    best_completed = -1
    best_history = None
    for trial in range(trials):
        order = rng.permutation(
            np.arange(1, prime, dtype=np.int64)
        ).astype(int).tolist()
        colors, history = sequential_coloring(
            matrices,
            order,
            source_order,
            rng,
        )
        completed = sum(
            bool(record.get("success"))
            for record in history
        )
        if completed > best_completed:
            best_completed = completed
            best_history = history
        attempts.append(
            {
                "trial": trial,
                "layer_order": order,
                "completed_nonzero_layers": completed,
                "success": colors is not None,
                "assignment_history": history,
            }
        )
        if colors is None:
            continue
        exact_conflicts: list[list[int]] = []
        for color_index, color in enumerate(colors):
            for left_index in range(len(color)):
                for right_index in range(left_index + 1, len(color)):
                    left_layer, left_vertex = color[left_index]
                    right_layer, right_vertex = color[right_index]
                    if left_layer > right_layer:
                        left_layer, right_layer = right_layer, left_layer
                        left_vertex, right_vertex = (
                            right_vertex,
                            left_vertex,
                        )
                    difference = int(
                        subtraction[right_vertex, left_vertex]
                    )
                    if connections[
                        (right_layer - left_layer) % prime,
                        difference,
                    ]:
                        exact_conflicts.append(
                            [
                                color_index,
                                left_layer,
                                left_vertex,
                                right_layer,
                                right_vertex,
                            ]
                        )
        if exact_conflicts:
            raise AssertionError(
                "sequential HiGHS coloring fails exact verification"
            )
        return {
            "status": "FEASIBLE",
            "feasible": True,
            "proven_infeasible": False,
            "phases": None,
            "colors": [
                [list(pair) for pair in color] for color in colors
            ],
            "exact_conflicts": 0,
            "trial": trial,
            "completed_nonzero_layers": prime - 1,
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started,
            "solver": (
                "successive totally-unimodular canonical-layer "
                "assignment LPs with open HiGHS"
            ),
        }
    return {
        "status": "HEURISTIC",
        "feasible": False,
        "proven_infeasible": False,
        "phases": None,
        "colors": None,
        "exact_conflicts": None,
        "trial": None,
        "completed_nonzero_layers": best_completed,
        "best_assignment_history": best_history,
        "attempts": attempts,
        "elapsed_seconds": time.perf_counter() - started,
        "solver": (
            "successive totally-unimodular canonical-layer "
            "assignment LPs with open HiGHS"
        ),
    }


def solve_phase_highs_cegar(
    connections: np.ndarray,
    subtraction: np.ndarray,
    *,
    time_limit: float,
    max_rounds: int,
    seed: int,
) -> dict:
    """Solve the phase CSP by a lazy binary HiGHS MIP."""
    connections = np.asarray(connections, dtype=bool)
    subtraction = np.asarray(subtraction, dtype=np.int64)
    prime, source_order = connections.shape
    if (
        prime < 2
        or subtraction.shape != (source_order, source_order)
        or time_limit <= 0
        or max_rounds < 1
    ):
        raise ValueError("invalid phase CEGAR instance")
    if connections[0, 0]:
        return {
            "status": "LOOP",
            "feasible": False,
            "proven_infeasible": True,
            "phases": None,
            "rounds": 0,
            "no_goods": 0,
            "elapsed_seconds": 0.0,
        }

    layer_count = prime - 1
    variable_count = layer_count * source_order

    def variable(layer: int, phase: int) -> int:
        if layer < 1 or layer >= prime:
            raise ValueError("layer zero is fixed and has no binary variable")
        return (layer - 1) * source_order + phase

    upper = np.ones(variable_count, dtype=np.float64)
    for layer in range(1, prime):
        forbidden_phases = np.flatnonzero(connections[layer])
        upper[
            [
                variable(layer, int(phase))
                for phase in forbidden_phases
            ]
        ] = 0.0
        if not np.any(
            upper[
                (layer - 1) * source_order : layer * source_order
            ]
            > 0.5
        ):
            return {
                "status": "PAIR_INFEASIBLE",
                "feasible": False,
                "proven_infeasible": True,
                "phases": None,
                "rounds": 0,
                "no_goods": 0,
                "empty_layer": layer,
                "elapsed_seconds": 0.0,
            }

    rng = np.random.default_rng(seed)
    objective = rng.random(variable_count) * 1e-8
    no_goods: set[tuple[int, int]] = set()
    history: list[dict] = []
    started = time.perf_counter()
    last_phases = None
    for round_number in range(1, max_rounds + 1):
        remaining = time_limit - (time.perf_counter() - started)
        if remaining <= 0:
            break
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        lower: list[float] = []
        upper_rows: list[float] = []
        row = 0
        for layer in range(1, prime):
            for phase in range(source_order):
                row_indices.append(row)
                column_indices.append(variable(layer, phase))
                values.append(1.0)
            lower.append(1.0)
            upper_rows.append(1.0)
            row += 1
        for left_variable, right_variable in sorted(no_goods):
            row_indices.extend([row, row])
            column_indices.extend([left_variable, right_variable])
            values.extend([1.0, 1.0])
            lower.append(-np.inf)
            upper_rows.append(1.0)
            row += 1
        matrix = coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(row, variable_count),
        ).tocsr()
        result = milp(
            objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(
                np.zeros(variable_count, dtype=np.float64),
                upper,
            ),
            constraints=LinearConstraint(
                matrix,
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper_rows, dtype=np.float64),
            ),
            options={
                "time_limit": max(1e-3, remaining),
                "presolve": True,
                "mip_rel_gap": 0.0,
            },
        )
        record = {
            "round": round_number,
            "status_code": int(result.status),
            "message": str(result.message),
            "no_goods": len(no_goods),
            "mip_node_count": (
                int(result.mip_node_count)
                if getattr(result, "mip_node_count", None) is not None
                else None
            ),
        }
        history.append(record)
        if result.status == 2:
            return {
                "status": "INFEASIBLE",
                "feasible": False,
                "proven_infeasible": True,
                "phases": None,
                "rounds": round_number,
                "no_goods": len(no_goods),
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
            }
        if result.x is None:
            return {
                "status": "LIMIT",
                "feasible": False,
                "proven_infeasible": False,
                "phases": None,
                "rounds": round_number,
                "no_goods": len(no_goods),
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
            }
        phases = np.zeros(prime, dtype=np.int64)
        for layer in range(1, prime):
            block = result.x[
                (layer - 1) * source_order : layer * source_order
            ]
            phases[layer] = int(np.argmax(block))
            if block[int(phases[layer])] < 0.5:
                raise RuntimeError("HiGHS incumbent is not integral")
        last_phases = phases
        conflicts = transversal_conflicts(
            phases,
            connections,
            subtraction,
        )
        record["conflicts"] = len(conflicts)
        if not conflicts:
            return {
                "status": "FEASIBLE",
                "feasible": True,
                "proven_infeasible": False,
                "phases": phases.astype(int).tolist(),
                "rounds": round_number,
                "no_goods": len(no_goods),
                "history": history,
                "elapsed_seconds": time.perf_counter() - started,
            }
        added = 0
        for left, right, left_phase, right_phase in conflicts:
            if left == 0:
                upper[variable(right, right_phase)] = 0.0
                added += 1
                continue
            pair = tuple(
                sorted(
                    (
                        variable(left, left_phase),
                        variable(right, right_phase),
                    )
                )
            )
            if pair not in no_goods:
                no_goods.add(pair)
                added += 1
        record["added_no_goods"] = added
        if not added:
            raise AssertionError("conflicting incumbent added no new cut")
    return {
        "status": "LIMIT",
        "feasible": False,
        "proven_infeasible": False,
        "phases": (
            last_phases.astype(int).tolist()
            if last_phases is not None
            else None
        ),
        "rounds": len(history),
        "no_goods": len(no_goods),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def solve_phase_cpsat(
    connections: np.ndarray,
    subtraction: np.ndarray,
    *,
    time_limit: float,
    workers: int,
    seed: int,
) -> dict:
    """Solve the complete phase CSP with compact allowed-pair tables."""
    connections = np.asarray(connections, dtype=bool)
    subtraction = np.asarray(subtraction, dtype=np.int64)
    prime, source_order = connections.shape
    if (
        prime < 2
        or subtraction.shape != (source_order, source_order)
        or time_limit <= 0
        or workers < 1
    ):
        raise ValueError("invalid phase CP-SAT instance")
    started = time.perf_counter()
    if connections[0, 0]:
        return {
            "status": "LOOP",
            "feasible": False,
            "proven_infeasible": True,
            "phases": None,
            "rounds": 0,
            "no_goods": 0,
            "branches": 0,
            "conflicts": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }

    domains: list[np.ndarray] = [
        np.asarray([0], dtype=np.int64)
    ]
    for layer in range(1, prime):
        allowed = np.flatnonzero(~connections[layer]).astype(np.int64)
        if not len(allowed):
            return {
                "status": "PAIR_INFEASIBLE",
                "feasible": False,
                "proven_infeasible": True,
                "phases": None,
                "rounds": 0,
                "no_goods": 0,
                "empty_layer": layer,
                "branches": 0,
                "conflicts": 0,
                "elapsed_seconds": time.perf_counter() - started,
            }
        domains.append(allowed)

    model = cp_model.CpModel()
    variables = [model.NewConstant(0)]
    for layer in range(1, prime):
        variables.append(
            model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(
                    domains[layer].astype(int).tolist()
                ),
                f"phase_{layer}",
            )
        )
    table_sizes: list[int] = []
    for left in range(1, prime):
        for right in range(left + 1, prime):
            layer_difference = (right - left) % prime
            allowed_pairs = [
                (int(left_phase), int(right_phase))
                for left_phase in domains[left]
                for right_phase in domains[right]
                if not connections[
                    layer_difference,
                    int(
                        subtraction[
                            int(right_phase),
                            int(left_phase),
                        ]
                    ),
                ]
            ]
            table_sizes.append(len(allowed_pairs))
            if not allowed_pairs:
                return {
                    "status": "PAIR_INFEASIBLE",
                    "feasible": False,
                    "proven_infeasible": True,
                    "phases": None,
                    "rounds": 0,
                    "no_goods": 0,
                    "empty_layer_pair": [left, right],
                    "allowed_table_sizes": table_sizes,
                    "branches": 0,
                    "conflicts": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            model.AddAllowedAssignments(
                [variables[left], variables[right]],
                allowed_pairs,
            )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    feasible = status_code in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    )
    phases = (
        [int(solver.Value(variable)) for variable in variables]
        if feasible
        else None
    )
    if phases is not None and transversal_conflicts(
        phases,
        connections,
        subtraction,
    ):
        raise AssertionError("CP-SAT phase incumbent fails exact verification")
    return {
        "status": status,
        "feasible": feasible,
        "proven_infeasible": status_code == cp_model.INFEASIBLE,
        "phases": phases,
        "rounds": 1,
        "no_goods": 0,
        "allowed_table_sizes": table_sizes,
        "branches": int(solver.NumBranches()),
        "conflicts": int(solver.NumConflicts()),
        "wall_time_seconds": float(solver.WallTime()),
        "elapsed_seconds": time.perf_counter() - started,
        "solver": "OR-Tools CP-SAT complete allowed-pair phase CSP",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--prime", type=int, default=11)
    parser.add_argument("--search-characters", type=int, default=24)
    parser.add_argument(
        "--character-samples",
        type=int,
        default=0,
        help=(
            "sample this many projective characters instead of exhaustive "
            "enumeration; zero keeps the exhaustive mode"
        ),
    )
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--max-rounds", type=int, default=400)
    parser.add_argument(
        "--solver",
        choices=("highs", "cpsat", "heuristic", "matching"),
        default="highs",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--heuristic-restarts", type=int, default=40)
    parser.add_argument("--heuristic-sweeps", type=int, default=24)
    parser.add_argument("--matching-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3361101)
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.prime < 2
        or not isprime(args.prime)
        or args.search_characters < 0
        or args.character_samples < 0
        or args.time_limit <= 0
        or args.max_rounds < 1
        or args.workers < 1
        or args.heuristic_restarts < 1
        or args.heuristic_sweeps < 1
        or args.matching_trials < 1
    ):
        parser.error("invalid group-transversal lift budget")

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
        metric["best"]["parameters"],
        dtype=np.float64,
    )
    evaluation = source_evaluator.evaluate(
        parameters,
        with_witnesses=True,
    )
    basis = evaluation.basis
    diameter = evaluation.diameter
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    source_order = abs(
        int(Matrix(source_kernel.astype(int).tolist()).det())
    )
    if math.gcd(source_order, args.prime) != 1:
        parser.error("--prime must be coprime to the source index")

    (
        quotient_indices,
        inverse_indices,
        layer_coordinates,
        source_representatives,
    ) = split_forbidden_coordinates(
        source_kernel,
        forbidden,
        args.prime,
    )
    _, _, _, subtraction = quotient_group_tables(source_kernel)
    total_projective_characters = (
        args.prime ** len(source_kernel) - 1
    ) // (args.prime - 1)
    if args.character_samples:
        characters = sampled_projective_forms(
            len(source_kernel),
            args.prime,
            args.character_samples,
            args.seed,
        )
    else:
        characters = projective_forms(
            len(source_kernel),
            args.prime,
        )
    source_loop_mask = quotient_indices == 0
    source_conflicts = layer_coordinates[source_loop_mask]
    good_mask = np.all(
        (source_conflicts @ characters.T) % args.prime != 0,
        axis=0,
    )
    good_characters = characters[good_mask]

    started = time.perf_counter()
    profile_records: list[dict] = []
    pair_viable_count = 0
    viable: list[
        tuple[tuple[int, int, int], int, np.ndarray]
    ] = []
    batch_size = 128
    for start in range(0, len(good_characters), batch_size):
        batch = good_characters[start : start + batch_size]
        layer_batch = np.remainder(
            layer_coordinates @ batch.T,
            args.prime,
        )
        for offset, character in enumerate(batch):
            layers = layer_batch[:, offset]
            connections = connection_table(
                layers,
                quotient_indices,
                inverse_indices,
                args.prime,
                source_order,
            )
            if connections[0, 0]:
                raise AssertionError("loop-free character retained a loop")
            forbidden_counts = connections.sum(axis=1).astype(int)
            allowed_counts = (
                source_order - forbidden_counts[1:]
            ).astype(int)
            minimum_allowed = int(allowed_counts.min())
            total_allowed = int(allowed_counts.sum())
            pair_viable = minimum_allowed > 0
            phase_pair_counts: list[int] = []
            minimum_phase_pair_count = 0
            if pair_viable:
                domains = [
                    np.asarray([0], dtype=np.int64),
                    *[
                        np.flatnonzero(~connections[layer]).astype(
                            np.int64
                        )
                        for layer in range(1, args.prime)
                    ],
                ]
                for left in range(1, args.prime):
                    for right in range(left + 1, args.prime):
                        differences = subtraction[
                            np.ix_(domains[right], domains[left])
                        ]
                        allowed_pair_count = int(
                            np.count_nonzero(
                                ~connections[
                                    (right - left) % args.prime,
                                    differences,
                                ]
                            )
                        )
                        phase_pair_counts.append(allowed_pair_count)
                minimum_phase_pair_count = min(phase_pair_counts)
                pair_viable_count += 1
            triple_viable = (
                pair_viable and minimum_phase_pair_count > 0
            )
            index = start + offset
            profile_records.append(
                {
                    "character_index": int(index),
                    "character": character.astype(int).tolist(),
                    "connection_count": int(connections.sum()),
                    "forbidden_counts_by_layer": (
                        forbidden_counts.astype(int).tolist()
                    ),
                    "allowed_counts_nonzero_layers": (
                        allowed_counts.astype(int).tolist()
                    ),
                    "minimum_allowed_difference_count": minimum_allowed,
                    "total_allowed_difference_count": total_allowed,
                    "pair_viable": pair_viable,
                    "allowed_phase_pair_counts": phase_pair_counts,
                    "minimum_allowed_phase_pair_count": (
                        minimum_phase_pair_count
                    ),
                    "triple_viable": triple_viable,
                }
            )
            if triple_viable:
                viable.append(
                    (
                        (
                            minimum_phase_pair_count,
                            minimum_allowed,
                            total_allowed,
                        ),
                        int(index),
                        character.copy(),
                    )
                )
    viable.sort(key=lambda item: item[0], reverse=True)

    payload: dict = {
        "method": (
            "coprime finite-abelian group-transversal lift with "
            "exact difference profiles and lazy HiGHS phase MIP"
        ),
        "source_metric": str(args.metric),
        "dimension": len(source_kernel),
        "source_index": source_order,
        "prime": args.prime,
        "period_index": source_order * args.prime,
        "target_colors": source_order,
        "parent_min_ratio": evaluation.min_ratio,
        "parent_diameter": diameter,
        "forbidden_projective_pairs": len(forbidden),
        "source_kernel_conflicts": int(source_loop_mask.sum()),
        "projective_characters": len(characters),
        "total_projective_characters": total_projective_characters,
        "exhaustive_characters": not args.character_samples,
        "loop_free_characters": int(good_mask.sum()),
        "pair_viable_characters": pair_viable_count,
        "triple_viable_characters": len(viable),
        "settings": {
            "search_characters": args.search_characters,
            "character_samples": args.character_samples,
            "time_limit_per_character": args.time_limit,
            "max_rounds": args.max_rounds,
            "solver": args.solver,
            "workers": args.workers,
            "heuristic_restarts": args.heuristic_restarts,
            "heuristic_sweeps": args.heuristic_sweeps,
            "matching_trials": args.matching_trials,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_h_norm": args.max_h_norm,
        },
        "profiles": profile_records,
        "searches": [],
        "coloring": None,
        "valid_numerical_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"source={source_order} p={args.prime} "
        f"characters={len(characters)} loop-free={len(good_characters)} "
        f"pair-viable={pair_viable_count} "
        f"triple-viable={len(viable)}",
        flush=True,
    )
    save()
    for rank, (_, character_index, character) in enumerate(
        viable[: args.search_characters],
        start=1,
    ):
        layers = np.remainder(
            layer_coordinates @ character,
            args.prime,
        )
        connections = connection_table(
            layers,
            quotient_indices,
            inverse_indices,
            args.prime,
            source_order,
        )
        profile = profile_records[character_index]
        print(
            f"[{rank}/{min(args.search_characters, len(viable))}] "
            f"char={character_index} "
            f"min-allowed={profile['minimum_allowed_difference_count']} "
            f"min-pairs={profile['minimum_allowed_phase_pair_count']} "
            f"total-allowed={profile['total_allowed_difference_count']}",
            flush=True,
        )
        if args.solver == "highs":
            solve = solve_phase_highs_cegar(
                connections,
                subtraction,
                time_limit=args.time_limit,
                max_rounds=args.max_rounds,
                seed=args.seed + 1_000_003 * rank,
            )
        elif args.solver == "cpsat":
            solve = solve_phase_cpsat(
                connections,
                subtraction,
                time_limit=args.time_limit,
                workers=args.workers,
                seed=args.seed + 1_000_003 * rank,
            )
        elif args.solver == "heuristic":
            solve = phase_coordinate_descent(
                connections,
                subtraction,
                restarts=args.heuristic_restarts,
                sweeps=args.heuristic_sweeps,
                seed=args.seed + 1_000_003 * rank,
            )
        else:
            solve = canonical_matching_coloring(
                connections,
                subtraction,
                trials=args.matching_trials,
                seed=args.seed + 1_000_003 * rank,
            )
        record = {
            "rank": rank,
            "character_index": character_index,
            "character": character.astype(int).tolist(),
            "profile": profile,
            "solve": solve,
        }
        payload["searches"].append(record)
        print(
            f"  status={solve['status']} "
            f"rounds={solve.get('rounds')} "
            f"cuts={solve.get('no_goods')} "
            f"branches={solve.get('branches')}",
            flush=True,
        )
        if not solve["feasible"]:
            save()
            continue

        lift = hnf_columns(
            kernel_basis(
                [np.asarray(character, dtype=np.int64)],
                [args.prime],
                len(source_kernel),
            )
        )
        period = hnf_columns(source_kernel @ lift)
        complement_scalar = args.prime * pow(
            args.prime,
            -1,
            source_order,
        )
        pivot = next(
            index
            for index, value in enumerate(character)
            if int(value) % args.prime
        )
        lift_coordinate = np.zeros(len(character), dtype=np.int64)
        lift_coordinate[pivot] = pow(
            int(character[pivot]),
            -1,
            args.prime,
        )
        layer_translation = source_kernel @ lift_coordinate
        if solve.get("colors") is None:
            phase_indices = np.asarray(
                solve["phases"],
                dtype=np.int64,
            )
            conflicts = transversal_conflicts(
                phase_indices,
                connections,
                subtraction,
            )
            if conflicts:
                raise AssertionError(
                    "decoded phase block still conflicts"
                )
            block = [
                (
                    complement_scalar
                    * source_representatives[
                        int(phase_indices[layer])
                    ]
                    + layer * layer_translation
                )
                .astype(int)
                .tolist()
                for layer in range(args.prime)
            ]
            coloring_data = {
                "phase_quotient_indices": (
                    phase_indices.astype(int).tolist()
                ),
                "block_coordinate_representatives": block,
            }
        else:
            finite_colors = solve["colors"]
            coordinate_colors = [
                [
                    (
                        complement_scalar
                        * source_representatives[int(vertex)]
                        + int(layer) * layer_translation
                    )
                    .astype(int)
                    .tolist()
                    for layer, vertex in color
                ]
                for color in finite_colors
            ]
            coloring_data = {
                "canonical_layer_color_labels": finite_colors,
                "color_coordinate_representatives": coordinate_colors,
            }
        payload["coloring"] = {
            "character_index": character_index,
            "character": character.astype(int).tolist(),
            "period_basis_columns": period.astype(int).tolist(),
            "period_determinant": source_order * args.prime,
            "color_count": source_order,
            "vertices_per_color": args.prime,
            **coloring_data,
            "exact_forbidden_conflicts": 0,
        }
        payload["valid_numerical_witness"] = True
        save()
        print("*** GROUP-TRANSVERSAL COLORING FOUND ***", flush=True)
        return 0
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
