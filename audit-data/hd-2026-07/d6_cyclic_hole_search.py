"""Joint search for a cyclic period and a missing compatible difference.

Previous period portfolios fixed an elementary-primary quotient and then
asked whether its conflict graph happened to omit useful differences.  This
file reverses the order.  For a cyclic quotient ``Z/N`` it chooses a desired
missing residue ``y`` and searches directly for a primitive row ``a`` such
that

    <a,f> != 0, +y, -y  (mod N)

for every forbidden displacement ``f``.  The first exclusion makes the period
valid; the other two make every translate of ``{0,y}`` an independent pair.
For ``343 < N <= 684`` a large enough matching in these compatible cycles
immediately gives a 342-coloring.

Multiplication by a unit of ``Z/N`` is a quotient automorphism, so only
``gcd(y,N)`` matters.  The campaign therefore tests divisor representatives
instead of every residue.  A discrete coordinate descent handles the modular
avoidance problem, and HiGHS verifies/constructs the final maximum matching.
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

import combigeo
from d6_torus_period_portfolio import (
    quotient_matching_coloring,
    signed_connection_images,
)
from determinant_repair import exact_det, load_preset
from metric_deform import exhaustive_covering_radius
from prime_radon import hnf_columns, kernel_basis, smith_diagonal
from prime_row_opt import _forbidden_with_weights


def compatible_cycle_matching_size(modulus: int, difference: int) -> int:
    """Maximum matching using only translates of one cyclic difference."""
    divisor = math.gcd(int(modulus), int(difference))
    cycle_order = int(modulus) // divisor
    return divisor * (cycle_order // 2)


def divisor_targets(
    modulus: int,
    target_colors: int,
) -> list[int]:
    """One useful residue representative for every unit orbit."""
    required_pairs = max(0, int(modulus) - int(target_colors))
    targets = [
        divisor
        for divisor in range(1, modulus)
        if modulus % divisor == 0
        and compatible_cycle_matching_size(modulus, divisor)
        >= required_pairs
    ]
    return sorted(
        targets,
        key=lambda value: (
            2 if (2 * value) % modulus == 0 else 3,
            -compatible_cycle_matching_size(modulus, value),
            value,
        ),
    )


def target_violation_mask(
    forbidden: np.ndarray,
    row: Sequence[int],
    modulus: int,
    target: int,
) -> np.ndarray:
    """Forbidden vectors mapped to ``{0,+target,-target}``."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    row = np.asarray(row, dtype=np.int64)
    residues = np.remainder(forbidden @ row, modulus)
    inverse_target = (-int(target)) % modulus
    return (
        (residues == 0)
        | (residues == int(target) % modulus)
        | (residues == inverse_target)
    )


def coordinate_scores(
    forbidden: np.ndarray,
    row: np.ndarray,
    coordinate: int,
    modulus: int,
    target: int,
) -> np.ndarray:
    """Exact violation counts for every value of one row coordinate."""
    current = np.remainder(forbidden @ row, modulus)
    coefficient = np.remainder(forbidden[:, coordinate], modulus)
    base = np.remainder(
        current - coefficient * int(row[coordinate]),
        modulus,
    )
    values = np.remainder(
        base[:, None]
        + coefficient[:, None]
        * np.arange(modulus, dtype=np.int64)[None, :],
        modulus,
    )
    inverse_target = (-int(target)) % modulus
    return np.count_nonzero(
        (values == 0)
        | (values == int(target) % modulus)
        | (values == inverse_target),
        axis=0,
    ).astype(np.int64)


def coordinate_weighted_scores(
    forbidden: np.ndarray,
    row: np.ndarray,
    coordinate: int,
    modulus: int,
    target: int,
    weights: Sequence[float],
) -> np.ndarray:
    """Weighted violation sum for every value of one row coordinate."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.shape != (len(forbidden),):
        raise ValueError("one violation weight is required per vector")
    current = np.remainder(forbidden @ row, modulus)
    coefficient = np.remainder(forbidden[:, coordinate], modulus)
    base = np.remainder(
        current - coefficient * int(row[coordinate]),
        modulus,
    )
    values = np.remainder(
        base[:, None]
        + coefficient[:, None]
        * np.arange(modulus, dtype=np.int64)[None, :],
        modulus,
    )
    inverse_target = (-int(target)) % modulus
    violations = (
        (values == 0)
        | (values == int(target) % modulus)
        | (values == inverse_target)
    )
    return weights_array @ violations


def weighted_violation_score(
    forbidden: np.ndarray,
    row: Sequence[int],
    modulus: int,
    target: int,
    weights: Sequence[float],
) -> float:
    mask = target_violation_mask(
        forbidden,
        row,
        modulus,
        target,
    )
    return float(np.asarray(weights, dtype=np.float64)[mask].sum())


def primitive_cyclic_row(row: Sequence[int], modulus: int) -> bool:
    common = int(modulus)
    for value in row:
        common = math.gcd(common, int(value))
    return common == 1


def pair_coordinate_escape(
    forbidden: np.ndarray,
    row: np.ndarray,
    modulus: int,
    target: int,
    *,
    pair_top: int,
    pair_trials: int,
    rng: np.random.Generator,
    violation_weights: Sequence[float] | None = None,
    weighted_first: bool = False,
) -> tuple[np.ndarray, int, float]:
    """Try Cartesian products of promising values for coordinate pairs."""
    if pair_top < 2 or pair_trials < 1:
        score = int(
            target_violation_mask(
                forbidden,
                row,
                modulus,
                target,
            ).sum()
        )
        weighted = (
            weighted_violation_score(
                forbidden,
                row,
                modulus,
                target,
                violation_weights,
            )
            if violation_weights is not None
            else float(score)
        )
        return row.copy(), score, weighted
    pairs = [
        (left, right)
        for left in range(len(row))
        for right in range(left + 1, len(row))
    ]
    rng.shuffle(pairs)
    best_row = row.copy()
    best_score = int(
        target_violation_mask(
            forbidden,
            row,
            modulus,
            target,
        ).sum()
    )
    best_weighted = (
        weighted_violation_score(
            forbidden,
            row,
            modulus,
            target,
            violation_weights,
        )
        if violation_weights is not None
        else float(best_score)
    )
    score_cache: dict[int, np.ndarray] = {}
    for left, right in pairs[: min(pair_trials, len(pairs))]:
        for coordinate in (left, right):
            if coordinate not in score_cache:
                score_cache[coordinate] = coordinate_scores(
                    forbidden,
                    row,
                    coordinate,
                    modulus,
                    target,
                )
        request = min(pair_top, modulus)
        left_values = np.argpartition(
            score_cache[left],
            request - 1,
        )[:request]
        right_values = np.argpartition(
            score_cache[right],
            request - 1,
        )[:request]
        variants = np.repeat(
            row[None, :],
            request * request,
            axis=0,
        )
        variants[:, left] = np.repeat(left_values, request)
        variants[:, right] = np.tile(right_values, request)
        primitive = np.asarray(
            [
                primitive_cyclic_row(candidate, modulus)
                for candidate in variants
            ],
            dtype=bool,
        )
        if not np.any(primitive):
            continue
        residues = np.remainder(forbidden @ variants.T, modulus)
        inverse_target = (-int(target)) % modulus
        scores = np.count_nonzero(
            (residues == 0)
            | (residues == int(target) % modulus)
            | (residues == inverse_target),
            axis=0,
        )
        scores[~primitive] = len(forbidden) + 1
        if violation_weights is not None:
            violations = (
                (residues == 0)
                | (residues == int(target) % modulus)
                | (residues == inverse_target)
            )
            weighted_scores = (
                np.asarray(violation_weights, dtype=np.float64)
                @ violations
            )
            weighted_scores[~primitive] = float("inf")
        else:
            weighted_scores = scores.astype(np.float64)
        candidate_index = int(
            (
                np.lexsort((scores, weighted_scores))
                if weighted_first
                else np.lexsort((weighted_scores, scores))
            )[0]
        )
        candidate_score = int(scores[candidate_index])
        candidate_weighted = float(weighted_scores[candidate_index])
        candidate_key = (
            (candidate_weighted, candidate_score)
            if weighted_first
            else (candidate_score, candidate_weighted)
        )
        best_key = (
            (best_weighted, best_score)
            if weighted_first
            else (best_score, best_weighted)
        )
        if candidate_key < best_key:
            best_score = candidate_score
            best_weighted = candidate_weighted
            best_row = variants[candidate_index].copy()
            if best_score == 0:
                break
    return best_row, best_score, best_weighted


def cyclic_target_descent(
    forbidden: np.ndarray,
    modulus: int,
    target: int,
    *,
    restarts: int = 100,
    sweeps: int = 20,
    top: int = 12,
    pair_top: int = 0,
    pair_trials: int = 0,
    initial_row: Sequence[int] | None = None,
    violation_weights: Sequence[float] | None = None,
    weighted_first: bool = False,
    seed: int = 0,
) -> dict:
    """Search a primitive cyclic row avoiding ``0,+target,-target``."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    if forbidden.ndim != 2:
        raise ValueError("forbidden catalogue must be a matrix")
    if modulus < 2 or not 0 < target < modulus:
        raise ValueError("invalid modulus or target residue")
    if (
        restarts < 1
        or sweeps < 1
        or top < 1
        or pair_top < 0
        or pair_trials < 0
    ):
        raise ValueError("search budgets must be positive")
    if initial_row is not None and len(initial_row) != forbidden.shape[1]:
        raise ValueError("initial row has the wrong dimension")
    if violation_weights is not None and np.asarray(
        violation_weights
    ).shape != (len(forbidden),):
        raise ValueError("one violation weight is required per vector")
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    best_score = len(forbidden) + 1
    best_row: np.ndarray | None = None
    best_weighted = float("inf")
    best_restart = -1
    best_sweep = -1

    def state_key(count: int, weighted: float) -> tuple[float, float]:
        return (
            (float(weighted), float(count))
            if weighted_first
            else (float(count), float(weighted))
        )

    for restart in range(restarts):
        if restart == 0 and initial_row is not None:
            row = np.remainder(
                np.asarray(initial_row, dtype=np.int64),
                modulus,
            )
        else:
            row = rng.integers(
                0,
                modulus,
                size=forbidden.shape[1],
                dtype=np.int64,
            )
            # A unit anchor makes surjectivity overwhelmingly robust without
            # excluding rows whose best representative has another unit pivot.
            anchor = int(rng.integers(forbidden.shape[1]))
            row[anchor] = 1
        current_score = int(
            target_violation_mask(
                forbidden,
                row,
                modulus,
                target,
            ).sum()
        )
        current_weighted = (
            weighted_violation_score(
                forbidden,
                row,
                modulus,
                target,
                violation_weights,
            )
            if violation_weights is not None
            else float(current_score)
        )
        for sweep in range(sweeps):
            improved = False
            for coordinate in rng.permutation(forbidden.shape[1]):
                scores = coordinate_scores(
                    forbidden,
                    row,
                    int(coordinate),
                    modulus,
                    target,
                )
                weighted_scores = (
                    coordinate_weighted_scores(
                        forbidden,
                        row,
                        int(coordinate),
                        modulus,
                        target,
                        violation_weights,
                    )
                    if violation_weights is not None
                    else scores.astype(np.float64)
                )
                order = (
                    np.lexsort((scores, weighted_scores))
                    if weighted_first
                    else np.lexsort((weighted_scores, scores))
                )
                request = min(int(top), modulus)
                candidates = order[:request]
                if weighted_first:
                    best_local_weight = float(
                        weighted_scores[candidates[0]]
                    )
                    tolerance = max(
                        1e-12,
                        0.02 * abs(best_local_weight),
                    )
                    near_best = candidates[
                        weighted_scores[candidates]
                        <= best_local_weight + tolerance
                    ]
                else:
                    best_count = int(scores[candidates[0]])
                    same_count = candidates[
                        scores[candidates] == best_count
                    ]
                    best_local_weight = float(
                        weighted_scores[same_count].min()
                    )
                    tolerance = max(
                        1e-12,
                        0.02 * abs(best_local_weight),
                    )
                    near_best = same_count[
                        weighted_scores[same_count]
                        <= best_local_weight + tolerance
                    ]
                chosen = int(
                    near_best[int(rng.integers(len(near_best)))]
                )
                chosen_key = (
                    int(scores[chosen]),
                    float(weighted_scores[chosen]),
                )
                if state_key(*chosen_key) <= state_key(
                    current_score,
                    current_weighted,
                ):
                    improved |= (
                        state_key(*chosen_key)
                        < state_key(current_score, current_weighted)
                        or chosen != int(row[coordinate])
                    )
                    row[coordinate] = chosen
                    current_score, current_weighted = chosen_key

            if current_score and pair_top >= 2 and pair_trials:
                pair_row, pair_score, pair_weighted = pair_coordinate_escape(
                    forbidden,
                    row,
                    modulus,
                    target,
                    pair_top=pair_top,
                    pair_trials=pair_trials,
                    rng=rng,
                    violation_weights=violation_weights,
                    weighted_first=weighted_first,
                )
                if state_key(
                    pair_score,
                    pair_weighted,
                ) <= state_key(current_score, current_weighted):
                    improved |= state_key(
                        pair_score,
                        pair_weighted,
                    ) < state_key(current_score, current_weighted)
                    row = pair_row
                    current_score = pair_score
                    current_weighted = pair_weighted

            if (
                state_key(current_score, current_weighted)
                < state_key(best_score, best_weighted)
                and primitive_cyclic_row(row, modulus)
            ):
                best_score = current_score
                best_weighted = current_weighted
                best_row = row.copy()
                best_restart = restart
                best_sweep = sweep
            if current_score == 0 and primitive_cyclic_row(row, modulus):
                return {
                    "success": True,
                    "score": 0,
                    "weighted_score": 0.0,
                    "row": row.astype(int).tolist(),
                    "restart": restart,
                    "sweep": sweep,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            if not improved:
                kick = int(rng.integers(forbidden.shape[1]))
                row[kick] = int(rng.integers(modulus))
                current_score = int(
                    target_violation_mask(
                        forbidden,
                        row,
                        modulus,
                        target,
                    ).sum()
                )
                current_weighted = (
                    weighted_violation_score(
                        forbidden,
                        row,
                        modulus,
                        target,
                        violation_weights,
                    )
                    if violation_weights is not None
                    else float(current_score)
                )

    return {
        "success": False,
        "score": int(best_score),
        "weighted_score": (
            float(best_weighted) if np.isfinite(best_weighted) else None
        ),
        "row": best_row.astype(int).tolist() if best_row is not None else None,
        "restart": best_restart,
        "sweep": best_sweep,
        "elapsed_seconds": time.perf_counter() - started,
    }


def prime_divisors(value: int) -> list[int]:
    """Distinct prime divisors of a positive integer."""
    residual = int(value)
    result: list[int] = []
    divisor = 2
    while divisor * divisor <= residual:
        if residual % divisor == 0:
            result.append(divisor)
            while residual % divisor == 0:
                residual //= divisor
        divisor += 1
    if residual > 1:
        result.append(residual)
    return result


def cyclic_target_cpsat(
    forbidden: np.ndarray,
    modulus: int,
    target: int,
    *,
    time_limit: float = 60.0,
    workers: int = 8,
    hint: Sequence[int] | None = None,
) -> dict:
    """Exact CP-SAT feasibility model for a primitive avoiding row."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    if forbidden.ndim != 2:
        raise ValueError("forbidden catalogue must be a matrix")
    if modulus < 2 or not 0 < target < modulus:
        raise ValueError("invalid modulus or target")
    if time_limit <= 0 or workers < 1:
        raise ValueError("time limit and workers must be positive")
    if hint is not None and len(hint) != forbidden.shape[1]:
        raise ValueError("hint has the wrong dimension")

    model = cp_model.CpModel()
    variables = [
        model.NewIntVar(0, modulus - 1, f"a_{coordinate}")
        for coordinate in range(forbidden.shape[1])
    ]
    forbidden_residues = {
        0,
        int(target) % modulus,
        (-int(target)) % modulus,
    }
    for vector_index, vector in enumerate(forbidden):
        residue = model.NewIntVar(
            0,
            modulus - 1,
            f"r_{vector_index}",
        )
        expression = sum(
            int(coefficient % modulus) * variables[coordinate]
            for coordinate, coefficient in enumerate(vector)
        )
        model.AddModuloEquality(residue, expression, modulus)
        for forbidden_residue in forbidden_residues:
            model.Add(residue != forbidden_residue)

    # A cyclic row is surjective precisely when its entries together with N
    # have gcd one.  For every prime q|N, at least one entry must be nonzero
    # modulo q.
    for prime in prime_divisors(modulus):
        nonzero: list[cp_model.IntVar] = []
        for coordinate, variable in enumerate(variables):
            residue = model.NewIntVar(
                0,
                prime - 1,
                f"a_{coordinate}_mod_{prime}",
            )
            model.AddModuloEquality(residue, variable, prime)
            indicator = model.NewBoolVar(
                f"a_{coordinate}_nonzero_mod_{prime}"
            )
            model.Add(residue != 0).OnlyEnforceIf(indicator)
            model.Add(residue == 0).OnlyEnforceIf(indicator.Not())
            nonzero.append(indicator)
        model.Add(sum(nonzero) >= 1)

    if hint is not None:
        for variable, value in zip(variables, hint):
            model.AddHint(variable, int(value) % modulus)
    model.AddDecisionStrategy(
        variables,
        cp_model.CHOOSE_MIN_DOMAIN_SIZE,
        cp_model.SELECT_MIN_VALUE,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = False
    started = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - started
    feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    row = (
        [int(solver.Value(variable)) for variable in variables]
        if feasible
        else None
    )
    if feasible and (
        not primitive_cyclic_row(row, modulus)
        or np.any(
            target_violation_mask(
                forbidden,
                row,
                modulus,
                target,
            )
        )
    ):
        raise AssertionError("CP-SAT returned an invalid cyclic row")
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
        "row": row,
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
        "wall_time": float(solver.WallTime()),
        "elapsed_seconds": elapsed,
        "solver": "OR-Tools CP-SAT exact modular feasibility model",
    }


def cyclic_target_highs_interval_mip(
    forbidden: np.ndarray,
    modulus: int,
    target: int,
    *,
    time_limit: float = 60.0,
    fixed_row: Sequence[int] | None = None,
    free_coordinates: Sequence[int] | None = None,
    avoid_radius: int = 1,
) -> dict:
    """Exact HiGHS MIP for the interval case ``target = +/-1``.

    Avoiding residues ``0,+1,-1`` leaves the single interval
    ``[2, modulus-2]``.  Hence every modular condition has the linear form

        2 <= forbidden[j] @ row - modulus*q[j] <= modulus-2.

    Surjectivity is encoded exactly, prime by prime: for every prime divisor
    of the modulus, at least one row coordinate must be nonzero modulo that
    prime.
    """
    forbidden = np.asarray(forbidden, dtype=np.int64)
    modulus = int(modulus)
    target = int(target) % modulus
    if forbidden.ndim != 2:
        raise ValueError("forbidden catalogue must be a matrix")
    avoid_radius = int(avoid_radius)
    if (
        modulus < 4
        or target not in {1, modulus - 1}
        or avoid_radius < 1
        or 2 * avoid_radius + 1 >= modulus
    ):
        raise ValueError(
            "HiGHS interval formulation requires target +/-1 and a "
            "nonempty allowed residue interval"
        )
    if time_limit <= 0:
        raise ValueError("time limit must be positive")

    dimension = forbidden.shape[1]
    allowed_lower = avoid_radius + 1
    allowed_upper = modulus - avoid_radius - 1
    lower_bounds: list[float] = [0.0] * dimension
    upper_bounds: list[float] = [float(modulus - 1)] * dimension
    fixed_coordinates: list[int] = []
    if fixed_row is not None:
        if len(fixed_row) != dimension:
            raise ValueError("fixed row has the wrong dimension")
        free_set = (
            set(range(dimension))
            if free_coordinates is None
            else {int(value) for value in free_coordinates}
        )
        if any(value < 0 or value >= dimension for value in free_set):
            raise ValueError("free coordinate lies outside the row")
        fixed_coordinates = [
            coordinate
            for coordinate in range(dimension)
            if coordinate not in free_set
        ]
        for coordinate in fixed_coordinates:
            value = int(fixed_row[coordinate]) % modulus
            lower_bounds[coordinate] = float(value)
            upper_bounds[coordinate] = float(value)
    elif free_coordinates is not None:
        raise ValueError("free coordinates require a fixed row")
    objective: list[float] = [0.0] * dimension
    integrality: list[int] = [1] * dimension

    q_indices: list[int] = []
    for vector in forbidden:
        dot_min = sum(
            min(0, int(coefficient) * (modulus - 1))
            for coefficient in vector
        )
        dot_max = sum(
            max(0, int(coefficient) * (modulus - 1))
            for coefficient in vector
        )
        q_lower = math.floor((dot_min - allowed_upper) / modulus)
        q_upper = math.ceil((dot_max - allowed_lower) / modulus)
        q_indices.append(len(lower_bounds))
        lower_bounds.append(float(q_lower))
        upper_bounds.append(float(q_upper))
        objective.append(0.0)
        integrality.append(1)

    primitive_blocks: list[tuple[int, list[int], list[int], list[int]]] = []
    for prime in prime_divisors(modulus):
        k_indices: list[int] = []
        residue_indices: list[int] = []
        nonzero_indices: list[int] = []
        for _ in range(dimension):
            k_indices.append(len(lower_bounds))
            lower_bounds.append(0.0)
            upper_bounds.append(float((modulus - 1) // prime))
            objective.append(0.0)
            integrality.append(1)

            residue_indices.append(len(lower_bounds))
            lower_bounds.append(0.0)
            upper_bounds.append(float(prime - 1))
            objective.append(0.0)
            integrality.append(1)

            nonzero_indices.append(len(lower_bounds))
            lower_bounds.append(0.0)
            upper_bounds.append(1.0)
            objective.append(0.0)
            integrality.append(1)
        primitive_blocks.append(
            (prime, k_indices, residue_indices, nonzero_indices)
        )

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    constraint_lower: list[float] = []
    constraint_upper: list[float] = []

    def add_constraint(
        entries: Sequence[tuple[int, int | float]],
        lower: float,
        upper: float,
    ) -> None:
        row_index = len(constraint_lower)
        for column, coefficient in entries:
            if coefficient:
                row_indices.append(row_index)
                column_indices.append(int(column))
                coefficients.append(float(coefficient))
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    for vector, q_index in zip(forbidden, q_indices):
        add_constraint(
            [
                *[
                    (coordinate, int(coefficient))
                    for coordinate, coefficient in enumerate(vector)
                ],
                (q_index, -modulus),
            ],
            float(allowed_lower),
            float(allowed_upper),
        )

    for prime, k_indices, residue_indices, nonzero_indices in primitive_blocks:
        for coordinate in range(dimension):
            # row_i = prime*k_i + residue_i.
            add_constraint(
                [
                    (coordinate, 1),
                    (k_indices[coordinate], -prime),
                    (residue_indices[coordinate], -1),
                ],
                0.0,
                0.0,
            )
            # z_i=0 iff residue_i=0; z_i=1 permits 1..prime-1.
            add_constraint(
                [
                    (residue_indices[coordinate], 1),
                    (nonzero_indices[coordinate], -1),
                ],
                0.0,
                np.inf,
            )
            add_constraint(
                [
                    (residue_indices[coordinate], 1),
                    (nonzero_indices[coordinate], -(prime - 1)),
                ],
                -np.inf,
                0.0,
            )
        add_constraint(
            [(index, 1) for index in nonzero_indices],
            1.0,
            np.inf,
        )

    variable_count = len(lower_bounds)
    matrix = coo_matrix(
        (
            np.asarray(coefficients, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(constraint_lower), variable_count),
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        c=np.asarray(objective, dtype=np.float64),
        integrality=np.asarray(integrality, dtype=np.int8),
        bounds=Bounds(
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(constraint_lower, dtype=np.float64),
            np.asarray(constraint_upper, dtype=np.float64),
        ),
        options={
            "time_limit": float(time_limit),
            "presolve": True,
            "mip_rel_gap": 0.0,
        },
    )
    elapsed = time.perf_counter() - started
    row = None
    feasible = result.x is not None
    if feasible:
        row = np.rint(result.x[:dimension]).astype(np.int64) % modulus
        residues = np.remainder(forbidden @ row, modulus)
        feasible = (
            primitive_cyclic_row(row, modulus)
            and bool(
                np.all(
                    (residues >= allowed_lower)
                    & (residues <= allowed_upper)
                )
            )
        )
        if not feasible:
            raise AssertionError(
                "HiGHS returned a numerically invalid interval-MIP incumbent"
            )
    labels = {
        0: "OPTIMAL",
        1: "LIMIT",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "OTHER",
    }
    return {
        "status": labels.get(int(result.status), str(result.status)),
        "status_code": int(result.status),
        "message": str(result.message),
        "feasible": bool(feasible),
        "proven_infeasible": int(result.status) == 2,
        "row": row.astype(int).tolist() if row is not None else None,
        "avoid_radius": avoid_radius,
        "allowed_residue_interval": [allowed_lower, allowed_upper],
        "fixed_coordinates": fixed_coordinates,
        "free_coordinates": [
            coordinate
            for coordinate in range(dimension)
            if coordinate not in fixed_coordinates
        ],
        "variables": variable_count,
        "constraints": len(constraint_lower),
        "mip_node_count": (
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "elapsed_seconds": elapsed,
        "solver": "SciPy milp with open HiGHS exact interval formulation",
    }


def parse_indices(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            values.extend(range(start, stop + step, step))
        else:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("at least one index is required")
    return values


def parse_row(text: str) -> list[int]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list):
        raise argparse.ArgumentTypeError("initial row must be a JSON list")
    return [int(value) for value in raw]


def load_metric_checkpoint(
    path: Path,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, float, dict]:
    """Load and independently recheck a saved deformed parent metric."""
    try:
        payload = json.loads(path.read_text())
        basis = np.asarray(payload["best"]["basis"], dtype=np.float64)
        recorded_diameter = float(payload["best"]["diameter"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid metric checkpoint {path}: {error}") from error
    if basis.shape != expected_shape or not np.all(np.isfinite(basis)):
        raise ValueError(
            f"metric checkpoint basis has shape {basis.shape}, "
            f"expected {expected_shape}"
        )
    if (
        not np.isfinite(recorded_diameter)
        or recorded_diameter <= 0
        or abs(float(np.linalg.det(basis))) < 1e-12
    ):
        raise ValueError("metric checkpoint has a degenerate basis or diameter")
    facets = combigeo.relevant_facets(basis.tolist())
    radius, _ = exhaustive_covering_radius(facets)
    diameter = 2.0 * radius
    tolerance = max(5e-8, 5e-7 * abs(recorded_diameter))
    if abs(diameter - recorded_diameter) > tolerance:
        raise ValueError(
            "metric checkpoint diameter mismatch: recomputed "
            f"{diameter:.12g}, recorded {recorded_diameter:.12g}"
        )
    return basis, diameter, payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices",
        type=parse_indices,
        default=list(range(684, 343, -1)),
    )
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--restarts", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--pair-top", type=int, default=0)
    parser.add_argument("--pair-trials", type=int, default=0)
    parser.add_argument("--initial-row", type=parse_row)
    parser.add_argument(
        "--highs-time-limit",
        type=float,
        default=0.0,
        help=(
            "run the exact HiGHS interval MIP when the target difference "
            "is congruent to +/-1; zero disables it"
        ),
    )
    parser.add_argument("--cpsat-time-limit", type=float, default=0.0)
    parser.add_argument("--cpsat-workers", type=int, default=8)
    parser.add_argument("--weight-power", type=float, default=0.0)
    parser.add_argument("--weighted-first", action="store_true")
    parser.add_argument(
        "--search-min-ratio",
        type=float,
        default=1.0,
        help=(
            "optimize only fixed-metric conflicts below this ratio; "
            "values below one produce near-feasible metric seeds, not colors"
        ),
    )
    parser.add_argument(
        "--max-targets-per-index",
        type=int,
        default=0,
        help="zero keeps every useful divisor orbit",
    )
    parser.add_argument(
        "--target-differences",
        type=parse_indices,
        help="optional comma/range list restricting divisor representatives",
    )
    parser.add_argument("--matching-time-limit", type=float, default=60.0)
    parser.add_argument(
        "--metric-checkpoint",
        type=Path,
        help=(
            "use the independently rechecked basis and diameter from a "
            "metric-deformation checkpoint for the next discrete search"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        any(index <= args.target_colors for index in args.indices)
        or args.target_colors < 1
        or args.restarts < 1
        or args.sweeps < 1
        or args.top < 1
        or args.pair_top < 0
        or args.pair_trials < 0
        or args.highs_time_limit < 0
        or args.cpsat_time_limit < 0
        or args.cpsat_workers < 1
        or args.weight_power < 0
        or not 0 < args.search_min_ratio <= 1.0
        or args.max_targets_per_index < 0
        or args.matching_time_limit <= 0
    ):
        parser.error("invalid indices, target, budgets, or time limit")

    lattice, basis, diameter, _, _ = load_preset("d6")
    metric_payload = None
    if args.metric_checkpoint is not None:
        try:
            basis, diameter, metric_payload = load_metric_checkpoint(
                args.metric_checkpoint,
                basis.shape,
            )
        except ValueError as error:
            parser.error(str(error))
    forbidden, forbidden_ratios, _ = _forbidden_with_weights(
        basis,
        diameter,
    )
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "joint cyclic-period and missing-difference modular descent, "
            "followed by HiGHS maximum matching"
        ),
        "lattice": lattice,
        "metric_checkpoint": (
            str(args.metric_checkpoint)
            if args.metric_checkpoint is not None
            else None
        ),
        "metric_checkpoint_method": (
            metric_payload.get("method")
            if metric_payload is not None
            else None
        ),
        "dimension": basis.shape[0],
        "parent_basis": basis.tolist(),
        "parent_diameter": diameter,
        "forbidden_projective_pairs": len(forbidden),
        "target_colors": args.target_colors,
        "settings": {
            "indices": args.indices,
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "pair_top": args.pair_top,
            "pair_trials": args.pair_trials,
            "initial_row": args.initial_row,
            "highs_time_limit": args.highs_time_limit,
            "cpsat_time_limit": args.cpsat_time_limit,
            "cpsat_workers": args.cpsat_workers,
            "weight_power": args.weight_power,
            "weighted_first": args.weighted_first,
            "search_min_ratio": args.search_min_ratio,
            "max_targets_per_index": args.max_targets_per_index,
            "target_differences": args.target_differences,
            "matching_time_limit": args.matching_time_limit,
            "seed": args.seed,
        },
        "records": [],
        "coloring": None,
        "valid_combinatorial_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for case_number, modulus in enumerate(args.indices):
        targets = divisor_targets(modulus, args.target_colors)
        if args.target_differences is not None:
            requested = set(args.target_differences)
            targets = [target for target in targets if target in requested]
        if args.max_targets_per_index:
            targets = targets[: args.max_targets_per_index]
        print(
            f"[{case_number+1}/{len(args.indices)}] N={modulus} "
            f"targets={targets}",
            flush=True,
        )
        for target_number, target in enumerate(targets):
            search_mask = (
                forbidden_ratios < args.search_min_ratio - 1e-12
            )
            search_forbidden = forbidden[search_mask]
            search_ratios = forbidden_ratios[search_mask]
            search = cyclic_target_descent(
                search_forbidden,
                modulus,
                target,
                restarts=args.restarts,
                sweeps=args.sweeps,
                top=args.top,
                pair_top=args.pair_top,
                pair_trials=args.pair_trials,
                initial_row=args.initial_row,
                violation_weights=(
                    np.power(
                        np.maximum(0.0, 1.0 - search_ratios),
                        args.weight_power,
                    )
                    if args.weight_power > 0
                    else None
                ),
                weighted_first=args.weighted_first,
                seed=(
                    args.seed
                    + 1_000_003 * case_number
                    + 10_007 * target_number
                ),
            )
            record: dict = {
                "period_index": modulus,
                "target_difference": target,
                "target_difference_order": (
                    modulus // math.gcd(modulus, target)
                ),
                "single_difference_matching_size": (
                    compatible_cycle_matching_size(modulus, target)
                ),
                "required_pairs": modulus - args.target_colors,
                "search_core_ratio": args.search_min_ratio,
                "search_core_vectors": int(len(search_forbidden)),
                "search": search,
            }
            highs_mip = None
            if (
                not search["success"]
                and args.highs_time_limit > 0
                and target % modulus in {1, modulus - 1}
            ):
                highs_mip = cyclic_target_highs_interval_mip(
                    search_forbidden,
                    modulus,
                    target,
                    time_limit=args.highs_time_limit,
                )
                record["highs_mip"] = highs_mip
                if highs_mip["feasible"]:
                    search = {
                        **search,
                        "success": True,
                        "score": 0,
                        "row": highs_mip["row"],
                        "completed_by": "HiGHS interval MIP",
                    }
                    record["search"] = search
            cpsat = None
            if not search["success"] and args.cpsat_time_limit > 0:
                cpsat = cyclic_target_cpsat(
                    search_forbidden,
                    modulus,
                    target,
                    time_limit=args.cpsat_time_limit,
                    workers=args.cpsat_workers,
                    hint=search["row"],
                )
                record["cpsat"] = cpsat
                if cpsat["feasible"]:
                    search = {
                        **search,
                        "success": True,
                        "score": 0,
                        "row": cpsat["row"],
                        "completed_by": "CP-SAT",
                    }
                    record["search"] = search
            full_conflict_count = None
            if search["row"] is not None:
                search_row = np.asarray(search["row"], dtype=np.int64)
                conflict_mask = target_violation_mask(
                    forbidden,
                    search_row,
                    modulus,
                    target,
                )
                conflict_indices = np.flatnonzero(conflict_mask)
                full_conflict_count = int(len(conflict_indices))
                record["fixed_metric_conflict_count"] = full_conflict_count
                record["fixed_metric_minimum_conflict_ratio"] = (
                    float(forbidden_ratios[conflict_indices].min())
                    if len(conflict_indices)
                    else None
                )
                record["fixed_metric_conflicts"] = [
                    {
                        "coordinate": forbidden[index].astype(int).tolist(),
                        "distance_ratio": float(forbidden_ratios[index]),
                        "quotient_residue": int(
                            forbidden[index] @ search_row % modulus
                        ),
                    }
                    for index in conflict_indices
                ]
            record["valid_on_fixed_metric"] = full_conflict_count == 0
            if record["valid_on_fixed_metric"]:
                row = np.asarray(search["row"], dtype=np.int64)
                connections = signed_connection_images(
                    forbidden,
                    [row],
                    [modulus],
                )
                connection_set = {
                    int(values[0]) for values in connections
                }
                if (
                    0 in connection_set
                    or target in connection_set
                    or (-target) % modulus in connection_set
                ):
                    raise AssertionError(
                        "descent witness does not omit the requested residues"
                    )
                kernel = hnf_columns(
                    kernel_basis([row], [modulus], basis.shape[0])
                )
                if abs(exact_det(kernel)) != modulus:
                    raise AssertionError("cyclic period has the wrong index")
                matching = quotient_matching_coloring(
                    connections,
                    [modulus],
                    args.target_colors,
                    time_limit=args.matching_time_limit,
                )
                record.update(
                    {
                        "row": row.astype(int).tolist(),
                        "kernel_basis_columns": kernel.astype(int).tolist(),
                        "kernel_smith": smith_diagonal(kernel),
                        "connection_keys": len(connections),
                        "missing_nonzero_quotient_classes": (
                            modulus - 1 - len(connections)
                        ),
                        "matching_coloring": matching,
                    }
                )
                if matching["success"]:
                    payload["coloring"] = record
                    payload["valid_combinatorial_witness"] = True
            payload["records"].append(record)
            save()
            print(
                f"  y={target} score={search['score']} "
                f"core-success={search['success']} "
                f"full-conflicts={full_conflict_count} "
                f"coloring={payload['valid_combinatorial_witness']}",
                flush=True,
            )
            if payload["valid_combinatorial_witness"]:
                break
        if payload["valid_combinatorial_witness"]:
            break
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
