"""Joint HiGHS search for arbitrary transversal blocks in a cyclic factor.

Let ``N = b*C`` and let a primitive row

    a : Z^d -> Z/N

define a cyclic period.  Consecutive-block colorings use
``B = {0, ..., b-1}``.  Here the base block is the arbitrary transversal

    B(t) = {r + b*t[r] : 0 <= r < b},       t[r] in Z/C.

The ``C`` translates ``B(t) + b*h`` partition ``Z/N``.  They therefore give
an exact ``C``-coloring whenever

    a*f not in B(t)-B(t)  (mod N)

for every forbidden lattice displacement ``f``.

This joint choice of ``a`` and ``t`` is an MILP, not a nonlinear modular
model.  For every ordered pair ``r != s`` introduce an integer ``q`` and
write

    1 <= a*f - (r-s) - b*(t[r]-t[s]) - N*q <= N-1.

One analogous constraint excludes the zero difference.  HiGHS solves the
resulting sparse feasibility MIP.  A CEGAR driver begins with the deepest
geometric conflicts, checks every incumbent against the complete forbidden
catalogue, and adds all newly violated vectors.

All quotient and verification arithmetic is exact integers.  A solver
``INFEASIBLE`` status is a computational MIP certificate, while a feasible
incumbent is independently rechecked before it is reported as a coloring.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from d6_cyclic_hole_search import (
    load_metric_checkpoint,
    parse_indices,
    prime_divisors,
    primitive_cyclic_row,
)
from determinant_repair import exact_det, load_preset
from prime_radon import hnf_columns, kernel_basis, smith_diagonal
from prime_row_opt import _forbidden_with_weights


def transversal_block(
    phases: Sequence[int],
    colors: int,
) -> np.ndarray:
    """Return the canonical base block ``r + b*t[r]`` in ``Z/(b*C)``."""
    phases_array = np.asarray(phases, dtype=np.int64)
    if phases_array.ndim != 1 or len(phases_array) < 2 or colors < 1:
        raise ValueError("phases must be a vector of length at least two")
    block_size = len(phases_array)
    if np.any(phases_array < 0) or np.any(phases_array >= colors):
        raise ValueError("phase lies outside Z/C")
    return (
        np.arange(block_size, dtype=np.int64)
        + block_size * phases_array
    )


def transversal_difference_residues(
    phases: Sequence[int],
    colors: int,
) -> np.ndarray:
    """Return the exact sorted difference set ``B-B`` modulo ``b*C``."""
    block = transversal_block(phases, colors)
    modulus = len(block) * int(colors)
    differences = np.remainder(
        block[:, None] - block[None, :],
        modulus,
    )
    return np.unique(differences)


def transversal_color(
    residue: int,
    phases: Sequence[int],
    colors: int,
) -> int:
    """Color one quotient residue under the transversal tiling."""
    phases_array = np.asarray(phases, dtype=np.int64)
    block_size = len(phases_array)
    modulus = block_size * int(colors)
    value = int(residue) % modulus
    low = value % block_size
    high = (value - low) // block_size
    return int((high - int(phases_array[low])) % colors)


def transversal_conflict_mask(
    forbidden: np.ndarray,
    row: Sequence[int],
    phases: Sequence[int],
    colors: int,
) -> np.ndarray:
    """Mark forbidden vectors whose quotient image lies in ``B-B``."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    row_array = np.asarray(row, dtype=np.int64)
    if forbidden_array.ndim != 2:
        raise ValueError("forbidden catalogue must be a matrix")
    if row_array.shape != (forbidden_array.shape[1],):
        raise ValueError("row has the wrong dimension")
    differences = transversal_difference_residues(phases, colors)
    modulus = len(phases) * int(colors)
    residues = np.remainder(forbidden_array @ row_array, modulus)
    lookup = np.zeros(modulus, dtype=bool)
    lookup[differences] = True
    return lookup[residues]


def transversal_row_coordinate_score_vectors(
    forbidden: np.ndarray,
    row: Sequence[int],
    phases: Sequence[int],
    colors: int,
    coordinate: int,
    *,
    hard_mask: Sequence[bool] | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every value of one row coordinate without approximation."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    row_array = np.asarray(row, dtype=np.int64)
    modulus = len(phases) * int(colors)
    if (
        forbidden_array.ndim != 2
        or row_array.shape != (forbidden_array.shape[1],)
        or coordinate < 0
        or coordinate >= forbidden_array.shape[1]
    ):
        raise ValueError("invalid row-coordinate scoring instance")
    hard_array = (
        np.zeros(len(forbidden_array), dtype=bool)
        if hard_mask is None
        else np.asarray(hard_mask, dtype=bool)
    )
    weight_array = (
        np.ones(len(forbidden_array), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if hard_array.shape != (len(forbidden_array),) or weight_array.shape != (
        len(forbidden_array),
    ):
        raise ValueError("hard mask or weights have the wrong shape")

    differences = transversal_difference_residues(phases, colors)
    lookup = np.zeros(modulus, dtype=bool)
    lookup[differences] = True
    current = np.remainder(forbidden_array @ row_array, modulus)
    coefficient = np.remainder(
        forbidden_array[:, coordinate],
        modulus,
    )
    base = np.remainder(
        current - coefficient * int(row_array[coordinate]),
        modulus,
    )
    residues = np.remainder(
        base[:, None]
        + coefficient[:, None]
        * np.arange(modulus, dtype=np.int64)[None, :],
        modulus,
    )
    violations = lookup[residues]
    total_scores = np.count_nonzero(violations, axis=0).astype(np.int64)
    hard_scores = np.count_nonzero(
        violations[hard_array],
        axis=0,
    ).astype(np.int64)
    weighted_scores = weight_array @ violations

    # Preserve exact surjectivity after every coordinate move.
    other_gcd = modulus
    for other_coordinate, value in enumerate(row_array):
        if other_coordinate != coordinate:
            other_gcd = math.gcd(other_gcd, int(value))
    primitive_values = np.asarray(
        [
            math.gcd(other_gcd, value) == 1
            for value in range(modulus)
        ],
        dtype=bool,
    )
    sentinel = len(forbidden_array) + 1
    total_scores[~primitive_values] = sentinel
    hard_scores[~primitive_values] = sentinel
    weighted_scores[~primitive_values] = np.inf
    return hard_scores, total_scores, weighted_scores


def transversal_phase_score_vectors(
    forbidden: np.ndarray,
    row: Sequence[int],
    phases: Sequence[int],
    colors: int,
    coordinate: int,
    *,
    hard_mask: Sequence[bool] | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every phase value using exact residue histograms."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    row_array = np.asarray(row, dtype=np.int64)
    phases_array = np.asarray(phases, dtype=np.int64)
    block_size = len(phases_array)
    modulus = block_size * int(colors)
    if (
        forbidden_array.ndim != 2
        or row_array.shape != (forbidden_array.shape[1],)
        or phases_array.ndim != 1
        or coordinate <= 0
        or coordinate >= block_size
    ):
        raise ValueError("invalid phase-coordinate scoring instance")
    hard_array = (
        np.zeros(len(forbidden_array), dtype=bool)
        if hard_mask is None
        else np.asarray(hard_mask, dtype=bool)
    )
    weight_array = (
        np.ones(len(forbidden_array), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if hard_array.shape != (len(forbidden_array),) or weight_array.shape != (
        len(forbidden_array),
    ):
        raise ValueError("hard mask or weights have the wrong shape")
    residues = np.remainder(forbidden_array @ row_array, modulus)
    total_histogram = np.bincount(residues, minlength=modulus)
    hard_histogram = np.bincount(
        residues[hard_array],
        minlength=modulus,
    )
    weighted_histogram = np.bincount(
        residues,
        weights=weight_array,
        minlength=modulus,
    )
    hard_scores = np.empty(colors, dtype=np.int64)
    total_scores = np.empty(colors, dtype=np.int64)
    weighted_scores = np.empty(colors, dtype=np.float64)
    candidate = phases_array.copy()
    for value in range(colors):
        candidate[coordinate] = value
        differences = transversal_difference_residues(candidate, colors)
        hard_scores[value] = int(hard_histogram[differences].sum())
        total_scores[value] = int(total_histogram[differences].sum())
        weighted_scores[value] = float(
            weighted_histogram[differences].sum()
        )
    return hard_scores, total_scores, weighted_scores


def transversal_pair_escape(
    forbidden: np.ndarray,
    row: Sequence[int],
    phases: Sequence[int],
    colors: int,
    *,
    hard_mask: Sequence[bool],
    weights: Sequence[float],
    pair_top: int,
    pair_trials: int,
    rng: np.random.Generator,
    geometry_first: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, float]]:
    """Try exact row-row and phase-row two-variable neighborhoods."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    row_array = np.asarray(row, dtype=np.int64)
    phases_array = np.asarray(phases, dtype=np.int64)
    hard_array = np.asarray(hard_mask, dtype=bool)
    weight_array = np.asarray(weights, dtype=np.float64)
    if pair_top < 1 or pair_trials < 1:
        raise ValueError("pair neighborhood budgets must be positive")

    def direct_key(
        candidate_row: np.ndarray,
        candidate_phases: np.ndarray,
    ) -> tuple[int, int, float]:
        mask = transversal_conflict_mask(
            forbidden_array,
            candidate_row,
            candidate_phases,
            colors,
        )
        return (
            int(np.count_nonzero(mask & hard_array)),
            int(np.count_nonzero(mask)),
            float(weight_array[mask].sum()),
        )

    def ordered_values(
        scores: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        order = (
            np.lexsort((scores[1], scores[2], scores[0]))
            if geometry_first
            else np.lexsort((scores[2], scores[1], scores[0]))
        )
        return order[: min(pair_top, len(order))]

    def ranking(key: tuple[int, int, float]) -> tuple[float, ...]:
        return (
            (float(key[0]), float(key[2]), float(key[1]))
            if geometry_first
            else (float(key[0]), float(key[1]), float(key[2]))
        )

    best_row = row_array.copy()
    best_phases = phases_array.copy()
    best_key = direct_key(best_row, best_phases)
    dimension = len(row_array)
    block_size = len(phases_array)

    for _ in range(pair_trials):
        if dimension >= 2 and (block_size < 2 or rng.random() < 0.5):
            left, right = rng.choice(
                dimension,
                size=2,
                replace=False,
            )
            left_scores = transversal_row_coordinate_score_vectors(
                forbidden_array,
                row_array,
                phases_array,
                colors,
                int(left),
                hard_mask=hard_array,
                weights=weight_array,
            )
            for left_value in ordered_values(left_scores):
                if not np.isfinite(left_scores[2][left_value]):
                    continue
                candidate_row = row_array.copy()
                candidate_row[left] = int(left_value)
                right_scores = transversal_row_coordinate_score_vectors(
                    forbidden_array,
                    candidate_row,
                    phases_array,
                    colors,
                    int(right),
                    hard_mask=hard_array,
                    weights=weight_array,
                )
                for right_value in ordered_values(right_scores):
                    if not np.isfinite(right_scores[2][right_value]):
                        continue
                    candidate_key = (
                        int(right_scores[0][right_value]),
                        int(right_scores[1][right_value]),
                        float(right_scores[2][right_value]),
                    )
                    if ranking(candidate_key) < ranking(best_key):
                        best_key = candidate_key
                        best_row = candidate_row.copy()
                        best_row[right] = int(right_value)
                        best_phases = phases_array.copy()
        else:
            phase_coordinate = int(rng.integers(1, block_size))
            row_coordinate = int(rng.integers(dimension))
            phase_scores = transversal_phase_score_vectors(
                forbidden_array,
                row_array,
                phases_array,
                colors,
                phase_coordinate,
                hard_mask=hard_array,
                weights=weight_array,
            )
            for phase_value in ordered_values(phase_scores):
                candidate_phases = phases_array.copy()
                candidate_phases[phase_coordinate] = int(phase_value)
                row_scores = transversal_row_coordinate_score_vectors(
                    forbidden_array,
                    row_array,
                    candidate_phases,
                    colors,
                    row_coordinate,
                    hard_mask=hard_array,
                    weights=weight_array,
                )
                for row_value in ordered_values(row_scores):
                    if not np.isfinite(row_scores[2][row_value]):
                        continue
                    candidate_key = (
                        int(row_scores[0][row_value]),
                        int(row_scores[1][row_value]),
                        float(row_scores[2][row_value]),
                    )
                    if ranking(candidate_key) < ranking(best_key):
                        best_key = candidate_key
                        best_row = row_array.copy()
                        best_row[row_coordinate] = int(row_value)
                        best_phases = candidate_phases.copy()

    exact_key = direct_key(best_row, best_phases)
    if (
        exact_key[:2] != best_key[:2]
        or not np.isclose(
            exact_key[2],
            best_key[2],
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise AssertionError("pair escape score disagrees with recount")
    return best_row, best_phases, exact_key


def alternating_transversal_descent(
    forbidden: np.ndarray,
    ratios: Sequence[float],
    colors: int,
    block_size: int,
    *,
    hard_ratio: float = 0.0,
    weight_power: float = 4.0,
    restarts: int = 20,
    sweeps: int = 12,
    top: int = 8,
    pair_top: int = 8,
    pair_trials: int = 6,
    geometry_first: bool = False,
    seed: int = 0,
    initial_states: Sequence[
        tuple[Sequence[int], Sequence[int]]
    ]
    | None = None,
) -> dict:
    """Alternate exact coordinate minimization over the row and block."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    ratio_array = np.asarray(ratios, dtype=np.float64)
    colors = int(colors)
    block_size = int(block_size)
    if (
        forbidden_array.ndim != 2
        or ratio_array.shape != (len(forbidden_array),)
        or colors < 1
        or block_size < 2
        or hard_ratio < 0
        or weight_power < 0
        or restarts < 1
        or sweeps < 1
        or top < 1
        or pair_top < 0
        or pair_trials < 0
    ):
        raise ValueError("invalid alternating-transversal instance")
    modulus = colors * block_size
    hard_mask = ratio_array < hard_ratio - 1e-12
    weights = np.power(
        np.maximum(0.0, 1.0 - ratio_array),
        weight_power,
    )
    if not np.any(weights):
        weights = np.ones(len(forbidden_array), dtype=np.float64)
    rng = np.random.default_rng(seed)
    normalized_initial_states: list[tuple[np.ndarray, np.ndarray]] = []
    for initial_row, initial_phases in initial_states or []:
        row_array = np.remainder(
            np.asarray(initial_row, dtype=np.int64),
            modulus,
        )
        phases_array = np.remainder(
            np.asarray(initial_phases, dtype=np.int64),
            colors,
        )
        if row_array.shape != (forbidden_array.shape[1],):
            raise ValueError("initial row has the wrong dimension")
        if phases_array.shape != (block_size,):
            raise ValueError("initial phases have the wrong length")
        if not primitive_cyclic_row(row_array, modulus):
            raise ValueError("initial row is not primitive")
        phases_array = np.remainder(
            phases_array - int(phases_array[0]),
            colors,
        )
        normalized_initial_states.append((row_array, phases_array))
    best_key = (len(forbidden_array) + 1,) * 2 + (float("inf"),)
    best_row = None
    best_phases = None
    best_restart = -1
    best_sweep = -1
    started = time.perf_counter()

    def state(
        row: np.ndarray,
        phases: np.ndarray,
    ) -> tuple[int, int, float]:
        mask = transversal_conflict_mask(
            forbidden_array,
            row,
            phases,
            colors,
        )
        return (
            int(np.count_nonzero(mask & hard_mask)),
            int(np.count_nonzero(mask)),
            float(weights[mask].sum()),
        )

    def choose(
        hard_scores: np.ndarray,
        total_scores: np.ndarray,
        weighted_scores: np.ndarray,
    ) -> int:
        order = (
            np.lexsort(
                (total_scores, weighted_scores, hard_scores)
            )
            if geometry_first
            else np.lexsort(
                (weighted_scores, total_scores, hard_scores)
            )
        )
        request = order[: min(top, len(order))]
        first = int(request[0])
        if geometry_first:
            same_hard = request[
                hard_scores[request] == hard_scores[first]
            ]
            best_weight = float(weighted_scores[same_hard].min())
            tolerance = max(1e-12, 0.02 * abs(best_weight))
            near = same_hard[
                weighted_scores[same_hard] <= best_weight + tolerance
            ]
        else:
            same_counts = request[
                (hard_scores[request] == hard_scores[first])
                & (total_scores[request] == total_scores[first])
            ]
            best_weight = float(weighted_scores[same_counts].min())
            tolerance = max(1e-12, 0.02 * abs(best_weight))
            near = same_counts[
                weighted_scores[same_counts] <= best_weight + tolerance
            ]
        return int(near[int(rng.integers(len(near)))])

    def ranking(key: tuple[int, int, float]) -> tuple[float, ...]:
        return (
            (float(key[0]), float(key[2]), float(key[1]))
            if geometry_first
            else (float(key[0]), float(key[1]), float(key[2]))
        )

    for restart in range(restarts):
        if restart < len(normalized_initial_states):
            row, phases = (
                normalized_initial_states[restart][0].copy(),
                normalized_initial_states[restart][1].copy(),
            )
        else:
            row = rng.integers(
                0,
                modulus,
                size=forbidden_array.shape[1],
                dtype=np.int64,
            )
            row[int(rng.integers(len(row)))] = 1
            phases = rng.integers(
                0,
                colors,
                size=block_size,
                dtype=np.int64,
            )
            phases[0] = 0
        current_key = state(row, phases)
        for sweep in range(sweeps):
            improved = False
            strictly_improved = False
            for coordinate in rng.permutation(len(row)):
                scores = transversal_row_coordinate_score_vectors(
                    forbidden_array,
                    row,
                    phases,
                    colors,
                    int(coordinate),
                    hard_mask=hard_mask,
                    weights=weights,
                )
                chosen = choose(*scores)
                candidate_key = (
                    int(scores[0][chosen]),
                    int(scores[1][chosen]),
                    float(scores[2][chosen]),
                )
                if ranking(candidate_key) <= ranking(current_key):
                    improved |= (
                        ranking(candidate_key) < ranking(current_key)
                        or chosen != int(row[coordinate])
                    )
                    strictly_improved |= (
                        ranking(candidate_key) < ranking(current_key)
                    )
                    row[coordinate] = chosen
                    current_key = candidate_key

            for coordinate in rng.permutation(
                np.arange(1, block_size, dtype=np.int64)
            ):
                scores = transversal_phase_score_vectors(
                    forbidden_array,
                    row,
                    phases,
                    colors,
                    int(coordinate),
                    hard_mask=hard_mask,
                    weights=weights,
                )
                chosen = choose(*scores)
                candidate_key = (
                    int(scores[0][chosen]),
                    int(scores[1][chosen]),
                    float(scores[2][chosen]),
                )
                if ranking(candidate_key) <= ranking(current_key):
                    improved |= (
                        ranking(candidate_key) < ranking(current_key)
                        or chosen != int(phases[coordinate])
                    )
                    strictly_improved |= (
                        ranking(candidate_key) < ranking(current_key)
                    )
                    phases[coordinate] = chosen
                    current_key = candidate_key

            exact_key = state(row, phases)
            if (
                exact_key[:2] != current_key[:2]
                or not np.isclose(
                    exact_key[2],
                    current_key[2],
                    rtol=1e-10,
                    atol=1e-12,
                )
            ):
                raise AssertionError(
                    "coordinate scores disagree with direct recount"
                )
            current_key = exact_key
            if (
                not strictly_improved
                and pair_top > 0
                and pair_trials > 0
                and current_key[1] > 0
            ):
                (
                    pair_row,
                    pair_phases,
                    pair_key,
                ) = transversal_pair_escape(
                    forbidden_array,
                    row,
                    phases,
                    colors,
                    hard_mask=hard_mask,
                    weights=weights,
                    pair_top=pair_top,
                    pair_trials=pair_trials,
                    rng=rng,
                    geometry_first=geometry_first,
                )
                if ranking(pair_key) < ranking(current_key):
                    row = pair_row
                    phases = pair_phases
                    current_key = pair_key
                    improved = True
                    strictly_improved = True
            if ranking(current_key) < ranking(best_key):
                best_key = current_key
                best_row = row.copy()
                best_phases = phases.copy()
                best_restart = restart
                best_sweep = sweep
            if current_key[1] == 0:
                break
            if not strictly_improved:
                if rng.random() < 0.7:
                    coordinate = int(rng.integers(len(row)))
                    row[coordinate] = int(rng.integers(modulus))
                    if not primitive_cyclic_row(row, modulus):
                        row[coordinate] = 1
                else:
                    coordinate = int(rng.integers(1, block_size))
                    phases[coordinate] = int(rng.integers(colors))
                current_key = state(row, phases)
        if best_key[1] == 0:
            break

    verification = (
        verify_transversal_coloring(
            forbidden_array,
            best_row,
            best_phases,
            colors,
        )
        if best_row is not None and best_phases is not None
        else None
    )
    return {
        "success": bool(
            verification is not None and verification["valid"]
        ),
        "hard_conflicts": int(best_key[0]),
        "conflicts": int(best_key[1]),
        "weighted_score": float(best_key[2]),
        "row": (
            best_row.astype(int).tolist() if best_row is not None else None
        ),
        "phases": (
            best_phases.astype(int).tolist()
            if best_phases is not None
            else None
        ),
        "verification": verification,
        "hard_ratio": float(hard_ratio),
        "weight_power": float(weight_power),
        "geometry_first": bool(geometry_first),
        "restart": best_restart,
        "sweep": best_sweep,
        "elapsed_seconds": time.perf_counter() - started,
    }


def verify_transversal_coloring(
    forbidden: np.ndarray,
    row: Sequence[int],
    phases: Sequence[int],
    colors: int,
) -> dict:
    """Independently check primitivity, tiling, and every forbidden vector."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    row_array = np.asarray(row, dtype=np.int64)
    phases_array = np.asarray(phases, dtype=np.int64)
    if forbidden_array.ndim != 2:
        raise ValueError("forbidden catalogue must be a matrix")
    if row_array.shape != (forbidden_array.shape[1],):
        raise ValueError("row has the wrong dimension")
    if phases_array.ndim != 1 or len(phases_array) < 2:
        raise ValueError("phases must contain at least two entries")
    block_size = len(phases_array)
    modulus = block_size * int(colors)
    phases_array = np.remainder(phases_array, int(colors))
    block = transversal_block(phases_array, int(colors))
    differences = transversal_difference_residues(
        phases_array,
        int(colors),
    )
    conflict_mask = transversal_conflict_mask(
        forbidden_array,
        row_array,
        phases_array,
        int(colors),
    )

    color_classes = [
        sorted(
            int((value + block_size * color) % modulus)
            for value in block
        )
        for color in range(int(colors))
    ]
    flattened = sorted(value for group in color_classes for value in group)
    partitions_quotient = flattened == list(range(modulus))
    formula_matches = all(
        sorted(
            residue
            for residue in range(modulus)
            if transversal_color(
                residue,
                phases_array,
                int(colors),
            )
            == color
        )
        == color_classes[color]
        for color in range(int(colors))
    )
    return {
        "valid": bool(
            primitive_cyclic_row(row_array, modulus)
            and partitions_quotient
            and formula_matches
            and not np.any(conflict_mask)
        ),
        "primitive_row": primitive_cyclic_row(row_array, modulus),
        "partitions_quotient": partitions_quotient,
        "color_formula_matches": formula_matches,
        "conflict_count": int(np.count_nonzero(conflict_mask)),
        "conflict_indices": np.flatnonzero(conflict_mask)
        .astype(int)
        .tolist(),
        "block": block.astype(int).tolist(),
        "difference_residues": differences.astype(int).tolist(),
        "difference_count": int(len(differences)),
        "modulus": modulus,
        "colors": int(colors),
        "block_size": block_size,
    }


def _linear_range(
    entries: Sequence[tuple[int, int | float]],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> tuple[float, float]:
    minimum = 0.0
    maximum = 0.0
    for index, coefficient_value in entries:
        coefficient = float(coefficient_value)
        lower = float(lower_bounds[int(index)])
        upper = float(upper_bounds[int(index)])
        if coefficient >= 0:
            minimum += coefficient * lower
            maximum += coefficient * upper
        else:
            minimum += coefficient * upper
            maximum += coefficient * lower
    return minimum, maximum


def joint_transversal_highs_mip(
    forbidden: np.ndarray,
    colors: int,
    block_size: int,
    *,
    time_limit: float = 60.0,
    fixed_row: Sequence[int] | None = None,
    fixed_phases: Sequence[int] | None = None,
) -> dict:
    """Solve the exact joint row/transversal feasibility MIP with HiGHS."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    colors = int(colors)
    block_size = int(block_size)
    if forbidden_array.ndim != 2 or not len(forbidden_array):
        raise ValueError("forbidden catalogue must be a nonempty matrix")
    if colors < 1 or block_size < 2 or time_limit <= 0:
        raise ValueError("invalid color count, block size, or time limit")
    dimension = forbidden_array.shape[1]
    modulus = colors * block_size
    if fixed_row is not None and len(fixed_row) != dimension:
        raise ValueError("fixed row has the wrong dimension")
    if fixed_phases is not None and len(fixed_phases) != block_size:
        raise ValueError("fixed phases have the wrong length")
    normalized_fixed_phases = None
    if fixed_phases is not None:
        normalized_fixed_phases = np.remainder(
            np.asarray(fixed_phases, dtype=np.int64)
            - int(fixed_phases[0]),
            colors,
        )

    lower_bounds: list[float] = [0.0] * dimension
    upper_bounds: list[float] = [float(modulus - 1)] * dimension
    objective: list[float] = [0.0] * dimension
    integrality: list[int] = [1] * dimension
    if fixed_row is not None:
        for coordinate, value in enumerate(fixed_row):
            normalized = int(value) % modulus
            lower_bounds[coordinate] = float(normalized)
            upper_bounds[coordinate] = float(normalized)

    phase_indices: list[int] = []
    for residue_class in range(block_size):
        phase_indices.append(len(lower_bounds))
        if residue_class == 0:
            lower = upper = 0
        elif normalized_fixed_phases is not None:
            lower = upper = int(
                normalized_fixed_phases[residue_class]
            )
        else:
            lower, upper = 0, colors - 1
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
        objective.append(0.0)
        integrality.append(1)

    # One zero-difference constraint and every ordered nonzero base-block
    # difference.  Ordered pairs are necessary because the input catalogue
    # contains one representative of each projective +/- pair.
    difference_specs: list[tuple[int | None, int | None]] = [(None, None)]
    difference_specs.extend(
        (left, right)
        for left in range(block_size)
        for right in range(block_size)
        if left != right
    )

    q_records: list[
        tuple[np.ndarray, int | None, int | None, int, float, float]
    ] = []
    for vector in forbidden_array:
        row_entries = [
            (coordinate, int(coefficient))
            for coordinate, coefficient in enumerate(vector)
            if coefficient
        ]
        for left, right in difference_specs:
            variable_entries = list(row_entries)
            constant = 0
            if left is not None and right is not None:
                variable_entries.extend(
                    [
                        (phase_indices[left], -block_size),
                        (phase_indices[right], block_size),
                    ]
                )
                constant = left - right
            allowed_lower = float(1 + constant)
            allowed_upper = float(modulus - 1 + constant)
            expression_min, expression_max = _linear_range(
                variable_entries,
                lower_bounds,
                upper_bounds,
            )
            q_lower = math.floor(
                (expression_min - allowed_upper) / modulus
            )
            q_upper = math.ceil(
                (expression_max - allowed_lower) / modulus
            )
            q_index = len(lower_bounds)
            lower_bounds.append(float(q_lower))
            upper_bounds.append(float(q_upper))
            objective.append(0.0)
            integrality.append(1)
            q_records.append(
                (
                    vector,
                    left,
                    right,
                    q_index,
                    allowed_lower,
                    allowed_upper,
                )
            )

    # Exact row primitivity: for every prime p|N at least one coordinate is
    # nonzero modulo p.
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
        for column, coefficient_value in entries:
            coefficient = float(coefficient_value)
            if coefficient:
                row_indices.append(row_index)
                column_indices.append(int(column))
                coefficients.append(coefficient)
        constraint_lower.append(float(lower))
        constraint_upper.append(float(upper))

    for (
        vector,
        left,
        right,
        q_index,
        allowed_lower,
        allowed_upper,
    ) in q_records:
        entries: list[tuple[int, int | float]] = [
            (coordinate, int(coefficient))
            for coordinate, coefficient in enumerate(vector)
            if coefficient
        ]
        if left is not None and right is not None:
            entries.extend(
                [
                    (phase_indices[left], -block_size),
                    (phase_indices[right], block_size),
                ]
            )
        entries.append((q_index, -modulus))
        add_constraint(entries, allowed_lower, allowed_upper)

    for prime, k_indices, residue_indices, nonzero_indices in primitive_blocks:
        for coordinate in range(dimension):
            add_constraint(
                [
                    (coordinate, 1),
                    (k_indices[coordinate], -prime),
                    (residue_indices[coordinate], -1),
                ],
                0.0,
                0.0,
            )
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
    feasible = result.x is not None
    row = None
    phases = None
    verification = None
    if feasible:
        row = (
            np.rint(result.x[:dimension]).astype(np.int64) % modulus
        )
        phases = np.asarray(
            [
                int(round(result.x[index])) % colors
                for index in phase_indices
            ],
            dtype=np.int64,
        )
        verification = verify_transversal_coloring(
            forbidden_array,
            row,
            phases,
            colors,
        )
        if not verification["valid"]:
            raise AssertionError(
                "HiGHS returned a numerically invalid transversal incumbent"
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
        "phases": (
            phases.astype(int).tolist() if phases is not None else None
        ),
        "verification": verification,
        "colors": colors,
        "block_size": block_size,
        "modulus": modulus,
        "forbidden_vectors": int(len(forbidden_array)),
        "difference_constraints_per_vector": len(difference_specs),
        "variables": variable_count,
        "constraints": len(constraint_lower),
        "nonzeros": int(matrix.nnz),
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
        "solver": (
            "SciPy milp with open HiGHS joint cyclic-row/transversal model"
        ),
    }


def joint_transversal_cpsat(
    forbidden: np.ndarray,
    colors: int,
    block_size: int,
    *,
    time_limit: float = 60.0,
    workers: int = 8,
    hint_row: Sequence[int] | None = None,
    hint_phases: Sequence[int] | None = None,
    minimize_conflicts: bool = False,
    hard_indices: Sequence[int] | None = None,
    conflict_weights: Sequence[int] | None = None,
) -> dict:
    """Solve the same joint model with native CP-SAT modular constraints."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    colors = int(colors)
    block_size = int(block_size)
    if forbidden_array.ndim != 2 or not len(forbidden_array):
        raise ValueError("forbidden catalogue must be a nonempty matrix")
    if (
        colors < 1
        or block_size < 2
        or time_limit <= 0
        or workers < 1
    ):
        raise ValueError("invalid color count, block size, or budget")
    dimension = forbidden_array.shape[1]
    modulus = colors * block_size
    if hint_row is not None and len(hint_row) != dimension:
        raise ValueError("row hint has the wrong dimension")
    if hint_phases is not None and len(hint_phases) != block_size:
        raise ValueError("phase hint has the wrong length")
    if hard_indices is not None and not minimize_conflicts:
        raise ValueError("hard indices require conflict minimization")
    hard_set = {
        int(index)
        for index in (hard_indices if hard_indices is not None else [])
    }
    if any(index < 0 or index >= len(forbidden_array) for index in hard_set):
        raise ValueError("hard conflict index lies outside catalogue")
    if conflict_weights is None:
        weight_array = np.ones(len(forbidden_array), dtype=np.int64)
    else:
        weight_array = np.asarray(conflict_weights, dtype=np.int64)
        if (
            weight_array.shape != (len(forbidden_array),)
            or np.any(weight_array < 1)
        ):
            raise ValueError("conflict weights must be positive integers")

    model = cp_model.CpModel()
    row_variables = [
        model.NewIntVar(0, modulus - 1, f"a_{coordinate}")
        for coordinate in range(dimension)
    ]
    phase_variables = [
        model.NewIntVar(
            0,
            0 if residue_class == 0 else colors - 1,
            f"t_{residue_class}",
        )
        for residue_class in range(block_size)
    ]
    difference_variables: list[cp_model.IntVar] = []
    for left in range(block_size):
        for right in range(block_size):
            if left == right:
                continue
            difference = model.NewIntVar(
                0,
                modulus - 1,
                f"delta_{left}_{right}",
            )
            expression = (
                (left - right) % modulus
                + block_size * phase_variables[left]
                + (modulus - block_size) * phase_variables[right]
            )
            model.AddModuloEquality(difference, expression, modulus)
            difference_variables.append(difference)

    image_variables: list[cp_model.IntVar] = []
    violation_variables: list[tuple[int, cp_model.IntVar]] = []
    for vector_index, vector in enumerate(forbidden_array):
        image = model.NewIntVar(
            0,
            modulus - 1,
            f"image_{vector_index}",
        )
        expression = sum(
            int(coefficient % modulus) * row_variables[coordinate]
            for coordinate, coefficient in enumerate(vector)
        )
        model.AddModuloEquality(image, expression, modulus)
        if minimize_conflicts and vector_index not in hard_set:
            violation = model.NewBoolVar(f"violation_{vector_index}")
            model.Add(image != 0).OnlyEnforceIf(violation.Not())
            for difference in difference_variables:
                model.Add(image != difference).OnlyEnforceIf(
                    violation.Not()
                )
            violation_variables.append((vector_index, violation))
        else:
            model.Add(image != 0)
            for difference in difference_variables:
                model.Add(image != difference)
        image_variables.append(image)

    for prime in prime_divisors(modulus):
        nonzero: list[cp_model.IntVar] = []
        for coordinate, variable in enumerate(row_variables):
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

    if hint_row is not None:
        for variable, value in zip(row_variables, hint_row):
            model.AddHint(variable, int(value) % modulus)
    if hint_phases is not None:
        shift = int(hint_phases[0])
        for variable, value in zip(phase_variables, hint_phases):
            model.AddHint(variable, (int(value) - shift) % colors)
    model.AddDecisionStrategy(
        [*phase_variables, *row_variables],
        cp_model.CHOOSE_MIN_DOMAIN_SIZE,
        cp_model.SELECT_MIN_VALUE,
    )
    if minimize_conflicts:
        model.Minimize(
            sum(
                int(weight_array[index]) * violation
                for index, violation in violation_variables
            )
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = False
    started = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - started
    feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    row = None
    phases = None
    verification = None
    if feasible:
        row = np.asarray(
            [int(solver.Value(variable)) for variable in row_variables],
            dtype=np.int64,
        )
        phases = np.asarray(
            [int(solver.Value(variable)) for variable in phase_variables],
            dtype=np.int64,
        )
        verification = verify_transversal_coloring(
            forbidden_array,
            row,
            phases,
            colors,
        )
        if not minimize_conflicts and not verification["valid"]:
            raise AssertionError(
                "CP-SAT returned an invalid transversal incumbent"
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
        "row": row.astype(int).tolist() if row is not None else None,
        "phases": (
            phases.astype(int).tolist() if phases is not None else None
        ),
        "verification": verification,
        "valid_coloring": bool(
            verification is not None and verification["valid"]
        ),
        "exact_conflict_count": (
            verification["conflict_count"]
            if verification is not None
            else None
        ),
        "exact_conflict_indices": (
            verification["conflict_indices"]
            if verification is not None
            else None
        ),
        "minimize_conflicts": bool(minimize_conflicts),
        "hard_conflict_vectors": int(len(hard_set)),
        "soft_conflict_vectors": int(len(violation_variables)),
        "objective_value": (
            float(solver.ObjectiveValue())
            if feasible and minimize_conflicts
            else None
        ),
        "best_objective_bound": (
            float(solver.BestObjectiveBound())
            if minimize_conflicts
            else None
        ),
        "colors": colors,
        "block_size": block_size,
        "modulus": modulus,
        "forbidden_vectors": int(len(forbidden_array)),
        "difference_variables": len(difference_variables),
        "image_variables": len(image_variables),
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
        "wall_time": float(solver.WallTime()),
        "elapsed_seconds": elapsed,
        "solver": "OR-Tools CP-SAT joint modular transversal model",
    }


def coloring_record(
    row: Sequence[int],
    phases: Sequence[int],
    colors: int,
    forbidden: np.ndarray,
) -> dict:
    """Build the exact period and public coloring certificate."""
    row_array = np.asarray(row, dtype=np.int64)
    phases_array = np.asarray(phases, dtype=np.int64)
    verification = verify_transversal_coloring(
        forbidden,
        row_array,
        phases_array,
        colors,
    )
    if not verification["valid"]:
        raise ValueError("cannot certify an invalid transversal coloring")
    modulus = len(phases_array) * int(colors)
    period = hnf_columns(
        kernel_basis(
            [row_array],
            [modulus],
            len(row_array),
        )
    )
    if abs(exact_det(period)) != modulus:
        raise AssertionError(
            "primitive transversal row has the wrong period index"
        )
    return {
        "row": row_array.astype(int).tolist(),
        "phases": phases_array.astype(int).tolist(),
        "base_block": verification["block"],
        "difference_residues": verification["difference_residues"],
        "coloring_rule": (
            "r = row*x mod N; low = r mod b; "
            "color = ((r-low)/b - phase[low]) mod C"
        ),
        "period_basis_columns": period.astype(int).tolist(),
        "period_smith": smith_diagonal(period),
        "verification": verification,
    }


def transversal_cegar(
    forbidden: np.ndarray,
    ratios: Sequence[float],
    colors: int,
    block_size: int,
    *,
    initial_ratio: float = 0.0,
    max_rounds: int = 12,
    time_limit: float = 60.0,
    initial_core_indices: Sequence[int] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run exact-core CEGAR and check every incumbent on the full catalogue."""
    forbidden_array = np.asarray(forbidden, dtype=np.int64)
    ratio_array = np.asarray(ratios, dtype=np.float64)
    if forbidden_array.ndim != 2 or ratio_array.shape != (
        len(forbidden_array),
    ):
        raise ValueError("forbidden vectors and ratios are incompatible")
    if initial_ratio < 0 or max_rounds < 1 or time_limit <= 0:
        raise ValueError("invalid CEGAR threshold or budget")

    core = set(
        int(index)
        for index in np.flatnonzero(
            ratio_array <= float(initial_ratio) + 1e-12
        )
    )
    if not core:
        core.add(int(np.argmin(ratio_array)))
    if initial_core_indices is not None:
        for index_value in initial_core_indices:
            index = int(index_value)
            if index < 0 or index >= len(forbidden_array):
                raise ValueError("initial core index lies outside catalogue")
            core.add(index)
    history: list[dict] = []
    started = time.perf_counter()
    outcome = "ROUND_LIMIT"
    coloring = None

    for round_number in range(max_rounds):
        core_indices = np.asarray(sorted(core), dtype=np.int64)
        solve = joint_transversal_highs_mip(
            forbidden_array[core_indices],
            colors,
            block_size,
            time_limit=time_limit,
        )
        record: dict = {
            "round": round_number,
            "core_size": int(len(core_indices)),
            "core_minimum_ratio": float(ratio_array[core_indices].min()),
            "core_maximum_ratio": float(ratio_array[core_indices].max()),
            "solve": solve,
        }
        if not solve["feasible"]:
            outcome = (
                "CORE_INFEASIBLE"
                if solve["proven_infeasible"]
                else "SOLVER_LIMIT"
            )
            history.append(record)
            if progress is not None:
                progress(record)
            break

        row = np.asarray(solve["row"], dtype=np.int64)
        phases = np.asarray(solve["phases"], dtype=np.int64)
        full_verification = verify_transversal_coloring(
            forbidden_array,
            row,
            phases,
            colors,
        )
        conflict_indices = np.asarray(
            full_verification["conflict_indices"],
            dtype=np.int64,
        )
        record["full_verification"] = full_verification
        record["new_conflict_count"] = int(
            sum(int(index) not in core for index in conflict_indices)
        )
        if len(conflict_indices):
            record["full_minimum_conflict_ratio"] = float(
                ratio_array[conflict_indices].min()
            )
            record["full_maximum_conflict_ratio"] = float(
                ratio_array[conflict_indices].max()
            )
            record["conflicts"] = [
                {
                    "index": int(index),
                    "coordinate": forbidden_array[index]
                    .astype(int)
                    .tolist(),
                    "distance_ratio": float(ratio_array[index]),
                    "quotient_residue": int(
                        forbidden_array[index] @ row
                        % (colors * block_size)
                    ),
                }
                for index in conflict_indices
            ]
            core.update(int(index) for index in conflict_indices)
        else:
            outcome = "COLORING"
            coloring = coloring_record(
                row,
                phases,
                colors,
                forbidden_array,
            )
        history.append(record)
        if progress is not None:
            progress(record)
        if coloring is not None:
            break

    return {
        "outcome": outcome,
        "colors": int(colors),
        "block_size": int(block_size),
        "modulus": int(colors * block_size),
        "initial_ratio": float(initial_ratio),
        "max_rounds": int(max_rounds),
        "time_limit_per_round": float(time_limit),
        "final_core_size": int(len(core)),
        "final_core_indices": sorted(core),
        "history": history,
        "coloring": coloring,
        "valid_combinatorial_witness": coloring is not None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-colors",
        type=parse_indices,
        default=[342],
    )
    parser.add_argument(
        "--block-sizes",
        type=parse_indices,
        default=[3],
    )
    parser.add_argument("--initial-ratio", type=float, default=0.0)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--heuristic-restarts", type=int, default=0)
    parser.add_argument("--heuristic-sweeps", type=int, default=12)
    parser.add_argument("--heuristic-top", type=int, default=8)
    parser.add_argument("--heuristic-pair-top", type=int, default=8)
    parser.add_argument("--heuristic-pair-trials", type=int, default=6)
    parser.add_argument("--heuristic-hard-ratio", type=float, default=0.0)
    parser.add_argument("--heuristic-weight-power", type=float, default=4.0)
    parser.add_argument(
        "--heuristic-geometry-first",
        action="store_true",
    )
    parser.add_argument("--heuristic-seed", type=int, default=20260731)
    parser.add_argument("--cpsat-time-limit", type=float, default=0.0)
    parser.add_argument(
        "--cpsat-soft-time-limit",
        type=float,
        default=0.0,
    )
    parser.add_argument("--cpsat-workers", type=int, default=8)
    parser.add_argument("--metric-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        any(value < 1 for value in args.target_colors)
        or any(value < 2 for value in args.block_sizes)
        or args.initial_ratio < 0
        or args.max_rounds < 1
        or args.time_limit <= 0
        or args.heuristic_restarts < 0
        or args.heuristic_sweeps < 1
        or args.heuristic_top < 1
        or args.heuristic_pair_top < 0
        or args.heuristic_pair_trials < 0
        or args.heuristic_hard_ratio < 0
        or args.heuristic_weight_power < 0
        or args.cpsat_time_limit < 0
        or args.cpsat_soft_time_limit < 0
        or args.cpsat_workers < 1
    ):
        parser.error("invalid colors, block sizes, threshold, or budget")

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
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    resume_payload = None
    resume_cores: dict[tuple[int, int], set[int]] = {}
    resume_hints: dict[tuple[int, int], dict] = {}
    if args.resume is not None:
        resume_payload = json.loads(args.resume.read_text())
        for record in resume_payload.get("records", []):
            key = (
                int(record["colors"]),
                int(record["block_size"]),
            )
            core = {
                int(index)
                for index in record.get("final_core_indices", [])
            }
            if not core:
                for history_record in record.get("history", []):
                    core.update(
                        int(conflict["index"])
                        for conflict in history_record.get("conflicts", [])
                    )
            resume_cores[key] = core
            hint = next(
                (
                    candidate
                    for candidate in (
                        record.get("soft_cpsat"),
                        record.get("full_cpsat"),
                    )
                    if candidate is not None
                    and candidate.get("feasible")
                    and candidate.get("row") is not None
                ),
                None,
            )
            if hint is None:
                hint = next(
                    (
                        history_record["solve"]
                        for history_record in reversed(
                            record.get("history", [])
                        )
                        if history_record["solve"].get("feasible")
                    ),
                    None,
                )
            if hint is not None:
                resume_hints[key] = hint
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "joint cyclic-row/arbitrary-transversal-block HiGHS MIP "
            "with full geometric CEGAR"
        ),
        "lattice": lattice,
        "dimension": len(basis),
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
        "resume": str(args.resume) if args.resume is not None else None,
        "resume_method": (
            resume_payload.get("method")
            if resume_payload is not None
            else None
        ),
        "parent_basis": basis.tolist(),
        "parent_diameter": float(diameter),
        "forbidden_projective_pairs": int(len(forbidden)),
        "settings": {
            "target_colors": args.target_colors,
            "block_sizes": args.block_sizes,
            "initial_ratio": args.initial_ratio,
            "max_rounds": args.max_rounds,
            "time_limit": args.time_limit,
            "heuristic_restarts": args.heuristic_restarts,
            "heuristic_sweeps": args.heuristic_sweeps,
            "heuristic_top": args.heuristic_top,
            "heuristic_pair_top": args.heuristic_pair_top,
            "heuristic_pair_trials": args.heuristic_pair_trials,
            "heuristic_hard_ratio": args.heuristic_hard_ratio,
            "heuristic_weight_power": args.heuristic_weight_power,
            "heuristic_geometry_first": args.heuristic_geometry_first,
            "heuristic_seed": args.heuristic_seed,
            "cpsat_time_limit": args.cpsat_time_limit,
            "cpsat_soft_time_limit": args.cpsat_soft_time_limit,
            "cpsat_workers": args.cpsat_workers,
        },
        "records": [],
        "coloring": None,
        "valid_combinatorial_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    cases = [
        (colors, block_size)
        for colors in args.target_colors
        for block_size in args.block_sizes
    ]
    for case_number, (colors, block_size) in enumerate(cases):
        print(
            f"[{case_number + 1}/{len(cases)}] "
            f"C={colors} b={block_size} N={colors * block_size}",
            flush=True,
        )

        def progress(record: dict) -> None:
            solve = record["solve"]
            print(
                f"  round={record['round']} core={record['core_size']} "
                f"status={solve['status']} feasible={solve['feasible']} "
                f"full={record.get('full_verification', {}).get('conflict_count')} "
                f"new={record.get('new_conflict_count')}",
                flush=True,
            )

        heuristic = None
        if args.heuristic_restarts > 0:
            resume_hint = resume_hints.get((colors, block_size))
            heuristic = alternating_transversal_descent(
                forbidden,
                ratios,
                colors,
                block_size,
                hard_ratio=args.heuristic_hard_ratio,
                weight_power=args.heuristic_weight_power,
                restarts=args.heuristic_restarts,
                sweeps=args.heuristic_sweeps,
                top=args.heuristic_top,
                pair_top=args.heuristic_pair_top,
                pair_trials=args.heuristic_pair_trials,
                geometry_first=args.heuristic_geometry_first,
                seed=(
                    args.heuristic_seed
                    + 1_000_003 * case_number
                ),
                initial_states=(
                    [
                        (
                            resume_hint["row"],
                            resume_hint["phases"],
                        )
                    ]
                    if resume_hint is not None
                    else None
                ),
            )
            print(
                f"  heuristic hard={heuristic['hard_conflicts']} "
                f"total={heuristic['conflicts']} "
                f"weighted={heuristic['weighted_score']:.9g}",
                flush=True,
            )
        if heuristic is not None and heuristic["success"]:
            record = {
                "outcome": "COLORING",
                "colors": colors,
                "block_size": block_size,
                "modulus": colors * block_size,
                "initial_ratio": args.initial_ratio,
                "max_rounds": args.max_rounds,
                "time_limit_per_round": args.time_limit,
                "final_core_size": 0,
                "final_core_indices": [],
                "history": [],
                "heuristic": heuristic,
                "coloring": coloring_record(
                    heuristic["row"],
                    heuristic["phases"],
                    colors,
                    forbidden,
                ),
                "valid_combinatorial_witness": True,
                "elapsed_seconds": heuristic["elapsed_seconds"],
            }
        else:
            record = transversal_cegar(
                forbidden,
                ratios,
                colors,
                block_size,
                initial_ratio=args.initial_ratio,
                max_rounds=args.max_rounds,
                time_limit=args.time_limit,
                initial_core_indices=resume_cores.get(
                    (colors, block_size),
                ),
                progress=progress,
            )
            record["heuristic"] = heuristic
        if (
            not record["valid_combinatorial_witness"]
            and args.cpsat_time_limit > 0
        ):
            last_hint = next(
                (
                    history_record["solve"]
                    for history_record in reversed(record["history"])
                    if history_record["solve"]["feasible"]
                ),
                resume_hints.get((colors, block_size)),
            )
            if heuristic is not None and heuristic["row"] is not None:
                last_hint = heuristic
            full_cpsat = joint_transversal_cpsat(
                forbidden,
                colors,
                block_size,
                time_limit=args.cpsat_time_limit,
                workers=args.cpsat_workers,
                hint_row=(
                    last_hint["row"] if last_hint is not None else None
                ),
                hint_phases=(
                    last_hint["phases"] if last_hint is not None else None
                ),
            )
            record["full_cpsat"] = full_cpsat
            print(
                f"  full-cpsat status={full_cpsat['status']} "
                f"feasible={full_cpsat['feasible']} "
                f"conflicts={full_cpsat['conflicts']} "
                f"branches={full_cpsat['branches']}",
                flush=True,
            )
            if full_cpsat["valid_coloring"]:
                record["outcome"] = "COLORING"
                record["coloring"] = coloring_record(
                    full_cpsat["row"],
                    full_cpsat["phases"],
                    colors,
                    forbidden,
                )
                record["valid_combinatorial_witness"] = True
        if (
            not record["valid_combinatorial_witness"]
            and args.cpsat_soft_time_limit > 0
        ):
            last_hint = next(
                (
                    history_record["solve"]
                    for history_record in reversed(record["history"])
                    if history_record["solve"]["feasible"]
                ),
                resume_hints.get((colors, block_size)),
            )
            if heuristic is not None and heuristic["row"] is not None:
                last_hint = heuristic
            soft_cpsat = joint_transversal_cpsat(
                forbidden,
                colors,
                block_size,
                time_limit=args.cpsat_soft_time_limit,
                workers=args.cpsat_workers,
                hint_row=(
                    last_hint["row"] if last_hint is not None else None
                ),
                hint_phases=(
                    last_hint["phases"] if last_hint is not None else None
                ),
                minimize_conflicts=True,
                hard_indices=record["final_core_indices"],
            )
            record["soft_cpsat"] = soft_cpsat
            print(
                f"  soft-cpsat status={soft_cpsat['status']} "
                f"exact={soft_cpsat['exact_conflict_count']} "
                f"objective={soft_cpsat['objective_value']} "
                f"bound={soft_cpsat['best_objective_bound']}",
                flush=True,
            )
            if soft_cpsat["valid_coloring"]:
                record["outcome"] = "COLORING"
                record["coloring"] = coloring_record(
                    soft_cpsat["row"],
                    soft_cpsat["phases"],
                    colors,
                    forbidden,
                )
                record["valid_combinatorial_witness"] = True
        payload["records"].append(record)
        if record["valid_combinatorial_witness"]:
            payload["coloring"] = record["coloring"]
            payload["valid_combinatorial_witness"] = True
        save()
        if payload["valid_combinatorial_witness"]:
            break
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
