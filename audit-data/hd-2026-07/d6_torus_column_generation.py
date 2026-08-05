"""HiGHS screens for non-coset periodic colorings in dimension six.

The usual lattice construction assigns a different color to every coset of a
valid sublattice ``K``.  This file studies a strictly larger class.  For a
finer period ``P <= K``, vertices are the cells in ``Z^n / P`` and two
vertices are adjacent when their difference is represented by a forbidden
cell displacement.  A proper coloring of this finite Cayley graph is a
periodic coloring, but a color class no longer has to be a coset of a
sublattice.

There are two complementary algorithms.

* A source-extension screen treats every prime-index refinement

      P = K ker(a : Z^n -> F_p)

  without explicitly constructing its graph.  For every nonzero coset of
  ``Z^n / K`` it records which of the ``p`` lifted residues are hit by the
  forbidden set.  If all residues are hit for every source coset, the graph is
  the complete ``|Z^n/K|``-partite graph with parts of size ``p``.  Its
  chromatic number is therefore exactly ``|Z^n/K|``: no non-coset improvement
  is possible for that period.

* For a non-complete screen, a generic finite Cayley graph is built.  HiGHS
  solves maximum(-weight) independent-set pricing as a MIP and the set-cover
  master as an LP or MIP.  Translating a priced independent set through the
  Cayley group adds its full orbit of columns.  Because a Cayley graph is
  vertex-transitive, its fractional chromatic number is ``|V| / alpha(G)``;
  the master is nevertheless retained both as an implementation check and as
  a route to an integral coloring.

All quotient arithmetic and graph incidences are exact integers.  The
forbidden displacement catalogue is inherited from the existing complete
Voronoi oracle.  A discovered coloring must still pass the independent
geometric verifier before it can be promoted to a rigorous Euclidean bound.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csc_matrix
from sympy import Matrix, isprime

from d6_periodic_lift_highs import (
    quotient_key,
    quotient_map,
    quotient_representatives,
)
from determinant_repair import exact_det, load_preset
from prime_radon import (
    hnf_columns,
    kernel_basis,
    projective_forms,
)
from prime_row_opt import _forbidden_with_weights


@dataclass
class CayleyConflictGraph:
    """A finite undirected Cayley graph in exact quotient coordinates."""

    period: np.ndarray
    determinant: int
    keys: tuple[tuple[int, ...], ...]
    index_by_key: dict[tuple[int, ...], int]
    connection_keys: frozenset[tuple[int, ...]]
    loop_keys: frozenset[tuple[int, ...]]

    @property
    def vertex_count(self) -> int:
        return len(self.keys)

    @property
    def loop_free(self) -> bool:
        return not self.loop_keys

    @property
    def degree(self) -> int:
        if not self.loop_free:
            raise ValueError("a looped conflict graph has no proper coloring")
        return len(self.connection_keys)

    def difference_key(self, left: int, right: int) -> tuple[int, ...]:
        return tuple(
            (self.keys[right][coordinate] - self.keys[left][coordinate])
            % self.determinant
            for coordinate in range(len(self.keys[left]))
        )

    def adjacent(self, left: int, right: int) -> bool:
        if left == right:
            return not self.loop_free
        return self.difference_key(left, right) in self.connection_keys

    def translate_vertex(self, vertex: int, shift: int) -> int:
        key = tuple(
            (self.keys[vertex][coordinate] + self.keys[shift][coordinate])
            % self.determinant
            for coordinate in range(len(self.keys[vertex]))
        )
        return self.index_by_key[key]

    def translate_set(
        self, vertices: Iterable[int], shift: int
    ) -> tuple[int, ...]:
        return tuple(
            sorted(self.translate_vertex(int(vertex), shift) for vertex in vertices)
        )

    def is_independent(self, vertices: Sequence[int]) -> bool:
        ordered = sorted(set(int(vertex) for vertex in vertices))
        if len(ordered) != len(vertices) or not self.loop_free:
            return False
        return all(
            not self.adjacent(ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        )

    def edge_array(
        self, vertices: Sequence[int] | None = None
    ) -> np.ndarray:
        """Return each induced edge once as a two-column integer array."""
        if vertices is None:
            chosen = list(range(self.vertex_count))
            position = None
        else:
            chosen = sorted(set(int(vertex) for vertex in vertices))
            position = {vertex: index for index, vertex in enumerate(chosen)}
        edges: list[tuple[int, int]] = []
        for local_left, left in enumerate(chosen):
            for local_right in range(local_left + 1, len(chosen)):
                right = chosen[local_right]
                if self.adjacent(left, right):
                    if position is None:
                        edges.append((left, right))
                    else:
                        edges.append((local_left, local_right))
        if not edges:
            return np.empty((0, 2), dtype=np.int64)
        return np.asarray(edges, dtype=np.int64)


def canonical_columns(
    columns: Iterable[Sequence[int]],
) -> list[tuple[int, ...]]:
    """Normalize a family of nonempty vertex subsets."""
    unique: set[tuple[int, ...]] = set()
    for column in columns:
        normalized = tuple(sorted(set(int(vertex) for vertex in column)))
        if not normalized:
            raise ValueError("set-cover columns must be nonempty")
        unique.add(normalized)
    return sorted(unique, key=lambda column: (len(column), column))


def build_cayley_graph(
    period: np.ndarray,
    forbidden: np.ndarray,
) -> CayleyConflictGraph:
    """Build the exact quotient graph induced by ``forbidden +/-``."""
    period = np.asarray(period, dtype=np.int64)
    forbidden = np.asarray(forbidden, dtype=np.int64)
    if (
        period.ndim != 2
        or period.shape[0] != period.shape[1]
        or forbidden.ndim != 2
        or forbidden.shape[1] != period.shape[0]
    ):
        raise ValueError("period and forbidden arrays have incompatible shapes")
    determinant, adjugate = quotient_map(period)
    representatives, index_by_key = quotient_representatives(period)
    keys = tuple(
        quotient_key(representative, adjugate, determinant)
        for representative in representatives
    )
    if len(set(keys)) != determinant or len(index_by_key) != determinant:
        raise AssertionError("quotient enumeration is not bijective")
    zero = tuple(0 for _ in range(period.shape[0]))
    raw_connections = {
        quotient_key(vector, adjugate, determinant)
        for forbidden_vector in forbidden
        for vector in (forbidden_vector, -forbidden_vector)
    }
    loop_keys = frozenset({zero} & raw_connections)
    connections = frozenset(raw_connections - {zero})
    graph = CayleyConflictGraph(
        period=period.copy(),
        determinant=determinant,
        keys=keys,
        index_by_key=dict(index_by_key),
        connection_keys=connections,
        loop_keys=loop_keys,
    )
    for connection in connections:
        inverse = tuple((-value) % determinant for value in connection)
        if inverse not in connections:
            raise AssertionError("connection set is not centrally symmetric")
    return graph


def source_extension_coordinates(
    source_kernel: np.ndarray,
    forbidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Express signed forbidden vectors as ``representative + K*q``.

    Returns ``q``, the source-coset index of every signed vector, and the
    number of source quotient classes.  The zero source class is forbidden:
    its presence would mean that the source kernel itself is not a coloring.
    """
    source_kernel = np.asarray(source_kernel, dtype=np.int64)
    forbidden = np.asarray(forbidden, dtype=np.int64)
    determinant, adjugate = quotient_map(source_kernel)
    representatives, index_by_key = quotient_representatives(source_kernel)
    signed = np.vstack([forbidden, -forbidden])
    class_ids = np.empty(len(signed), dtype=np.int64)
    deltas = np.empty_like(signed)
    for index, vector in enumerate(signed):
        key = quotient_key(vector, adjugate, determinant)
        class_id = index_by_key[key]
        class_ids[index] = class_id
        deltas[index] = vector - representatives[class_id]
    if np.any(class_ids == index_by_key[tuple(0 for _ in range(len(source_kernel)))]):
        raise ValueError("source kernel contains a forbidden displacement")

    inverse = Matrix(source_kernel.tolist()).inv()
    coordinates: list[list[int]] = []
    for delta in deltas:
        exact = inverse * Matrix(delta.astype(int).tolist())
        if any(value.q != 1 for value in exact):
            raise AssertionError("source-extension coordinate is not integral")
        coordinates.append([int(value) for value in exact])
    return np.asarray(coordinates, dtype=np.int64), class_ids, determinant


def character_extension_profiles(
    coordinates: np.ndarray,
    class_ids: np.ndarray,
    source_order: int,
    characters: np.ndarray,
    prime: int,
    *,
    batch_size: int = 512,
) -> dict:
    """Count lifted connection classes for every prime character.

    The result is exact integer arithmetic.  ``minimum_residues_per_class``
    equals ``prime`` precisely when the refinement graph is the complete
    ``source_order``-partite graph with parts of size ``prime``.
    """
    coordinates = np.asarray(coordinates, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    characters = np.asarray(characters, dtype=np.int64)
    if coordinates.ndim != 2 or characters.ndim != 2:
        raise ValueError("coordinates and characters must be matrices")
    if coordinates.shape[1] != characters.shape[1]:
        raise ValueError("coordinate and character dimensions differ")
    if len(class_ids) != len(coordinates):
        raise ValueError("one class id is required per coordinate")
    if prime < 2 or not isprime(prime):
        raise ValueError("prime must be prime")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    populated_classes = sorted(set(int(value) for value in class_ids))
    if len(populated_classes) != source_order - 1:
        raise ValueError(
            "the signed forbidden set must hit every nonzero source class: "
            f"{len(populated_classes)} of {source_order - 1}"
        )
    groups = [np.flatnonzero(class_ids == value) for value in populated_classes]
    lookup = np.asarray(
        [int(value).bit_count() for value in range(1 << prime)],
        dtype=np.int16,
    )
    full_mask = (1 << prime) - 1
    connection_counts = np.empty(len(characters), dtype=np.int64)
    minimum_coverages = np.empty(len(characters), dtype=np.int16)

    for start in range(0, len(characters), batch_size):
        stop = min(start + batch_size, len(characters))
        batch = characters[start:stop]
        residues = (coordinates @ batch.T) % prime
        counts = np.zeros(len(batch), dtype=np.int64)
        minima = np.full(len(batch), prime, dtype=np.int16)
        for indices in groups:
            masks = np.bitwise_or.reduce(
                np.left_shift(
                    np.uint16(1),
                    residues[indices].astype(np.uint16),
                ),
                axis=0,
            )
            coverages = lookup[masks]
            counts += coverages
            minima = np.minimum(minima, coverages)
        connection_counts[start:stop] = counts
        minimum_coverages[start:stop] = minima

    complete = minimum_coverages == prime
    histogram: dict[str, int] = {}
    for count in connection_counts:
        key = str(int(count))
        histogram[key] = histogram.get(key, 0) + 1
    incomplete_indices = np.flatnonzero(~complete)
    return {
        "prime": int(prime),
        "character_count": int(len(characters)),
        "period_order": int(source_order * prime),
        "expected_complete_multipartite_connections": int(
            (source_order - 1) * prime
        ),
        "connection_count_minimum": int(connection_counts.min()),
        "connection_count_maximum": int(connection_counts.max()),
        "connection_count_histogram": histogram,
        "minimum_residues_per_source_class": int(
            minimum_coverages.min()
        ),
        "complete_multipartite_characters": int(complete.sum()),
        "incomplete_character_indices": incomplete_indices.astype(int).tolist(),
        "_connection_counts": connection_counts,
        "_minimum_coverages": minimum_coverages,
    }


def subspace_extension_profiles(
    coordinates: np.ndarray,
    class_ids: np.ndarray,
    source_order: int,
    subspaces: Sequence[np.ndarray],
    prime: int,
) -> dict:
    """Exact lifted-connection profiles for rank-``r`` field quotients.

    Each RREF matrix in ``subspaces`` defines a map to ``F_p^r`` and hence a
    refinement with ``p^r`` vertices over every source color.  Bit masks encode
    which quotient values occur above each nonzero source class.
    """
    coordinates = np.asarray(coordinates, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    matrices = [np.asarray(matrix, dtype=np.int64) for matrix in subspaces]
    if not matrices:
        raise ValueError("at least one subspace is required")
    rank = matrices[0].shape[0]
    dimension = coordinates.shape[1]
    if rank < 1 or any(
        matrix.shape != (rank, dimension) for matrix in matrices
    ):
        raise ValueError("subspace matrices have inconsistent shapes")
    quotient_size = prime**rank
    if quotient_size > 16:
        raise ValueError(
            "bit-mask profile currently supports quotient size at most 16"
        )
    populated_classes = sorted(set(int(value) for value in class_ids))
    if len(populated_classes) != source_order - 1:
        raise ValueError(
            "the signed forbidden set must hit every nonzero source class"
        )
    groups = [np.flatnonzero(class_ids == value) for value in populated_classes]
    powers = np.asarray(
        [prime**coordinate for coordinate in range(rank)],
        dtype=np.int64,
    )
    connection_counts = np.empty(len(matrices), dtype=np.int64)
    minimum_coverages = np.empty(len(matrices), dtype=np.int16)
    full_mask = (1 << quotient_size) - 1
    for matrix_index, matrix in enumerate(matrices):
        residue_vectors = (coordinates @ matrix.T) % prime
        residue_codes = residue_vectors @ powers
        count = 0
        minimum = quotient_size
        for indices in groups:
            mask = int(
                np.bitwise_or.reduce(
                    np.left_shift(
                        np.uint64(1),
                        residue_codes[indices].astype(np.uint64),
                    )
                )
            )
            coverage = mask.bit_count()
            count += coverage
            minimum = min(minimum, coverage)
        connection_counts[matrix_index] = count
        minimum_coverages[matrix_index] = minimum
    complete = minimum_coverages == quotient_size
    histogram: dict[str, int] = {}
    for count in connection_counts:
        key = str(int(count))
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "prime": int(prime),
        "rank": int(rank),
        "quotient_size": int(quotient_size),
        "subspace_count": int(len(matrices)),
        "period_order": int(source_order * quotient_size),
        "expected_complete_multipartite_connections": int(
            (source_order - 1) * quotient_size
        ),
        "connection_count_minimum": int(connection_counts.min()),
        "connection_count_maximum": int(connection_counts.max()),
        "connection_count_histogram": histogram,
        "minimum_residues_per_source_class": int(
            minimum_coverages.min()
        ),
        "complete_multipartite_subspaces": int(complete.sum()),
        "incomplete_subspace_indices": np.flatnonzero(
            ~complete
        ).astype(int).tolist(),
        "_connection_counts": connection_counts,
        "_minimum_coverages": minimum_coverages,
        "_full_residue_mask": full_mask,
    }


def refinement_period(
    source_kernel: np.ndarray,
    character: Sequence[int],
    prime: int,
) -> np.ndarray:
    """Return column HNF of ``K ker(character mod prime)``."""
    source_kernel = np.asarray(source_kernel, dtype=np.int64)
    character = np.asarray(character, dtype=np.int64)
    lift = hnf_columns(
        kernel_basis([character], [prime], len(source_kernel))
    )
    return hnf_columns(source_kernel @ lift)


def subspace_refinement_period(
    source_kernel: np.ndarray,
    rows: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Return ``K ker(rows : Z^n -> F_p^r)`` in column HNF."""
    source_kernel = np.asarray(source_kernel, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    lift = hnf_columns(
        kernel_basis(
            [row for row in rows],
            [prime] * len(rows),
            len(source_kernel),
        )
    )
    return hnf_columns(source_kernel @ lift)


def source_fiber_columns(
    graph: CayleyConflictGraph,
    source_kernel: np.ndarray,
) -> list[tuple[int, ...]]:
    """Partition a refinement quotient by its source-kernel cosets."""
    source_kernel = np.asarray(source_kernel, dtype=np.int64)
    source_det, source_adjugate = quotient_map(source_kernel)
    period_representatives, _ = quotient_representatives(graph.period)
    groups: dict[tuple[int, ...], list[int]] = {}
    for vertex, representative in enumerate(period_representatives):
        key = quotient_key(representative, source_adjugate, source_det)
        groups.setdefault(key, []).append(vertex)
    columns = canonical_columns(groups.values())
    if sum(len(column) for column in columns) != graph.vertex_count:
        raise AssertionError("source fibers do not partition the quotient")
    if any(not graph.is_independent(column) for column in columns):
        raise AssertionError("a source fiber is not independent")
    return columns


def maximum_weight_independent_set(
    graph: CayleyConflictGraph,
    weights: Sequence[float] | None = None,
    *,
    force_vertex: int | None = None,
    time_limit: float = 60.0,
    mip_rel_gap: float = 0.0,
) -> dict:
    """Solve exact MWIS pricing with the HiGHS MIP interface."""
    if not graph.loop_free:
        return {
            "success": False,
            "optimal": False,
            "status": "looped quotient",
            "vertices": [],
        }
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if weights is None:
        objective_weights = np.ones(graph.vertex_count, dtype=np.float64)
    else:
        objective_weights = np.asarray(weights, dtype=np.float64)
        if objective_weights.shape != (graph.vertex_count,):
            raise ValueError("one weight is required per graph vertex")
        if np.any(~np.isfinite(objective_weights)):
            raise ValueError("weights must be finite")

    if force_vertex is None:
        candidates = list(range(graph.vertex_count))
    else:
        force_vertex = int(force_vertex)
        if not 0 <= force_vertex < graph.vertex_count:
            raise ValueError("forced vertex lies outside the graph")
        candidates = [
            vertex
            for vertex in range(graph.vertex_count)
            if vertex == force_vertex
            or not graph.adjacent(force_vertex, vertex)
        ]
    local_weights = objective_weights[candidates]
    edges = graph.edge_array(candidates)
    variable_count = len(candidates)

    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.ones(variable_count, dtype=np.float64)
    if force_vertex is not None:
        forced_local = candidates.index(force_vertex)
        lower[forced_local] = 1.0
    bounds = Bounds(lower, upper)
    constraints = None
    if len(edges):
        edge_rows = np.repeat(np.arange(len(edges), dtype=np.int64), 2)
        edge_columns = edges.reshape(-1)
        matrix = coo_matrix(
            (
                np.ones(2 * len(edges), dtype=np.float64),
                (edge_rows, edge_columns),
            ),
            shape=(len(edges), variable_count),
        ).tocsr()
        constraints = LinearConstraint(
            matrix,
            lb=np.full(len(edges), -np.inf),
            ub=np.ones(len(edges), dtype=np.float64),
        )

    started = time.perf_counter()
    if constraints is None:
        selected_local = np.flatnonzero(
            (local_weights > 0) | (lower > 0.5)
        )
        selected = [candidates[int(index)] for index in selected_local]
        return {
            "success": True,
            "optimal": True,
            "status": 0,
            "message": "edgeless forced subproblem solved directly",
            "vertices": selected,
            "weight": float(objective_weights[selected].sum()),
            "candidate_vertices": variable_count,
            "edge_constraints": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "solver": "closed form",
        }

    result = milp(
        c=-local_weights,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": True,
        },
    )
    selected: list[int] = []
    if result.x is not None:
        selected = [
            candidates[int(index)]
            for index in np.flatnonzero(result.x > 0.5)
        ]
    success = bool(selected) and graph.is_independent(selected)
    if force_vertex is not None and force_vertex not in selected:
        success = False
    return {
        "success": success,
        "optimal": bool(result.status == 0),
        "status": int(result.status),
        "message": str(result.message),
        "vertices": selected,
        "weight": (
            float(objective_weights[selected].sum()) if success else None
        ),
        "candidate_vertices": variable_count,
        "edge_constraints": int(len(edges)),
        "elapsed_seconds": time.perf_counter() - started,
        "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "mip_dual_bound": (
            float(result.mip_dual_bound)
            if getattr(result, "mip_dual_bound", None) is not None
            else None
        ),
        "solver": "HiGHS MIP via scipy.optimize.milp",
    }


def independent_set_target_cpsat(
    graph: CayleyConflictGraph,
    target_size: int,
    *,
    force_vertex: int = 0,
    time_limit: float = 60.0,
    workers: int = 8,
) -> dict:
    """Decide whether a Cayley graph has an independent set of target size.

    Vertex transitivity makes fixing one vertex lossless: translate any
    nonempty independent set so that one of its vertices is at the identity.
    The decision model is often substantially faster than optimizing the full
    independence number.
    """
    if not graph.loop_free:
        return {
            "status": "LOOPED",
            "feasible": False,
            "proven_infeasible": True,
            "vertices": [],
            "solver": "CP-SAT",
        }
    if target_size < 1 or target_size > graph.vertex_count:
        raise ValueError("target size lies outside the graph")
    if time_limit <= 0 or workers < 1:
        raise ValueError("time limit and worker count must be positive")
    force_vertex = int(force_vertex)
    candidates = [
        vertex
        for vertex in range(graph.vertex_count)
        if vertex == force_vertex
        or not graph.adjacent(force_vertex, vertex)
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
            "solver": "cardinality precheck",
        }
    edges = graph.edge_array(candidates)
    model = cp_model.CpModel()
    variables = [
        model.NewBoolVar(f"x_{vertex}") for vertex in candidates
    ]
    model.Add(variables[candidates.index(force_vertex)] == 1)
    for left, right in edges:
        model.Add(variables[int(left)] + variables[int(right)] <= 1)
    model.Add(sum(variables) >= int(target_size))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = False
    started = time.perf_counter()
    status = solver.Solve(model)
    elapsed = time.perf_counter() - started
    feasible = status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
    selected = (
        [
            candidates[index]
            for index, variable in enumerate(variables)
            if solver.Value(variable)
        ]
        if feasible
        else []
    )
    if feasible and (
        len(selected) < target_size or not graph.is_independent(selected)
    ):
        raise AssertionError("CP-SAT returned an invalid independent set")
    status_names = {
        cp_model.UNKNOWN: "UNKNOWN",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.OPTIMAL: "OPTIMAL",
    }
    return {
        "status": status_names.get(status, str(status)),
        "feasible": feasible,
        "proven_infeasible": status == cp_model.INFEASIBLE,
        "vertices": selected,
        "candidate_vertices": len(candidates),
        "edge_constraints": int(len(edges)),
        "elapsed_seconds": elapsed,
        "conflicts": int(solver.NumConflicts()),
        "branches": int(solver.NumBranches()),
        "wall_time": float(solver.WallTime()),
        "solver": "OR-Tools CP-SAT target-feasibility oracle",
    }


def translation_orbit(
    graph: CayleyConflictGraph,
    column: Sequence[int],
) -> list[tuple[int, ...]]:
    """Return all distinct Cayley translates of an independent set."""
    normalized = tuple(sorted(set(int(vertex) for vertex in column)))
    if not graph.is_independent(normalized):
        raise ValueError("column is not an independent set")
    return canonical_columns(
        graph.translate_set(normalized, shift)
        for shift in range(graph.vertex_count)
    )


def solve_set_cover_master(
    vertex_count: int,
    columns: Iterable[Sequence[int]],
    *,
    integer: bool,
    time_limit: float = 60.0,
) -> dict:
    """Solve an independent-set cover master with HiGHS."""
    normalized = canonical_columns(columns)
    row_indices: list[int] = []
    column_indices: list[int] = []
    for column_index, column in enumerate(normalized):
        for vertex in column:
            if not 0 <= vertex < vertex_count:
                raise ValueError("column contains a vertex outside the graph")
            row_indices.append(vertex)
            column_indices.append(column_index)
    incidence = csc_matrix(
        (
            np.ones(len(row_indices), dtype=np.float64),
            (row_indices, column_indices),
        ),
        shape=(vertex_count, len(normalized)),
    )
    if np.any(np.asarray(incidence.sum(axis=1)).ravel() == 0):
        raise ValueError("initial columns do not cover every vertex")
    objective = np.ones(len(normalized), dtype=np.float64)
    started = time.perf_counter()
    if integer:
        result = milp(
            c=objective,
            integrality=np.ones(len(normalized), dtype=np.int8),
            bounds=Bounds(
                np.zeros(len(normalized)), np.ones(len(normalized))
            ),
            constraints=LinearConstraint(
                incidence,
                lb=np.ones(vertex_count),
                ub=np.full(vertex_count, np.inf),
            ),
            options={"time_limit": float(time_limit), "presolve": True},
        )
        dual_weights = None
    else:
        result = linprog(
            objective,
            A_ub=-incidence,
            b_ub=-np.ones(vertex_count),
            bounds=(0.0, 1.0),
            method="highs",
            options={"time_limit": float(time_limit), "presolve": True},
        )
        dual_weights = (
            -np.asarray(result.ineqlin.marginals, dtype=np.float64)
            if result.success
            else None
        )
    selected = (
        np.flatnonzero(result.x > 0.5).astype(int).tolist()
        if result.x is not None
        else []
    )
    return {
        "success": bool(result.success),
        "optimal": bool(result.status == 0),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun) if result.fun is not None else None,
        "selected_column_indices": selected,
        "column_count": len(normalized),
        "elapsed_seconds": time.perf_counter() - started,
        "dual_weights": dual_weights,
        "columns": normalized,
        "solver": (
            "HiGHS MIP via scipy.optimize.milp"
            if integer
            else "HiGHS LP via scipy.optimize.linprog"
        ),
    }


def column_generation(
    graph: CayleyConflictGraph,
    initial_columns: Iterable[Sequence[int]],
    *,
    max_rounds: int = 20,
    pricing_time_limit: float = 60.0,
    reduced_cost_tolerance: float = 1e-8,
) -> dict:
    """Run a HiGHS set-cover LP with MWIS pricing and orbit insertion."""
    columns = canonical_columns(initial_columns)
    history: list[dict] = []
    converged = False
    for round_index in range(max_rounds):
        master = solve_set_cover_master(
            graph.vertex_count,
            columns,
            integer=False,
            time_limit=pricing_time_limit,
        )
        if not master["success"]:
            history.append(
                {
                    "round": round_index,
                    "master_status": master["status"],
                    "master_message": master["message"],
                }
            )
            break
        pricing = maximum_weight_independent_set(
            graph,
            master["dual_weights"],
            time_limit=pricing_time_limit,
        )
        record = {
            "round": round_index,
            "master_objective": master["objective"],
            "columns_before": len(columns),
            "pricing_success": pricing["success"],
            "pricing_optimal": pricing["optimal"],
            "pricing_weight": pricing.get("weight"),
            "pricing_size": len(pricing.get("vertices", [])),
            "pricing_status": pricing["status"],
        }
        if not pricing["success"] or not pricing["optimal"]:
            history.append(record)
            break
        if pricing["weight"] <= 1.0 + reduced_cost_tolerance:
            converged = True
            history.append(record)
            break
        existing = set(columns)
        additions = [
            translated
            for translated in translation_orbit(
                graph, pricing["vertices"]
            )
            if translated not in existing
        ]
        columns = canonical_columns([*columns, *additions])
        record["orbit_columns_added"] = len(additions)
        record["columns_after"] = len(columns)
        history.append(record)
        if not additions:
            break

    final_lp = solve_set_cover_master(
        graph.vertex_count,
        columns,
        integer=False,
        time_limit=pricing_time_limit,
    )
    return {
        "converged": converged,
        "rounds": history,
        "columns": final_lp["columns"],
        "column_count": final_lp["column_count"],
        "fractional_objective": final_lp["objective"],
        "final_lp_success": final_lp["success"],
        "final_lp_optimal": final_lp["optimal"],
    }


def _parse_primes(text: str) -> list[int]:
    try:
        values = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(values, list) or not values:
        raise argparse.ArgumentTypeError("expected a nonempty JSON list")
    primes = [int(value) for value in values]
    if any(value < 2 or not isprime(value) for value in primes):
        raise argparse.ArgumentTypeError("all entries must be primes")
    return primes


def _json_profile(profile: dict) -> dict:
    public = {
        key: value
        for key, value in profile.items()
        if not key.startswith("_") and key != "incomplete_character_indices"
    }
    incomplete = profile["incomplete_character_indices"]
    public["incomplete_character_count"] = len(incomplete)
    public["incomplete_character_indices_preview"] = incomplete[:40]
    return public


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primes", type=_parse_primes, default=[2, 3, 5, 7]
    )
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-characters", type=int, default=0)
    parser.add_argument(
        "--mwis-exceptions",
        type=int,
        default=0,
        help=(
            "per prime, solve this many non-complete refinements, ordered "
            "from the sparsest connection set; zero performs only the "
            "complete residue screen"
        ),
    )
    parser.add_argument(
        "--mwis-per-profile",
        type=int,
        default=0,
        help=(
            "also solve this many exceptional characters for every distinct "
            "(connection count, minimum residue coverage) profile"
        ),
    )
    parser.add_argument("--mwis-time-limit", type=float, default=60.0)
    parser.add_argument("--decision-time-limit", type=float, default=60.0)
    parser.add_argument("--decision-workers", type=int, default=8)
    parser.add_argument("--master-time-limit", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.target_colors < 1:
        parser.error("--target-colors must be positive")
    if (
        args.batch_size < 1
        or args.max_characters < 0
        or args.mwis_exceptions < 0
        or args.mwis_per_profile < 0
    ):
        parser.error("invalid character budget")
    if (
        args.mwis_time_limit <= 0
        or args.decision_time_limit <= 0
        or args.master_time_limit <= 0
        or args.decision_workers < 1
    ):
        parser.error("time limits must be positive")

    lattice, basis, diameter, _, source_kernel = load_preset("d6")
    forbidden, _, _ = _forbidden_with_weights(basis, diameter)
    coordinates, class_ids, source_order = source_extension_coordinates(
        source_kernel, forbidden
    )
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "all prime-index covers of the exact 343-color kernel; "
            "complete-multipartite residue screen followed, when needed, "
            "by HiGHS MWIS pricing and an independent-set cover master"
        ),
        "lattice": lattice,
        "dimension": len(source_kernel),
        "source_index": source_order,
        "source_kernel_basis_columns": source_kernel.astype(int).tolist(),
        "source_kernel_determinant": abs(exact_det(source_kernel)),
        "target_colors": args.target_colors,
        "forbidden_projective_pairs": len(forbidden),
        "signed_forbidden_vectors": len(coordinates),
        "settings": {
            "primes": args.primes,
            "batch_size": args.batch_size,
            "max_characters": args.max_characters,
            "mwis_exceptions": args.mwis_exceptions,
            "mwis_per_profile": args.mwis_per_profile,
            "mwis_time_limit": args.mwis_time_limit,
            "decision_time_limit": args.decision_time_limit,
            "decision_workers": args.decision_workers,
            "master_time_limit": args.master_time_limit,
        },
        "prime_screens": [],
        "exceptional_graphs": [],
        "new_coloring": None,
        "valid_numerical_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"source={source_order} signed-forbidden={len(coordinates)} "
        f"target={args.target_colors}",
        flush=True,
    )
    for prime in args.primes:
        characters = projective_forms(len(source_kernel), prime)
        if args.max_characters:
            characters = characters[: args.max_characters]
        profile = character_extension_profiles(
            coordinates,
            class_ids,
            source_order,
            characters,
            prime,
            batch_size=args.batch_size,
        )
        public_profile = _json_profile(profile)
        public_profile["necessary_independence_number_for_target"] = int(
            math.ceil((source_order * prime) / args.target_colors)
        )
        if (
            profile["complete_multipartite_characters"]
            == profile["character_count"]
        ):
            public_profile["exact_chromatic_number"] = source_order
            public_profile["conclusion"] = (
                "every refinement graph is complete source-index-partite; "
                "non-coset coloring cannot improve the source bound"
            )
        payload["prime_screens"].append(public_profile)
        save()
        print(
            f"p={prime} chars={profile['character_count']} "
            f"connections=[{profile['connection_count_minimum']},"
            f"{profile['connection_count_maximum']}] "
            f"complete={profile['complete_multipartite_characters']}",
            flush=True,
        )

        ranked_exceptions = sorted(
            profile["incomplete_character_indices"],
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        selected_exception_set = set(
            ranked_exceptions[: args.mwis_exceptions]
        )
        if args.mwis_per_profile:
            by_profile: dict[tuple[int, int], list[int]] = {}
            for character_index in ranked_exceptions:
                key = (
                    int(profile["_connection_counts"][character_index]),
                    int(profile["_minimum_coverages"][character_index]),
                )
                by_profile.setdefault(key, []).append(character_index)
            for indices in by_profile.values():
                selected_exception_set.update(
                    indices[: args.mwis_per_profile]
                )
        selected_exceptions = sorted(
            selected_exception_set,
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        public_profile["mwis_exception_budget"] = args.mwis_exceptions
        public_profile["mwis_per_profile"] = args.mwis_per_profile
        public_profile["mwis_exception_selected"] = len(selected_exceptions)
        public_profile["mwis_exception_unscreened"] = (
            len(ranked_exceptions) - len(selected_exceptions)
        )
        save()
        for character_index in selected_exceptions:
            character = characters[character_index]
            period = refinement_period(source_kernel, character, prime)
            graph = build_cayley_graph(period, forbidden)
            fibers = source_fiber_columns(graph, source_kernel)
            necessary_alpha = public_profile[
                "necessary_independence_number_for_target"
            ]
            decision = independent_set_target_cpsat(
                graph,
                necessary_alpha,
                force_vertex=0,
                time_limit=args.decision_time_limit,
                workers=args.decision_workers,
            )
            record: dict = {
                "prime": prime,
                "character_index": int(character_index),
                "character": character.astype(int).tolist(),
                "period_basis_columns": period.astype(int).tolist(),
                "period_index": graph.vertex_count,
                "connection_keys": len(graph.connection_keys),
                "loop_free": graph.loop_free,
                "source_fiber_count": len(fibers),
                "source_fiber_size": len(fibers[0]),
                "target_independent_set": decision,
            }
            if decision["proven_infeasible"]:
                record["independence_number"] = len(fibers[0])
                record["independence_number_proof"] = (
                    "source fiber gives the lower bound; CP-SAT proves that "
                    "the next cardinality is infeasible"
                )
                record["fractional_chromatic_number"] = float(source_order)
                payload["exceptional_graphs"].append(record)
                save()
                continue

            mwis = maximum_weight_independent_set(
                graph,
                force_vertex=0,
                time_limit=args.mwis_time_limit,
            )
            record["mwis"] = mwis
            if mwis["success"] and mwis["optimal"]:
                alpha = len(mwis["vertices"])
                record["independence_number"] = alpha
                record["fractional_chromatic_number"] = (
                    graph.vertex_count / alpha
                )
                if alpha >= public_profile[
                    "necessary_independence_number_for_target"
                ]:
                    generated = column_generation(
                        graph,
                        [*fibers, *[(vertex,) for vertex in range(graph.vertex_count)]],
                        pricing_time_limit=args.mwis_time_limit,
                    )
                    integer_master = solve_set_cover_master(
                        graph.vertex_count,
                        generated["columns"],
                        integer=True,
                        time_limit=args.master_time_limit,
                    )
                    record["column_generation"] = generated
                    record["integer_master"] = {
                        key: value
                        for key, value in integer_master.items()
                        if key not in {"columns", "dual_weights"}
                    }
                    if (
                        integer_master["success"]
                        and integer_master["objective"]
                        <= args.target_colors + 1e-7
                    ):
                        selected = integer_master[
                            "selected_column_indices"
                        ]
                        payload["new_coloring"] = {
                            "prime": prime,
                            "character": character.astype(int).tolist(),
                            "period_basis_columns": period.astype(int).tolist(),
                            "color_count": len(selected),
                            "colors": [
                                list(integer_master["columns"][index])
                                for index in selected
                            ],
                        }
            payload["exceptional_graphs"].append(record)
            save()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
