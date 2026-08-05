import importlib.util
import sys
from pathlib import Path

import numpy as np


from chromatic_research.campaigns import d6_group_transversal_lift as MODULE


def test_quotient_subtraction_table_is_exact_for_cyclic_four():
    _, _, inverse, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    assert inverse.tolist() == [0, 3, 2, 1]
    assert subtraction.tolist() == [
        [0, 3, 2, 1],
        [1, 0, 3, 2],
        [2, 1, 0, 3],
        [3, 2, 1, 0],
    ]


def test_connection_table_adds_central_inverses():
    table = MODULE.connection_table(
        [1],
        [1],
        [3],
        3,
        4,
    )
    assert np.argwhere(table).tolist() == [[1, 1], [2, 3]]


def test_sampled_projective_forms_are_unique_and_normalized():
    forms = MODULE.sampled_projective_forms(3, 5, 31, 17)
    assert forms.shape == (31, 3)
    assert len({tuple(int(value) for value in row) for row in forms}) == 31
    for row in forms:
        first = next(int(value) for value in row if value)
        assert first == 1


def test_highs_phase_cegar_finds_and_verifies_small_transversal():
    _, _, _, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    connections = np.zeros((3, 4), dtype=bool)
    connections[1, 0] = True
    connections[2, 0] = True
    result = MODULE.solve_phase_highs_cegar(
        connections,
        subtraction,
        time_limit=5.0,
        max_rounds=100,
        seed=7,
    )
    assert result["feasible"]
    assert not MODULE.transversal_conflicts(
        result["phases"],
        connections,
        subtraction,
    )


def test_phase_coordinate_descent_finds_small_transversal():
    _, _, _, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    connections = np.zeros((3, 4), dtype=bool)
    connections[1, 0] = True
    connections[2, 0] = True
    result = MODULE.phase_coordinate_descent(
        connections,
        subtraction,
        restarts=4,
        sweeps=8,
        seed=19,
    )
    assert result["feasible"]
    assert result["exact_conflicts"] == 0


def test_canonical_highs_matchings_find_small_coloring():
    _, _, _, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    connections = np.zeros((3, 4), dtype=bool)
    connections[1, 0] = True
    connections[2, 0] = True
    matrices = MODULE.canonical_conflict_matrices(
        connections,
        subtraction,
    )
    assert matrices[(0, 1)].shape == (4, 4)
    assert np.all(matrices[(0, 1)].sum(axis=1) == 1)
    result = MODULE.canonical_matching_coloring(
        connections,
        subtraction,
        trials=3,
        seed=23,
    )
    assert result["feasible"]
    assert len(result["colors"]) == 4
    assert all(len(color) == 3 for color in result["colors"])


def test_highs_phase_cegar_detects_empty_first_layer():
    _, _, _, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    connections = np.zeros((3, 4), dtype=bool)
    connections[1, :] = True
    result = MODULE.solve_phase_highs_cegar(
        connections,
        subtraction,
        time_limit=5.0,
        max_rounds=100,
        seed=7,
    )
    assert result["status"] == "PAIR_INFEASIBLE"
    assert result["proven_infeasible"]


def test_cpsat_phase_solver_finds_and_verifies_small_transversal():
    _, _, _, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    connections = np.zeros((3, 4), dtype=bool)
    connections[1, 0] = True
    connections[2, 0] = True
    result = MODULE.solve_phase_cpsat(
        connections,
        subtraction,
        time_limit=5.0,
        workers=1,
        seed=11,
    )
    assert result["feasible"]
    assert not MODULE.transversal_conflicts(
        result["phases"],
        connections,
        subtraction,
    )


def test_cpsat_phase_solver_proves_empty_pair_table():
    _, _, _, subtraction = MODULE.quotient_group_tables(
        np.asarray([[4]], dtype=np.int64)
    )
    connections = np.zeros((3, 4), dtype=bool)
    connections[1, 0] = True
    connections[2, 0] = True
    connections[1, 1:] = True
    result = MODULE.solve_phase_cpsat(
        connections,
        subtraction,
        time_limit=5.0,
        workers=1,
        seed=11,
    )
    assert result["status"] == "PAIR_INFEASIBLE"
    assert result["proven_infeasible"]
