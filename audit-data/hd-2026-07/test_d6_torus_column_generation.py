import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("d6_torus_column_generation.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_torus_column_generation", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cycle_five_column_generation_reaches_fractional_and_integer_values():
    graph = MODULE.build_cayley_graph(
        np.asarray([[5]], dtype=np.int64),
        np.asarray([[1]], dtype=np.int64),
    )
    assert graph.loop_free
    assert graph.vertex_count == 5
    assert graph.degree == 2

    mwis = MODULE.maximum_weight_independent_set(
        graph, force_vertex=0, time_limit=5.0
    )
    assert mwis["success"]
    assert mwis["optimal"]
    assert len(mwis["vertices"]) == 2
    target_two = MODULE.independent_set_target_cpsat(
        graph, 2, time_limit=5.0, workers=1
    )
    target_three = MODULE.independent_set_target_cpsat(
        graph, 3, time_limit=5.0, workers=1
    )
    assert target_two["feasible"]
    assert target_three["proven_infeasible"]

    generated = MODULE.column_generation(
        graph,
        [(vertex,) for vertex in range(5)],
        max_rounds=5,
        pricing_time_limit=5.0,
    )
    assert generated["converged"]
    assert abs(generated["fractional_objective"] - 2.5) < 1e-8

    integer = MODULE.solve_set_cover_master(
        5, generated["columns"], integer=True, time_limit=5.0
    )
    assert integer["success"]
    assert integer["optimal"]
    assert abs(integer["objective"] - 3.0) < 1e-8


def test_complete_tripartite_refinement_is_detected():
    graph = MODULE.build_cayley_graph(
        np.asarray([[6]], dtype=np.int64),
        np.asarray([[1], [2]], dtype=np.int64),
    )
    assert graph.loop_free
    assert graph.degree == 4
    mwis = MODULE.maximum_weight_independent_set(
        graph, force_vertex=0, time_limit=5.0
    )
    assert mwis["success"]
    assert mwis["optimal"]
    assert set(mwis["vertices"]) == {0, 3}

    fibers = [(0, 3), (1, 4), (2, 5)]
    lp = MODULE.solve_set_cover_master(
        6, fibers, integer=False, time_limit=5.0
    )
    mip = MODULE.solve_set_cover_master(
        6, fibers, integer=True, time_limit=5.0
    )
    assert lp["success"] and mip["success"]
    assert abs(lp["objective"] - 3.0) < 1e-8
    assert abs(mip["objective"] - 3.0) < 1e-8


def test_source_extension_profile_matches_direct_quotient_graph():
    source = np.asarray([[2]], dtype=np.int64)
    forbidden = np.asarray([[1]], dtype=np.int64)
    coordinates, class_ids, source_order = (
        MODULE.source_extension_coordinates(source, forbidden)
    )
    characters = np.asarray([[1]], dtype=np.int64)
    profile = MODULE.character_extension_profiles(
        coordinates,
        class_ids,
        source_order,
        characters,
        3,
        batch_size=1,
    )
    period = MODULE.refinement_period(source, characters[0], 3)
    graph = MODULE.build_cayley_graph(period, forbidden)
    assert graph.vertex_count == 6
    assert profile["_connection_counts"].tolist() == [
        len(graph.connection_keys)
    ]
    assert profile["complete_multipartite_characters"] == 0


def test_rank_two_profile_matches_direct_quotient_graph():
    source = np.asarray([[2, 0], [0, 1]], dtype=np.int64)
    forbidden = np.asarray(
        [[1, 0], [1, 1], [3, 0], [3, 1]], dtype=np.int64
    )
    coordinates, class_ids, source_order = (
        MODULE.source_extension_coordinates(source, forbidden)
    )
    rows = np.eye(2, dtype=np.int64)
    profile = MODULE.subspace_extension_profiles(
        coordinates,
        class_ids,
        source_order,
        [rows],
        2,
    )
    period = MODULE.subspace_refinement_period(source, rows, 2)
    graph = MODULE.build_cayley_graph(period, forbidden)
    assert graph.vertex_count == 8
    assert profile["_connection_counts"].tolist() == [
        len(graph.connection_keys)
    ]


def test_looped_period_is_rejected_by_pricing():
    graph = MODULE.build_cayley_graph(
        np.asarray([[2]], dtype=np.int64),
        np.asarray([[2]], dtype=np.int64),
    )
    assert not graph.loop_free
    result = MODULE.maximum_weight_independent_set(graph)
    assert not result["success"]
