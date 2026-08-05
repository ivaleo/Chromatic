import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("d6_torus_prime_exhaustive.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_torus_prime_exhaustive", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _adjacency_bitsets(vertex_count, edges):
    adjacency = [0] * vertex_count
    for left, right in edges:
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left
    return adjacency


def test_bitset_clique_decision_on_cycle_five():
    adjacency = _adjacency_bitsets(
        5,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    )
    two = MODULE.bitset_clique_target(
        adjacency,
        2,
        time_limit=5.0,
    )
    three = MODULE.bitset_clique_target(
        adjacency,
        3,
        time_limit=5.0,
    )
    assert two["feasible"]
    assert len(two["vertices"]) == 2
    assert three["proven_infeasible"]


def test_prime_extension_model_matches_cycle_six_target():
    source_kernel = np.asarray([[2]], dtype=np.int64)
    forbidden = np.asarray([[1]], dtype=np.int64)
    coordinates, class_ids, source_order = (
        MODULE.source_extension_coordinates(source_kernel, forbidden)
    )
    model = MODULE.build_prime_extension_model(source_kernel)
    character = np.asarray([1], dtype=np.int64)
    masks = MODULE.prime_connection_masks(
        coordinates,
        class_ids,
        source_order,
        character,
        3,
    )
    assert masks.tolist() == [0, 1 | (1 << 2)]
    graph = MODULE.prime_extension_compatibility_graph(
        model,
        masks,
        character,
        3,
    )
    assert len(graph["adjacency"]) == 4

    alpha_three = MODULE.prime_extension_target(
        model,
        coordinates,
        class_ids,
        character,
        3,
        3,
        time_limit=5.0,
    )
    alpha_four = MODULE.prime_extension_target(
        model,
        coordinates,
        class_ids,
        character,
        3,
        4,
        time_limit=5.0,
    )
    assert alpha_three["feasible"]
    assert alpha_four["proven_infeasible"]
