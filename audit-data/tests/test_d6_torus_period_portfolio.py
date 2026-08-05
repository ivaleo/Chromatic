import importlib.util
import sys
from pathlib import Path

import numpy as np


from chromatic_research.campaigns import d6_torus_period_portfolio as MODULE


def test_signed_connection_images_are_exact_and_symmetric():
    forbidden = np.asarray([[1], [2]], dtype=np.int64)
    images = MODULE.signed_connection_images(
        forbidden,
        [np.asarray([1], dtype=np.int64)],
        [7],
    )
    assert {tuple(row) for row in images.tolist()} == {
        (1,),
        (2,),
        (5,),
        (6,),
    }


def test_nested_sublattice_uses_exact_arithmetic():
    parent = np.asarray([[2, 0], [0, 1]], dtype=np.int64)
    nested = np.asarray([[4, 0], [0, 1]], dtype=np.int64)
    outside = np.asarray([[1, 0], [0, 2]], dtype=np.int64)
    assert MODULE.is_nested_sublattice(nested, parent)
    assert not MODULE.is_nested_sublattice(outside, parent)


def test_direct_product_independent_set_decision_on_cycle_five():
    connections = np.asarray([[1], [4]], dtype=np.int64)
    size_two = MODULE.quotient_independent_set_target(
        connections, [5], 2, workers=1
    )
    size_three = MODULE.quotient_independent_set_target(
        connections, [5], 3, workers=1
    )
    assert size_two["feasible"]
    assert size_three["proven_infeasible"]
    coloring = MODULE.quotient_matching_coloring(
        connections, [5], 3, time_limit=5.0
    )
    assert coloring["success"]
    assert coloring["optimal"]
    assert coloring["matching_size"] == 2
    assert coloring["color_count"] == 3
