import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("d6_torus_fourier_lp.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_torus_fourier_lp", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_real_fourier_basis_partitions_all_characters():
    graph = MODULE.build_cayley_graph(
        np.asarray([[5]], dtype=np.int64),
        np.asarray([[1]], dtype=np.int64),
    )
    basis = MODULE.real_fourier_basis(graph.period)
    assert basis.orbit_count == 3
    assert basis.multiplicities.tolist() == [1, 2, 2]
    values = basis.evaluate(graph.keys)
    assert values.shape == (5, 3)
    assert np.allclose(values[0], [1.0, 2.0, 2.0])


def test_cycle_five_has_sqrt_five_hoffman_and_theta_bounds():
    graph = MODULE.build_cayley_graph(
        np.asarray([[5]], dtype=np.int64),
        np.asarray([[1]], dtype=np.int64),
    )
    hoffman = MODULE.hoffman_ratio_bound(graph)
    theta = MODULE.abelian_theta_lp(graph, time_limit=5.0)
    theta_prime = MODULE.abelian_theta_lp(
        graph,
        nonnegative=True,
        time_limit=5.0,
    )
    assert abs(hoffman["upper_bound"] - math.sqrt(5.0)) < 1e-8
    assert theta["optimal"]
    assert theta_prime["optimal"]
    assert abs(theta["upper_bound"] - math.sqrt(5.0)) < 1e-8
    assert abs(theta_prime["upper_bound"] - math.sqrt(5.0)) < 1e-8
    assert theta["maximum_edge_residual"] < 1e-8
    assert theta_prime["minimum_function_value"] > -1e-8


def test_complete_tripartite_graph_is_certified_at_alpha_two():
    graph = MODULE.build_cayley_graph(
        np.asarray([[6]], dtype=np.int64),
        np.asarray([[1], [2]], dtype=np.int64),
    )
    hoffman = MODULE.hoffman_ratio_bound(graph)
    theta = MODULE.abelian_theta_lp(graph, time_limit=5.0)
    theta_prime = MODULE.abelian_theta_lp(
        graph,
        nonnegative=True,
        time_limit=5.0,
    )
    assert abs(hoffman["upper_bound"] - 2.0) < 1e-8
    assert theta["optimal"] and theta_prime["optimal"]
    assert abs(theta["upper_bound"] - 2.0) < 1e-8
    assert abs(theta_prime["upper_bound"] - 2.0) < 1e-8
