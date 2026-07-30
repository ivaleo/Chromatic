import math

import numpy as np

import combigeo
from lattices import Astar
from metric_deform import exhaustive_covering_radius
from permutohedral_cover import (
    covering_radius,
    permutohedral_facet_coordinates,
)


def test_covering_radius_matches_generic_oracle_in_small_dimensions():
    for dimension in (2, 3, 4):
        basis = Astar(dimension)
        specialized, vertex_count, _ = covering_radius(basis)
        generic, generic_vertex_count = exhaustive_covering_radius(
            combigeo.relevant_facets(basis.tolist())
        )
        assert math.isclose(specialized, generic, rel_tol=2e-12)
        assert vertex_count == math.factorial(dimension + 1)
        assert generic_vertex_count == vertex_count


def test_subset_facets_match_every_a4star_relevant_vector():
    basis = Astar(4)
    expected = {
        tuple(row)
        for row in permutohedral_facet_coordinates(4).tolist()
    }
    observed = {
        tuple(
            np.rint(np.linalg.solve(basis.T, np.asarray(normal)))
            .astype(int)
            .tolist()
        )
        for normal, _ in combigeo.relevant_facets(basis.tolist())
    }
    assert observed == expected


def test_a9star_known_diameter_without_qhull_vertex_enumeration():
    radius, vertex_count, witness = covering_radius(
        Astar(9), with_witness=True
    )
    assert math.isclose(2.0 * radius, 2.0644887731710035, rel_tol=2e-12)
    assert vertex_count == math.factorial(10)
    assert witness is not None
    assert sorted(witness["permutation"]) == list(range(10))
