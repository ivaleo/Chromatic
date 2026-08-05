"""Векторизованный dist_to_s обязан совпадать с каскадным на всех решётках."""

import numpy as np
import pytest

from voronoi4d import VoronoiPolyhedra, dist_to_s
from voronoi4d.distances import dist_to_s_cascade

LATTICES = {
    "Z4": np.eye(4),
    "D4": np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float),
    "skew": np.array([[1.3, 0.2, 0, 0], [0.1, 1.1, 0.3, 0],
                      [0, 0.2, 1.2, 0.1], [0.1, 0, 0.2, 0.9]], float),
}


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_vectorized_matches_cascade(name):
    vor = VoronoiPolyhedra(LATTICES[name])
    vor.build(verbose=False)
    rng = np.random.default_rng(7)

    for _ in range(50):
        point = rng.normal(size=4) * 1.5
        fast = dist_to_s(vor, point, vor.max_len, early_stop=0.0)
        slow = dist_to_s_cascade(vor, point, vor.max_len, early_stop=0.0)
        assert fast == pytest.approx(slow, abs=1e-12), f"{name}: {point}"


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_points_inside_the_cell_give_zero(name):
    vor = VoronoiPolyhedra(LATTICES[name])
    vor.build(verbose=False)
    assert dist_to_s(vor, np.zeros(4), vor.max_len, early_stop=0.0) == 0.0
