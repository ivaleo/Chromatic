import importlib.util
import sys
from pathlib import Path


from chromatic_research.campaigns import d6_torus_index_sweep as MODULE


def test_smooth_structures_cover_expected_indices():
    structures = dict(MODULE.smooth_prime_structures(344, 684))
    assert structures[350] == [2, 5, 5, 7]
    assert structures[360] == [2, 2, 2, 3, 3, 5]
    assert structures[448] == [2, 2, 2, 2, 2, 2, 7]
    assert structures[675] == [3, 3, 3, 5, 5]
    assert 343 not in structures
    assert 684 not in structures
