import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("d6_torus_index_sweep.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_torus_index_sweep", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_smooth_structures_cover_expected_indices():
    structures = dict(MODULE.smooth_prime_structures(344, 684))
    assert structures[350] == [2, 5, 5, 7]
    assert structures[360] == [2, 2, 2, 3, 3, 5]
    assert structures[448] == [2, 2, 2, 2, 2, 2, 7]
    assert structures[675] == [3, 3, 3, 5, 5]
    assert 343 not in structures
    assert 684 not in structures
