import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("d6_highs_kernel_refine.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_highs_kernel_refine", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _state(key, score, trajectory):
    return {
        "key": [key],
        "current": {"min_ratio": score},
        "rounds": [
            {"optimization": {"best": {"min_ratio": value}}}
            for value in trajectory
        ],
    }


def test_last_gain():
    assert MODULE.last_gain(_state(1, 0.9, [0.7, 0.8, 0.9])) == (
        0.9 - 0.8
    )
    assert MODULE.last_gain(_state(1, 0.9, [0.9])) == 0.0


def test_select_candidates_unites_leaders_and_risers():
    leader = _state(1, 0.99, [0.98, 0.99])
    riser = _state(2, 0.90, [0.70, 0.90])
    middle = _state(3, 0.95, [0.94, 0.95])
    selected = MODULE.select_candidates(
        {"portfolio": [middle, riser, leader]},
        top=1,
        rising=1,
    )
    assert selected == [leader, riser]
