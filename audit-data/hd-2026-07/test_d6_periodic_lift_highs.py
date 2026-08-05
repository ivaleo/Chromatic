import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("d6_periodic_lift_highs.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_periodic_lift_highs", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_quotient_representatives_exact_count():
    columns = np.diag([2, 3]).astype(np.int64)
    representatives, index = MODULE.quotient_representatives(columns)
    assert len(representatives) == 6
    assert len(index) == 6


def test_solve_assignment_uses_only_allowed_edges():
    allowed = np.asarray(
        [
            [True, False, False],
            [False, False, True],
            [False, True, False],
        ]
    )
    assignment, metadata = MODULE.solve_assignment(
        allowed, np.random.default_rng(1)
    )
    assert metadata["success"]
    assert assignment.tolist() == [0, 2, 1]


def test_solve_assignment_detects_empty_row():
    allowed = np.asarray([[True, False], [False, False]])
    assignment, metadata = MODULE.solve_assignment(
        allowed, np.random.default_rng(1)
    )
    assert assignment is None
    assert not metadata["success"]
