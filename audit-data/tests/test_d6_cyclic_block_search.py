import importlib.util
import sys
from pathlib import Path

import numpy as np


from chromatic_research.campaigns import d6_cyclic_block_search as MODULE


def test_block_violation_mask_uses_exact_consecutive_difference_set():
    forbidden = np.arange(15, dtype=np.int64)[:, None]
    mask = MODULE.block_violation_mask(
        forbidden,
        [1],
        15,
        3,
    )
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 13, 14]


def test_block_coordinate_scores_match_direct_recount():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1], [2, 3]],
        dtype=np.int64,
    )
    row = np.asarray([4, 7], dtype=np.int64)
    scores = MODULE.block_coordinate_scores(
        forbidden,
        row,
        1,
        15,
        3,
    )
    direct = []
    for value in range(15):
        candidate = row.copy()
        candidate[1] = value
        direct.append(
            int(
                MODULE.block_violation_mask(
                    forbidden,
                    candidate,
                    15,
                    3,
                ).sum()
            )
        )
    assert scores.tolist() == direct


def test_descent_and_highs_find_small_block_avoiding_rows():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    descent = MODULE.cyclic_block_descent(
        forbidden,
        15,
        3,
        restarts=20,
        sweeps=10,
        top=6,
        seed=31,
    )
    assert descent["success"]
    assert not np.any(
        MODULE.block_violation_mask(
            forbidden,
            descent["row"],
            15,
            3,
        )
    )

    highs = MODULE.cyclic_target_highs_interval_mip(
        forbidden,
        15,
        1,
        avoid_radius=2,
        time_limit=5.0,
    )
    assert highs["feasible"]
    assert not np.any(
        MODULE.block_violation_mask(
            forbidden,
            highs["row"],
            15,
            3,
        )
    )


def test_composite_primitive_row_without_unit_coordinate_has_kernel():
    modulus = 1026
    row = np.asarray(
        [321, 928, 600, 420, 482, 672],
        dtype=np.int64,
    )
    assert MODULE.primitive_cyclic_row(row, modulus)
    assert all(np.gcd(int(value), modulus) > 1 for value in row)
    kernel = MODULE.hnf_columns(
        MODULE.kernel_basis([row], [modulus], len(row))
    )
    assert abs(MODULE.exact_det(kernel)) == modulus
    assert np.all((row @ kernel) % modulus == 0)
