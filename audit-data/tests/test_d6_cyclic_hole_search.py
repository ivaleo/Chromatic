import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


from chromatic_research.core import d6_cyclic_hole_search as MODULE


def test_cycle_matching_size_and_divisor_filter():
    assert MODULE.compatible_cycle_matching_size(10, 1) == 5
    assert MODULE.compatible_cycle_matching_size(10, 2) == 4
    assert MODULE.compatible_cycle_matching_size(9, 3) == 3
    assert MODULE.divisor_targets(10, 5) == [5, 1]
    assert MODULE.divisor_targets(9, 5) == [1]


def test_coordinate_scores_match_direct_recount():
    forbidden = np.asarray(
        [[1, 0], [1, 1], [2, 1], [3, 2]],
        dtype=np.int64,
    )
    row = np.asarray([2, 4], dtype=np.int64)
    scores = MODULE.coordinate_scores(
        forbidden,
        row,
        1,
        7,
        2,
    )
    direct = []
    for value in range(7):
        candidate = row.copy()
        candidate[1] = value
        direct.append(
            int(
                MODULE.target_violation_mask(
                    forbidden,
                    candidate,
                    7,
                    2,
                ).sum()
            )
        )
    assert scores.tolist() == direct


def test_small_descent_finds_a_primitive_avoiding_row():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    result = MODULE.cyclic_target_descent(
        forbidden,
        11,
        1,
        restarts=20,
        sweeps=10,
        top=4,
        seed=7,
    )
    assert result["success"]
    row = np.asarray(result["row"], dtype=np.int64)
    assert MODULE.primitive_cyclic_row(row, 11)
    assert not np.any(
        MODULE.target_violation_mask(forbidden, row, 11, 1)
    )


def test_pair_escape_never_misreports_its_score():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1], [2, 3]],
        dtype=np.int64,
    )
    row = np.asarray([1, 1], dtype=np.int64)
    candidate, score, weighted = MODULE.pair_coordinate_escape(
        forbidden,
        row,
        11,
        2,
        pair_top=4,
        pair_trials=1,
        rng=np.random.default_rng(19),
    )
    assert MODULE.primitive_cyclic_row(candidate, 11)
    assert score == int(
        MODULE.target_violation_mask(
            forbidden,
            candidate,
            11,
            2,
        ).sum()
    )
    assert weighted == float(score)


def test_cpsat_modular_model_finds_small_avoiding_row():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    result = MODULE.cyclic_target_cpsat(
        forbidden,
        11,
        1,
        time_limit=5.0,
        workers=1,
    )
    assert result["feasible"]
    assert MODULE.primitive_cyclic_row(result["row"], 11)
    assert not np.any(
        MODULE.target_violation_mask(
            forbidden,
            result["row"],
            11,
            1,
        )
    )


def test_highs_interval_mip_finds_small_primitive_avoiding_row():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    result = MODULE.cyclic_target_highs_interval_mip(
        forbidden,
        11,
        1,
        time_limit=5.0,
        fixed_row=[2, 0],
        free_coordinates=[1],
    )
    assert result["feasible"]
    assert result["row"][0] == 2
    assert result["fixed_coordinates"] == [0]
    assert MODULE.primitive_cyclic_row(result["row"], 11)
    assert not np.any(
        MODULE.target_violation_mask(
            forbidden,
            result["row"],
            11,
            1,
        )
    )


def test_metric_checkpoint_is_independently_rechecked(tmp_path):
    checkpoint = tmp_path / "metric.json"
    checkpoint.write_text(
        json.dumps(
            {
                "method": "test",
                "best": {
                    "basis": np.eye(2).tolist(),
                    "diameter": float(np.sqrt(2.0)),
                },
            }
        )
    )
    basis, diameter, payload = MODULE.load_metric_checkpoint(
        checkpoint,
        (2, 2),
    )
    assert np.array_equal(basis, np.eye(2))
    assert np.isclose(diameter, np.sqrt(2.0))
    assert payload["method"] == "test"

    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps(
            {
                "best": {
                    "basis": np.eye(2).tolist(),
                    "diameter": 1.0,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="diameter mismatch"):
        MODULE.load_metric_checkpoint(broken, (2, 2))
