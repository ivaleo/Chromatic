import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("d6_highs_kernel_race.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_highs_kernel_race", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_survivors_requires_strict_decrease():
    assert MODULE.parse_survivors("64,24,8") == [64, 24, 8]
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_survivors("8,8")
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_survivors("8,12")


def test_complete_ratio_prefers_complete_oracle():
    record = {
        "minimum_conflict_ratio": 0.8,
        "complete_separation": {"minimum_distance_ratio": 0.9},
    }
    assert MODULE.complete_ratio(record) == 0.9


def test_deduplicate_portfolio_keeps_best_probe_and_excludes_source(
    monkeypatch,
):
    source = np.eye(2, dtype=np.int64)
    other = np.asarray([[2, 0], [0, 1]], dtype=np.int64)
    monkeypatch.setattr(
        MODULE,
        "kernel_key",
        lambda kernel: tuple(np.asarray(kernel).flat),
    )
    monkeypatch.setattr(
        MODULE,
        "hnf_columns",
        lambda kernel: np.asarray(kernel, dtype=np.int64),
    )
    cycle = {
        "probes": [
            {"label": "a", "parameters": [0.0]},
            {"label": "b", "parameters": [1.0]},
        ],
        "discrete_candidates": [
            {
                "kernel_basis_columns": source.tolist(),
                "probe_index": 0,
                "minimum_conflict_ratio": 1.0,
            },
            {
                "kernel_basis_columns": other.tolist(),
                "probe_index": 0,
                "minimum_conflict_ratio": 0.7,
                "complete_separation": {
                    "minimum_distance_ratio": 0.75
                },
                "label": "old",
            },
            {
                "kernel_basis_columns": other.tolist(),
                "probe_index": 1,
                "minimum_conflict_ratio": 0.8,
                "complete_separation": {
                    "minimum_distance_ratio": 0.85
                },
                "label": "new",
            },
        ],
    }
    result = MODULE.deduplicate_portfolio(cycle, source)
    assert len(result) == 1
    assert result[0]["discrete_label"] == "new"
    assert result[0]["probe_ratio"] == 0.85
    assert result[0]["parameters"] == [1.0]
