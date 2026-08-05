import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("d6_highs_index_refine.py")
SPEC = importlib.util.spec_from_file_location(
    "d6_highs_index_refine", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_indices_and_starts():
    assert MODULE.parse_indices("329,322") == [329, 322]
    assert MODULE.parse_starts("e6-generic,reference") == [
        "e6-generic",
        "reference",
    ]
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_indices("1")
    with pytest.raises(argparse.ArgumentTypeError):
        MODULE.parse_starts("e6-generic,e6-generic")


def test_select_portfolio_deduplicates_and_groups(monkeypatch):
    monkeypatch.setattr(
        MODULE,
        "hnf_columns",
        lambda kernel: np.asarray(kernel, dtype=np.int64),
    )
    monkeypatch.setattr(
        MODULE,
        "kernel_key",
        lambda kernel: tuple(np.asarray(kernel).flat),
    )
    campaign = {
        "results": [
            {
                "image_index": 329,
                "label": "old",
                "minimum_conflict_ratio": 0.8,
                "kernel_basis_columns": [[329]],
            },
            {
                "image_index": 329,
                "label": "new",
                "minimum_conflict_ratio": 0.9,
                "kernel_basis_columns": [[329]],
            },
            {
                "image_index": 322,
                "label": "other",
                "minimum_conflict_ratio": 0.7,
                "kernel_basis_columns": [[322]],
            },
        ]
    }
    selected = MODULE.select_portfolio(
        [(Path("campaign.json"), campaign)],
        [329, 322],
        per_index=1,
    )
    assert [item["label"] for item in selected] == ["new", "other"]
