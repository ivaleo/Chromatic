from pathlib import Path

import numpy as np

from d6_discrete_highs_cycle import kernel_key, select_probes


def test_kernel_key_is_stable_under_equivalent_column_generators() -> None:
    first = np.asarray([[2, 0], [0, 3]], dtype=np.int64)
    second = np.asarray([[2, 2], [0, 3]], dtype=np.int64)
    assert kernel_key(first) == kernel_key(second)


def test_probe_selection_combines_high_ratio_and_far_metrics() -> None:
    payload = {
        "branches": [
            {
                "label": "near",
                "voronoi_signature": "a",
                "oracle": {
                    "min_ratio": 0.99,
                    "parameters": [0.01, 0.0],
                },
            },
            {
                "label": "middle",
                "voronoi_signature": "b",
                "oracle": {
                    "min_ratio": 0.98,
                    "parameters": [0.2, 0.0],
                },
            },
            {
                "label": "far",
                "voronoi_signature": "c",
                "oracle": {
                    "min_ratio": 0.9,
                    "parameters": [1.0, 0.0],
                },
            },
        ]
    }
    probes = select_probes(
        [(Path("portfolio.json"), payload)],
        np.zeros(2),
        2,
    )
    assert {probe["label"] for probe in probes} == {"near", "far"}
