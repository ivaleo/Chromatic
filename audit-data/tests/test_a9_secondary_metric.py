import json
from pathlib import Path

import numpy as np
from sympy import Matrix

from chromatic_research.campaigns.a9_secondary_metric import (
    campaign_records,
    exact_lll_kernel_rows,
    load_base_metric,
    obtuse_superbase_check,
)
from chromatic_research.core.lattices import Astar
from chromatic_research.campaigns.permutohedral_cover import radii_for_orders
from chromatic_research.campaigns.verify_a9_secondary import direct_order_vertex


def test_lazy_campaign_candidate_is_normalized():
    source = {
        "results": [
            {
                "best": {
                    "kernel_basis_columns": np.diag(
                        [17251] + [1] * 8
                    ).tolist(),
                    "kernel_smith": [1] * 8 + [17251],
                    "separation": {
                        "minimum_distance_ratio": 0.696,
                        "conflicts": [
                            {
                                "coordinate": [1] + [0] * 8,
                                "distance_ratio": 0.696,
                            }
                        ],
                    },
                }
            }
        ]
    }

    records, container = campaign_records(source)

    assert container == "results[].best"
    assert records[0]["distance_ratio"] == 0.696
    assert records[0]["conflicts"][0]["coordinate"][0] == 1
    assert records[0]["smith"][-1] == 17251
    assert records[0]["_campaign_result_index"] == 0


def test_exact_lll_rows_preserve_kernel_index():
    kernel = np.eye(9, dtype=np.int64)
    kernel[0, 0] = 17251
    kernel[0, 1:] = np.arange(8) + 1000

    rows = exact_lll_kernel_rows(kernel)

    assert abs(int(Matrix(rows.tolist()).det())) == 17251


def test_lazy_source_metric_is_inherited(tmp_path: Path):
    metric = tmp_path / "metric.json"
    basis = Astar(9)
    metric.write_text(json.dumps({"best": {"basis": basis.tolist()}}))
    campaign = tmp_path / "campaign.json"
    campaign.write_text("{}")

    loaded, reference = load_base_metric(
        campaign,
        {"source_metric": str(metric)},
        None,
    )

    assert np.allclose(loaded, basis)
    assert reference == str(metric)


def test_obtuse_superbase_detects_secondary_cone_crossing():
    basis = Astar(3)
    inside = obtuse_superbase_check(basis)
    deformation = np.diag([np.exp(0.5), np.exp(-0.25), np.exp(-0.25)])
    outside = obtuse_superbase_check(basis @ deformation)

    assert inside["feasible"]
    assert inside["maximum_off_diagonal_inner_product"] < 0
    assert not outside["feasible"]
    assert outside["maximum_off_diagonal_inner_product"] > 0


def test_direct_vertex_solves_agree_with_specialized_order_radii():
    basis = Astar(4)
    orders = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [4, 2, 0, 3, 1],
            [1, 3, 4, 0, 2],
        ],
        dtype=np.uint8,
    )
    specialized = radii_for_orders(basis, orders)
    direct = np.asarray(
        [direct_order_vertex(basis, order)[0] for order in orders]
    )

    assert np.allclose(specialized, direct, rtol=2e-12, atol=2e-12)
