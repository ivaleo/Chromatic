from pathlib import Path

import numpy as np
import pytest

from chromatic_research.campaigns import d6_fixed7_campaign as campaign
from chromatic_research.core.prime_radon import image_size, rank_mod


def test_default_structures_are_all_primary_types_of_residual_order() -> None:
    parsed = campaign.parse_structures(
        "[[16,3],[8,2,3],[4,4,3],[4,2,2,3],[2,2,2,2,3]]"
    )
    assert parsed == campaign.DEFAULT_STRUCTURES


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "[[12,4]]",
        "[[8,3]]",
        "[[6,8]]",
    ],
)
def test_structure_validation_rejects_wrong_primary_factorization(
    text: str,
) -> None:
    with pytest.raises(Exception):
        campaign.parse_structures(text)


def test_source_rows_are_exact_independent_index_343_characters() -> None:
    rows = campaign.load_e6_source_rows(campaign.DEFAULT_SOURCE)
    assert len(rows) == 3
    assert rank_mod(np.asarray(rows), 7) == 3
    assert image_size(rows, [7, 7, 7], 6) == 343


def test_residual_mask_and_full_rows() -> None:
    forbidden = np.asarray(
        [[1, 0], [0, 1], [7, 1], [2, 7]], dtype=np.int64
    )
    fixed = np.asarray([1, 0], dtype=np.int64)
    mask = campaign.residual_mask(forbidden, fixed)
    assert mask.tolist() == [False, True, True, False]
    rows = campaign.full_rows(fixed, [np.asarray([0, 1])])
    assert [row.tolist() for row in rows] == [[1, 0], [0, 1]]


def test_cyclic_coordinate_search_checks_exact_index_and_weights() -> None:
    forbidden = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    initial = [np.asarray([1, 1]), np.asarray([1, 1])]
    search = campaign.CoordinatePrimarySearch(
        forbidden, [16, 3], seed=17
    )
    count = search.run(
        restarts=0,
        max_sweeps=3,
        top=4,
        progress_every=0,
        initial_rows=initial,
    )
    assert count.killed == 0
    assert count.image_index == 48

    weighted = search.run_weighted(
        np.asarray([2.0, 3.0]),
        restarts=0,
        max_sweeps=3,
        top=4,
        progress_every=0,
        initial_rows=initial,
    )
    assert weighted.killed == 0
    assert weighted.weighted_loss == 0.0
    assert weighted.image_index == 48
