import pytest

from chromatic_research.campaigns.d6_fixed7_index_scan import (
    coordinate_search_needed,
    parse_targets,
    projective_pool_size,
)


def test_target_parser_and_pool_routing() -> None:
    assert parse_targets("[[47],[23,2],[9,5]]") == [
        [47],
        [23, 2],
        [9, 5],
    ]
    assert coordinate_search_needed(6, [47])
    assert coordinate_search_needed(6, [23, 2])
    assert not coordinate_search_needed(6, [9, 5])
    assert projective_pool_size(6, 2) == 63


def test_target_parser_rejects_non_primary_or_order_49() -> None:
    with pytest.raises(Exception):
        parse_targets("[[6]]")
    with pytest.raises(Exception):
        parse_targets("[[7,7]]")
