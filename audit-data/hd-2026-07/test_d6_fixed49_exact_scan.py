import numpy as np

from d6_fixed49_exact_scan import best_rows_for_objectives
from prime_radon import killed_mask
from threshold_multiblock_search import (
    build_groups,
    free_submodules_prime_power,
    killed_masks,
    row_module_key,
    source_option_index,
    source_option_distances,
)


def test_exact_pair_scan_matches_direct_scores() -> None:
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]], dtype=np.int64
    )
    ratios = np.asarray([0.4, 0.8, 0.9])
    weights = np.asarray([0.6, 0.2, 0.1])
    result = best_rows_for_objectives(forbidden, ratios, weights)
    for entry in result.values():
        mask = killed_mask(forbidden, entry["rows"], [2, 3])
        assert int(mask.sum()) == entry["killed"]
        if np.any(mask):
            assert float(ratios[mask].min()) == entry["minimum_ratio"]
            assert np.isclose(
                float(weights[mask].sum()), entry["weighted_loss"]
            )


def test_free_rank_two_submodules_modulo_four_are_canonical() -> None:
    options = np.asarray(
        list(free_submodules_prime_power(3, 4, 2))
    )
    # [3 choose 2]_2 * 2^(2*(3-2)) = 7 * 4.
    assert options.shape == (28, 2, 3)
    assert len({tuple(option.ravel()) for option in options}) == 28
    for option in options:
        assert np.linalg.matrix_rank(option % 2) == 2


def test_repeated_mod_four_group_masks_match_direct_arithmetic() -> None:
    group = build_groups(3, [4, 4])[0]
    forbidden = np.asarray(
        [[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.int64
    )
    masks = killed_masks(forbidden, group, batch_size=5)
    assert masks.shape == (28, 1)
    for index, rows in enumerate(group.options):
        direct = np.ones(len(forbidden), dtype=bool)
        for row in rows:
            direct &= (forbidden @ row) % 4 == 0
        encoded = np.unpackbits(
            masks[index].view(np.uint8), bitorder="little"
        )[: len(forbidden)]
        assert encoded.astype(bool).tolist() == direct.tolist()


def test_source_option_is_found_up_to_row_basis_change() -> None:
    group = build_groups(3, [4, 4])[0]
    option = group.options[17]
    changed_basis = np.asarray(
        [
            (option[0] + option[1]) % 4,
            option[1],
        ]
    )
    assert row_module_key(option, 4) == row_module_key(changed_basis, 4)
    assert source_option_index(group, changed_basis) == 17
    distances = source_option_distances(group, changed_basis)
    assert distances is not None
    assert int(np.argmin(distances)) == 17
    assert int(distances[17]) == 0
