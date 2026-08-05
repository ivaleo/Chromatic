import numpy as np
import pytest

from chromatic_research.campaigns.hybrid_multiblock_opt import (
    parse_rows,
    projective_hamming_distance,
    projective_trust_pool,
    trusted_pair_values,
    validated_seed_rows,
)


def test_parse_rows_accepts_integer_json():
    assert parse_rows("[[1, 2, 3], [0, 1, 0]]") == [
        [1, 2, 3],
        [0, 1, 0],
    ]


@pytest.mark.parametrize("text", ["[]", "[1, 2]", "[[]]", '[[1, "x"]]'])
def test_parse_rows_rejects_malformed_json(text):
    with pytest.raises(Exception):
        parse_rows(text)


def test_validated_seed_rows_normalizes_and_accepts_independent_blocks():
    rows = validated_seed_rows(
        [
            [20, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        [19, 3, 3, 2],
        4,
    )

    assert np.array_equal(rows[0], [1, 18, 0, 0])
    assert np.array_equal(rows[1], [1, 0, 0, 0])
    assert np.array_equal(rows[2], [0, 1, 0, 0])
    assert np.array_equal(rows[3], [0, 0, 1, 0])


def test_validated_seed_rows_rejects_nonprimitive_prime_power_row():
    with pytest.raises(ValueError, match="not primitive modulo 4"):
        validated_seed_rows(
            [[1, 0, 0], [2, 0, 0]],
            [17, 4],
            3,
        )


def test_validated_seed_rows_rejects_dependent_same_prime_blocks():
    with pytest.raises(ValueError, match="dependent modulo 3"):
        validated_seed_rows(
            [
                [1, 0, 0],
                [1, 1, 0],
                [2, 2, 0],
            ],
            [19, 3, 3],
            3,
        )


def test_validated_seed_rows_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected \\(3,\\)"):
        validated_seed_rows(
            [[1, 0], [0, 1, 0]],
            [17, 2],
            3,
        )


def test_projective_distance_quotients_out_unit_scaling():
    assert projective_hamming_distance([2, 4, 0], [1, 2, 0], 5) == 0
    assert projective_hamming_distance([2, 4, 1], [1, 2, 0], 5) == 1
    assert projective_hamming_distance([5, 2, 0], [1, 2, 0], 9) == 1


def test_projective_trust_pool_keeps_only_nearby_forms():
    pool = np.asarray(
        [
            [1, 2, 0],
            [2, 4, 0],
            [1, 2, 1],
            [1, 3, 1],
        ],
        dtype=np.int64,
    )
    nearby = projective_trust_pool(pool, [1, 2, 0], 5, 1)

    assert nearby.tolist() == [
        [1, 2, 0],
        [2, 4, 0],
        [1, 2, 1],
    ]


def test_trusted_pair_values_respect_projective_radius():
    candidates = trusted_pair_values(
        np.asarray([1, 2, 3]),
        0,
        1,
        5,
        np.asarray([1, 2, 3]),
        1,
    )

    assert len(candidates)
    for first, second in candidates:
        row = np.asarray([first, second, 3])
        assert projective_hamming_distance(row, [1, 2, 3], 5) <= 1
