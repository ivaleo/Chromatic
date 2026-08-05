import importlib.util
import sys
from pathlib import Path

import numpy as np


from chromatic_research.campaigns import d6_affine_metric_opt as MODULE


def test_bezout_cyclic_representatives_have_requested_residues():
    row = [6, 10, 15]
    assert MODULE.primitive_cyclic_row(row, 7)
    for residue in (0, 1, 6):
        representative = MODULE.cyclic_residue_representative(
            row,
            7,
            residue,
        )
        assert int(np.dot(row, representative)) % 7 == residue


def test_affine_enumeration_is_exact_on_square_lattice():
    basis = np.eye(2, dtype=np.float64)
    row = np.asarray([1, 0], dtype=np.int64)
    modulus = 4
    period = MODULE.hnf_columns(
        MODULE.kernel_basis([row], [modulus], 2)
    )
    cosets = MODULE.affine_coset_representatives(
        row,
        modulus,
        1,
    )
    coordinates = MODULE.affine_coordinates_within(
        basis,
        period,
        cosets,
        1.1,
    )
    assert {
        tuple(int(value) for value in coordinate)
        for coordinate in coordinates
    } == {(-1, 0), (0, -1)}
    for coordinate in coordinates:
        assert int(row @ coordinate) % modulus in {0, 1, 3}


def test_arbitrary_cyclic_residue_cosets_are_represented_exactly():
    row = np.asarray([2, 3], dtype=np.int64)
    representatives = MODULE.cyclic_residue_representatives(
        row,
        11,
        [-2, -1, 0, 1, 2],
    )
    assert {
        int(row @ representative) % 11
        for representative in representatives
    } == {0, 1, 2, 9, 10}


def test_checkpoint_cosets_preserve_explicit_transversal_differences():
    row = np.asarray([2, 3], dtype=np.int64)
    cosets, difference, block_size, residues = (
        MODULE.checkpoint_affine_cosets(
            {
                "target_difference": None,
                "block_size": 3,
                "difference_residues": [0, 2, 9, 2],
            },
            row,
            11,
        )
    )
    assert difference is None
    assert block_size == 3
    assert residues.tolist() == [0, 2, 9]
    assert {
        int(row @ representative) % 11
        for representative in cosets
    } == {0, 2, 9}


def test_checkpoint_cosets_remain_backward_compatible():
    row = np.asarray([2, 3], dtype=np.int64)
    cosets, difference, block_size, residues = (
        MODULE.checkpoint_affine_cosets(
            {"target_difference": 2},
            row,
            11,
        )
    )
    assert difference == 2
    assert block_size is None
    assert residues is None
    assert {
        int(row @ representative) % 11
        for representative in cosets
    } == {0, 2, 9}
