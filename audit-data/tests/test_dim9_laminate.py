import json
import math

import numpy as np
from sympy import Matrix

from chromatic_research.campaigns.dim9_laminate import base_characters, base_geometry
from chromatic_research.campaigns.e8_neighbor_search import C2401_ROWS
from chromatic_research.core.lamination import (
    Lamination, enumerate_upto, kernel_rows, min_separation, unit_facets,
)
from chromatic_research.core.minkowski import brunn_minkowski_bound, unit_ball_volume
from chromatic_research.paths import results_path


def test_base_characters_annihilate_the_eisenstein_sublattice():
    rows = base_characters()
    assert rows.shape == (4, 8)
    assert np.all((C2401_ROWS @ rows.T) % 7 == 0)


def test_kernel_basis_is_returned_as_rows_and_has_the_stated_index():
    """kernel_basis returns generators as COLUMNS; kernel_rows must transpose."""
    rows = base_characters()
    kernel = kernel_rows(rows, [7] * 4, [0, 0, 0, 0], 5)
    assert abs(int(Matrix(kernel.tolist()).det())) == 2401 * 5


def test_enumerate_upto_matches_a_brute_force_box():
    basis = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2]])
    bound = 2.5
    found = enumerate_upto(basis, bound)
    brute = [
        c @ basis
        for c in ((i, j) for i in range(-6, 7) for j in range(-6, 7))
        if any(c) and np.linalg.norm(np.asarray(c, float) @ basis) <= bound + 1e-12
    ]
    assert len(found) == len(brute)


def test_safe_diameter_bounds_the_measured_one():
    """(P1): nearest layer + nearest base point gives diam <= sqrt(diam_base^2+t^2)."""
    basis, radius, hole, _ = base_geometry()
    lam = Lamination(basis, radius, hole, 0.69)
    assert lam.measured_diameter(n_dirs=120) <= lam.safe_diameter + 1e-9


def test_stored_index_12005_colouring_is_valid_against_the_rigorous_diameter():
    config = json.loads(results_path("dim9_laminate_m5.json").read_text())
    basis, radius, _, rows = base_geometry()
    lam = Lamination(basis, radius, np.asarray(config["offset"], float),
                     float(config["height"]))
    kernel = kernel_rows(rows, [7] * 4, config["glue"], config["modulus"])
    assert abs(int(Matrix(kernel.tolist()).det())) == 12005
    separation = min_separation(kernel @ lam.basis, lam.safe_diameter,
                                unit_facets(lam.basis))
    # the binding vectors are horizontal, so the separation is D_{E8} = sqrt(7)
    assert separation == 0 or abs(separation - math.sqrt(7)) < 1e-9
    assert separation / lam.safe_diameter > 1.0395


def test_minkowski_closed_form_is_below_the_sampled_volume_for_z2():
    # V0 = unit square, R = sqrt(2)/2; Steiner gives vol(V0+RB) = 1+4R+pi R^2
    radius = math.sqrt(2) / 2
    exact = 1 + 4 * radius + math.pi * radius**2
    assert brunn_minkowski_bound(2, radius) <= exact + 1e-9
    assert unit_ball_volume(2) == math.pi
