"""Инварианты стандартных решёток из core.lattices."""

import numpy as np
import pytest

from chromatic_research.core import lattices


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6])
def test_a_star_has_unit_covolume(dim):
    assert abs(np.linalg.det(lattices.Astar(dim))) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("dim", [4, 5, 6, 8])
def test_d_lattice_has_unit_covolume(dim):
    assert abs(np.linalg.det(lattices.D(dim))) == pytest.approx(1.0, rel=1e-12)


def test_a5_star_invariants_are_stable():
    """Опорные значения A5* — страховка для задачи о дублировании конструкции."""
    import combigeo

    basis = lattices.Astar(5)
    cell = combigeo.voronoi_cell(basis.tolist())
    shortest = combigeo.shortest_vector(basis.tolist())

    assert cell.diameter == pytest.approx(1.668064710953, abs=1e-11)
    assert float(np.linalg.norm(shortest)) == pytest.approx(1.092004686004, abs=1e-11)
