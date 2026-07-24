import numpy as np
import pytest

from voronoi4d import VoronoiPolyhedra


@pytest.fixture(scope="session")
def vor():
    """Разбиение Вороного для единичной решётки Z^4 (общая фикстура)."""
    grid = np.eye(4, dtype=float)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)
    return vor
