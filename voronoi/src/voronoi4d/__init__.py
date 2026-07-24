"""voronoi4d — поиск хроматического числа R^4 через разбиения Вороного 4-мерных решёток.

Основные компоненты:
- VoronoiPolyhedra — построение разбиения Вороного и центрального многогранника;
- dist_to_s        — расстояние от точки до центрального многогранника;
- find_optimal     — поиск оптимальных подрешёток по диапазону определителей;
- lll_reduce       — LLL-приведение базиса (fpylll или чистый python);
- generate_grids / generate_integer_grids — генерация базисов решёток.
"""

from importlib.metadata import PackageNotFoundError, version

from .distances import (
    check_dist,
    dist_to_s,
    find_faces_from_nearest_vertices,
)
from .enumeration import lattice_points_within, shortest_vector
from .factorization import (
    compute_factorizations,
    ordered_factorizations,
    pad_lists_with_ones,
)
from .grids import (
    canonical_form_by_rows,
    check_grid,
    generate_grids,
    generate_integer_grids,
    generate_random_matrix,
    min_max_det,
    normalize_rows,
)
from .io import plot_results, save_result
from .lll import HAS_FPYLLL, gram_schmidt, lll_reduce, lll_reduce_python
from .polyhedra import Edge2D, Face2D, Face3D, VoronoiPolyhedra
from .search import find_optimal, lattice_points_no_central_symmetry

# единый источник версии — поле version в pyproject.toml (читаем из метаданных)
try:
    __version__ = version("voronoi4d")
except PackageNotFoundError:
    # пакет не установлен (запуск из исходников без установки)
    __version__ = "0.0.0+unknown"

__all__ = [
    "Edge2D",
    "Face2D",
    "Face3D",
    "HAS_FPYLLL",
    "VoronoiPolyhedra",
    "canonical_form_by_rows",
    "check_dist",
    "check_grid",
    "compute_factorizations",
    "dist_to_s",
    "find_faces_from_nearest_vertices",
    "find_optimal",
    "generate_grids",
    "generate_integer_grids",
    "generate_random_matrix",
    "gram_schmidt",
    "lattice_points_no_central_symmetry",
    "lattice_points_within",
    "lll_reduce",
    "lll_reduce_python",
    "ordered_factorizations",
    "shortest_vector",
    "min_max_det",
    "normalize_rows",
    "pad_lists_with_ones",
    "plot_results",
    "save_result",
]
