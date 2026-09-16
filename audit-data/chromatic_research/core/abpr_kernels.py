"""Опубликованные ядра раскрасок Армана–Бондаренко–Прымака–Радченко (АБПР).

C5 — ℝ⁵ (A₅*, индекс 140), C9 — ℝ⁹ (A₉*, индекс 17253; в базисе минимальных
весов и в фундаментальном), C2401_ROWS — ℝ⁸ (E₈, индекс 2401), а также переход
между базисами A_n* и геометрия E₈. Ядро E₇* (C7) — в `e7_abpr`.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

import combigeo


C5 = np.array([
    [-2, 1,-2,-1, 0],
    [-3, 1, 0,-1,-2],
    [ 0, 1, 1,-1,-3],
    [-2, 0,-2, 2,-2],
    [-2,-2, 0, 0,-2]], dtype=np.int64)


def minimal_to_fundamental_transform(n):
    """Coordinate transform from the minimal-weight to fundamental-weight basis.

    Two coordinate conventions for ``A_n^*`` occur in the audit scripts:

    * the columns of ``M_Anstar`` are ``-(n+1) q_i``, where
      ``q_i=e_i-(1/(n+1))1`` are minimal weights;
    * :func:`An_star_ambient` and :func:`lattices.Astar` use the fundamental
      weights ``omega_i=q_1+...+q_i``.

    If a row coordinate ``x_q`` is written in the ``q`` basis, then

        x_omega = x_q U,   q_i = omega_i - omega_{i-1},

    where ``U`` is returned here.  Consequently a column-basis matrix for a
    sublattice transforms as ``C_omega = U.T @ C_q``.
    """
    if n < 1:
        raise ValueError("dimension must be positive")
    transform = np.eye(n, dtype=np.int64)
    for index in range(1, n):
        transform[index, index - 1] = -1
    return transform


def kernel_minimal_to_fundamental(kernel):
    """Convert an ``A_n^*`` sublattice column basis between the conventions."""
    matrix = np.asarray(kernel, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kernel must be a square matrix")
    return minimal_to_fundamental_transform(len(matrix)).T @ matrix


C9 = np.asarray(
    [
        [0, 0, -3, 1, 0, 0, -1, 1, 0],
        [1, 0, -3, 1, 1, 0, -1, 4, 1],
        [0, 0, -2, 1, 0, -1, -1, 1, 3],
        [0, 0, -3, 4, 0, 0, -1, 1, 0],
        [0, 3, -3, 1, 0, 0, -1, 1, 0],
        [3, 0, -3, 1, 0, 0, 2, 1, 0],
        [0, 0, -4, 2, 0, 3, -1, 2, 0],
        [0, 0, -3, 1, 3, 0, -1, 1, 0],
        [-1, 0, -3, 1, -1, 1, 1, 1, -1],
    ],
    dtype=np.int64,
)


# ABPR's C9 is expressed in the minimal-weight coordinate basis used by
# ``M_Anstar``.  ``parent_geometry("A9*")`` uses the fundamental-weight basis
# from ``lattices.Astar``.  The distinction matters for modular characters:
# final geometric validation made the old campaign safe, but its supposedly
# frozen F_3 block did not belong to the published coloring.  Convert the
# kernel before taking annihilators.
C9_FUNDAMENTAL = kernel_minimal_to_fundamental(C9)


BINT = np.asarray(
    [
        [3, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 2, 0, 0, 0, 1, 0],
        [1, 0, 2, 0, 0, 0, 0, 1],
    ],
    dtype=np.int64,
)


C2401_ROWS = np.asarray(
    [
        [1, 3, 0, 0, 0, 0, 0, 0],
        [-1, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 3, 0, 0, 0, 0],
        [0, 0, -1, 4, 0, 0, 0, 0],
        [-1, 1, -1, 1, 3, 1, 0, 0],
        [0, 1, 0, 1, -1, 2, 0, 0],
        [-1, 1, -2, 2, 0, 0, 3, 1],
        [0, 1, 0, 2, 0, 0, -1, 2],
    ],
    dtype=np.int64,
)


def e8_geometry() -> tuple[
    np.ndarray, float, list[tuple[list[float], float]]
]:
    transform = np.zeros((8, 8), dtype=np.float64)
    for index in range(4):
        transform[2 * index, 2 * index : 2 * index + 2] = [1.0, 0.0]
        transform[2 * index + 1, 2 * index : 2 * index + 2] = [
            -0.5,
            math.sqrt(3.0) / 2.0,
        ]
    basis = BINT @ transform
    shortest = float(
        np.linalg.norm(combigeo.shortest_vector(basis.tolist()))
    )
    diameter = math.sqrt(2.0) * shortest
    facets = combigeo.relevant_facets(basis.tolist())
    return basis, diameter, facets
