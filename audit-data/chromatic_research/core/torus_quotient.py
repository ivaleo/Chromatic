"""Периодические раскраски через факторгруппу решётки по ядру.

`signed_connection_images` — образы запрещённых векторов в факторгруппе,
`quotient_matching_coloring` — раскраска факторгруппы (MILP на паросочетаниях).
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def signed_connection_images(
    forbidden: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
) -> np.ndarray:
    """Return sorted distinct nonzero signed quotient images."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    rows_array = [np.asarray(row, dtype=np.int64) for row in rows]
    if len(rows_array) != len(moduli):
        raise ValueError("one modulus is required per quotient row")
    images = np.column_stack(
        [
            (forbidden @ row) % int(modulus)
            for row, modulus in zip(rows_array, moduli)
        ]
    )
    modulus_array = np.asarray(moduli, dtype=np.int64)
    signed = np.vstack([images, (-images) % modulus_array])
    unique = np.unique(signed, axis=0)
    nonzero = unique[np.any(unique != 0, axis=1)]
    return nonzero.astype(np.int64)


def quotient_matching_coloring(
    connection_images: np.ndarray,
    moduli: Sequence[int],
    target_colors: int,
    *,
    time_limit: float = 60.0,
) -> dict:
    """Merge compatible residue pairs using a HiGHS maximum-matching MIP."""
    moduli_array = np.asarray(moduli, dtype=np.int64)
    connection_set = {
        tuple(int(value) for value in row)
        for row in np.asarray(connection_images, dtype=np.int64)
    }
    elements = [
        tuple(int(value) for value in values)
        for values in np.ndindex(*(int(modulus) for modulus in moduli_array))
    ]
    vertex_count = len(elements)
    required_pairs = max(0, vertex_count - int(target_colors))
    if required_pairs == 0:
        return {
            "success": True,
            "optimal": True,
            "status": 0,
            "message": "target has at least one color per quotient element",
            "matching_size": 0,
            "required_pairs": 0,
            "color_count": vertex_count,
            "colors": [[list(element)] for element in elements],
            "solver": "closed form",
        }
    index = {element: number for number, element in enumerate(elements)}
    zero = tuple(0 for _ in moduli_array)
    missing = [
        element
        for element in elements
        if element != zero and element not in connection_set
    ]
    edges: set[tuple[int, int]] = set()
    for left, element in enumerate(elements):
        for difference in missing:
            right_element = tuple(
                (
                    element[coordinate] + difference[coordinate]
                )
                % int(moduli_array[coordinate])
                for coordinate in range(len(moduli_array))
            )
            right = index[right_element]
            if left < right:
                edges.add((left, right))
            elif right < left:
                edges.add((right, left))
    edge_list = sorted(edges)
    if len(edge_list) < required_pairs:
        return {
            "success": False,
            "optimal": True,
            "status": "edge-count precheck",
            "matching_size": 0,
            "required_pairs": required_pairs,
            "compatible_edges": len(edge_list),
            "solver": "closed form",
        }
    edge_array = np.asarray(edge_list, dtype=np.int64)
    columns = np.repeat(np.arange(len(edge_array), dtype=np.int64), 2)
    rows = edge_array.reshape(-1)
    incidence = coo_matrix(
        (
            np.ones(2 * len(edge_array), dtype=np.float64),
            (rows, columns),
        ),
        shape=(vertex_count, len(edge_array)),
    ).tocsr()
    started = time.perf_counter()
    result = milp(
        c=-np.ones(len(edge_array), dtype=np.float64),
        integrality=np.ones(len(edge_array), dtype=np.int8),
        bounds=Bounds(
            np.zeros(len(edge_array)), np.ones(len(edge_array))
        ),
        constraints=LinearConstraint(
            incidence,
            lb=np.full(vertex_count, -np.inf),
            ub=np.ones(vertex_count),
        ),
        options={"time_limit": float(time_limit), "presolve": True},
    )
    selected_edges = (
        edge_array[np.flatnonzero(result.x > 0.5)]
        if result.x is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    matching_size = len(selected_edges)
    success = matching_size >= required_pairs
    colors: list[list[list[int]]] = []
    if success:
        used: set[int] = set()
        for left, right in selected_edges[:required_pairs]:
            left = int(left)
            right = int(right)
            if left in used or right in used:
                raise AssertionError("HiGHS matching repeats a vertex")
            used.add(left)
            used.add(right)
            colors.append([list(elements[left]), list(elements[right])])
        colors.extend(
            [[list(element)]]
            for vertex, element in enumerate(elements)
            if vertex not in used
        )
        if len(colors) != vertex_count - required_pairs:
            raise AssertionError("decoded matching has the wrong color count")
        for color in colors:
            if len(color) == 2:
                difference = tuple(
                    (color[1][coordinate] - color[0][coordinate])
                    % int(moduli_array[coordinate])
                    for coordinate in range(len(moduli_array))
                )
                if difference in connection_set:
                    raise AssertionError("matching uses a conflicting pair")
    return {
        "success": success,
        "optimal": bool(result.status == 0),
        "status": int(result.status),
        "message": str(result.message),
        "matching_size": matching_size,
        "required_pairs": required_pairs,
        "compatible_edges": len(edge_array),
        "color_count": len(colors) if success else None,
        "colors": colors if success else None,
        "elapsed_seconds": time.perf_counter() - started,
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "solver": "HiGHS maximum-matching MIP via scipy.optimize.milp",
    }
