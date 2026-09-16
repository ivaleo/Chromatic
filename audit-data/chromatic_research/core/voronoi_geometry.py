"""Ячейка Вороного решётки: фасеты, вершины, активные фасеты, радиус и сигнатура L-типа."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import HalfspaceIntersection

import combigeo


@dataclass
class VoronoiGeometry:
    facet_coordinates: np.ndarray
    vertices: np.ndarray
    active_facets: list[list[int]]
    radius: float
    signature: str


def _vertex_signature(
    facet_coordinates: np.ndarray,
    active_facets: Sequence[Sequence[int]],
) -> str:
    """Order-independent hash of all Delone simplices at the origin."""
    simplices = [
        tuple(
            sorted(
                tuple(int(value) for value in facet_coordinates[index])
                for index in active
            )
        )
        for active in active_facets
    ]
    encoded = json.dumps(
        sorted(simplices), separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def voronoi_geometry(basis: np.ndarray) -> VoronoiGeometry:
    basis = np.asarray(basis, dtype=np.float64)
    facets = combigeo.relevant_facets(basis.tolist())
    normals = np.asarray([facet[0] for facet in facets], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    offsets = np.asarray([facet[1] for facet in facets], dtype=np.float64)
    halfspaces = np.column_stack(
        (normals / lengths[:, None], -offsets)
    )
    hull = HalfspaceIntersection(
        halfspaces, np.zeros(len(basis)), qhull_options="Qx"
    )
    inverse = np.linalg.inv(basis)
    raw_coordinates = normals @ inverse
    facet_coordinates = np.rint(raw_coordinates).astype(np.int64)
    residual = float(
        np.max(np.abs(raw_coordinates - facet_coordinates))
    )
    if residual > 2e-7:
        raise RuntimeError(
            f"could not recover integer facet coordinates: {residual:g}"
        )
    active_facets = [
        [int(index) for index in active]
        for active in hull.dual_facets
    ]
    if any(len(active) != len(basis) for active in active_facets):
        raise RuntimeError("source Voronoi cell is not simple")
    vertices = np.asarray(hull.intersections, dtype=np.float64)
    return VoronoiGeometry(
        facet_coordinates=facet_coordinates,
        vertices=vertices,
        active_facets=active_facets,
        radius=float(np.linalg.norm(vertices, axis=1).max()),
        signature=_vertex_signature(facet_coordinates, active_facets),
    )
