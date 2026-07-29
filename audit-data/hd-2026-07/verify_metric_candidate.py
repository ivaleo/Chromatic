"""Independent numerical/exact audit of a deformed lattice coloring.

The input is ``metric_deform.py`` output.  Its floating Gram matrix is rounded
to an explicitly rational matrix Q = Q_integer / denominator.  The verifier:

1. checks positive definiteness by exact Sylvester minors;
2. checks the modular image, integer kernel, determinant, HNF, and Smith form;
3. enumerates every Voronoi vertex with Qhull and solves every corresponding
   vertex system over the rationals to obtain the exact covering radius;
4. LLL-reduces the coloring sublattice, proves an a priori coefficient box for
   every vector with |v| < 2 diam(V), and enumerates that box exactly;
5. gives an exact KKT projection certificate for every such vector;
6. independently rebuilds the parent forbidden set and checks that none of
   its vectors lies in the modular kernel.

The only non-rational combinatorial oracle is Qhull's exhaustive list of simple
Voronoi vertices/facet incidences.  Every value attached to that incidence list
is recomputed with SymPy rational arithmetic.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import HalfspaceIntersection
from sympy import Matrix, Rational

import combigeo
from prime_radon import (
    hnf_columns,
    image_size,
    kernel_basis,
    killed_mask,
    smith_diagonal,
)


def fraction_text(value) -> str:
    value = Rational(value)
    return f"{value.p}/{value.q}" if value.q != 1 else str(value.p)


def exact_positive_definite(integer_gram: np.ndarray) -> list[int]:
    matrix = Matrix(integer_gram.tolist())
    minors = [
        int(matrix[:size, :size].det())
        for size in range(1, matrix.rows + 1)
    ]
    if any(value <= 0 for value in minors):
        raise AssertionError(f"Gram matrix is not positive definite: {minors}")
    return minors


def voronoi_data(
    basis: np.ndarray,
) -> tuple[
    list[tuple[list[float], float]],
    np.ndarray,
    HalfspaceIntersection,
    float,
]:
    facets = combigeo.relevant_facets(basis.tolist())
    normals = np.asarray([facet[0] for facet in facets], dtype=np.float64)
    unit = normals / np.linalg.norm(normals, axis=1)[:, None]
    offsets = np.asarray([facet[1] for facet in facets], dtype=np.float64)
    halfspaces = np.column_stack((unit, -offsets))
    hull = HalfspaceIntersection(
        halfspaces, np.zeros(basis.shape[0]), qhull_options="Qx"
    )
    radii_squared = np.einsum(
        "ij,ij->i", hull.intersections, hull.intersections
    )
    return facets, normals, hull, float(radii_squared.max())


def exact_vertex_radius(
    integer_gram: np.ndarray,
    denominator: int,
    facet_coordinates: np.ndarray,
    hull: HalfspaceIntersection,
    *,
    progress_every: int = 5_000,
) -> tuple[Rational, list[list[int]], int]:
    """Recompute every Qhull vertex norm with exact rational arithmetic."""
    gram = Matrix(integer_gram.tolist())
    best: Rational | None = None
    best_active: list[list[int]] = []
    singular = 0
    start = time.perf_counter()
    for number, active_ids in enumerate(hull.dual_facets, start=1):
        active = Matrix(facet_coordinates[active_ids].tolist())
        system = active * gram
        right = Matrix(
            [
                (active.row(index) * gram * active.row(index).T)[0] / 2
                for index in range(active.rows)
            ]
        )
        try:
            vertex = system.inv() * right
        except Exception:
            singular += 1
            continue
        norm_squared = Rational(
            (vertex.T * gram * vertex)[0]
        ) / denominator
        if best is None or norm_squared > best:
            best = norm_squared
            best_active = [list(map(int, active_ids))]
        elif norm_squared == best:
            best_active.append(list(map(int, active_ids)))
        if progress_every and number % progress_every == 0:
            print(
                f"  exact vertices {number}/{len(hull.dual_facets)} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
    if best is None or singular:
        raise AssertionError(
            f"exact vertex enumeration failed: best={best}, singular={singular}"
        )
    return best, best_active, singular


def lll_coordinate_basis(
    basis: np.ndarray, kernel_columns: np.ndarray
) -> np.ndarray:
    sub_basis = kernel_columns.T @ basis
    reduced_physical = np.asarray(
        combigeo.lll_reduce(sub_basis.tolist()), dtype=np.float64
    )
    coordinates = np.rint(
        reduced_physical @ np.linalg.inv(basis)
    ).astype(np.int64)
    if not np.allclose(coordinates @ basis, reduced_physical, atol=1e-7):
        raise AssertionError("could not recover integer LLL coordinates")
    if not np.array_equal(hnf_columns(coordinates.T), kernel_columns):
        raise AssertionError("LLL basis does not generate the stored kernel")
    return coordinates


def exact_short_sublattice_vectors(
    integer_gram: np.ndarray,
    denominator: int,
    reduced_rows: np.ndarray,
    covering_radius_squared: Rational,
) -> tuple[list[np.ndarray], list[Rational]]:
    """Prove a coefficient box and enumerate all |v| < 4 R exactly."""
    gram = Matrix(integer_gram.tolist())
    reduced = Matrix(reduced_rows.tolist())
    reduced_gram_integer = reduced * gram * reduced.T
    reduced_gram_inverse = reduced_gram_integer.inv() * denominator
    norm_bound = 16 * covering_radius_squared
    coordinate_bounds = [
        Rational(norm_bound * reduced_gram_inverse[index, index])
        for index in range(reduced.rows)
    ]
    # z_i^2 <= ||z U||_Q^2 (S^{-1})_ii.  A bound below 4 proves
    # |z_i| <= 1 for every vector strictly inside the length cutoff.
    if any(value >= 4 for value in coordinate_bounds):
        raise AssertionError(
            "the [-1,1]^n coefficient certificate failed: "
            + ", ".join(fraction_text(value) for value in coordinate_bounds)
        )

    vectors: list[np.ndarray] = []
    norms: list[Rational] = []
    for coefficient in itertools.product((-1, 0, 1), repeat=reduced.rows):
        if not any(coefficient):
            continue
        coordinate = np.asarray(coefficient, dtype=np.int64) @ reduced_rows
        row = Matrix([coordinate.astype(int).tolist()])
        norm_squared = Rational(
            (row * gram * row.T)[0], denominator
        )
        if norm_squared < norm_bound:
            vectors.append(coordinate.astype(np.int64))
            norms.append(norm_squared)
    return vectors, coordinate_bounds


def exact_projection_certificate(
    coordinate: np.ndarray,
    basis: np.ndarray,
    facets: Sequence[tuple[Sequence[float], float]],
    facet_coordinates: np.ndarray,
    integer_gram: np.ndarray,
    denominator: int,
) -> dict:
    """Find active facets numerically, then verify the KKT system exactly."""
    normals = np.asarray([facet[0] for facet in facets], dtype=np.float64)
    unit = normals / np.linalg.norm(normals, axis=1)[:, None]
    offsets = np.asarray([facet[1] for facet in facets], dtype=np.float64)
    point = 0.5 * (coordinate @ basis)
    result = minimize(
        lambda x: 0.5 * float(np.dot(x - point, x - point)),
        np.zeros(basis.shape[0]),
        jac=lambda x: x - point,
        constraints=[LinearConstraint(unit, -np.inf, offsets)],
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 2_000},
    )
    if not result.success:
        raise AssertionError(f"SLSQP active-set oracle failed: {result.message}")
    slack = offsets - unit @ result.x
    candidates = np.flatnonzero(slack < 2e-7).tolist()
    if not candidates:
        raise AssertionError("projection has no numerically active facet")

    gram = Matrix(integer_gram.tolist())
    point_exact = Matrix([coordinate.astype(int).tolist()]).T / 2
    all_facets = Matrix(facet_coordinates.tolist())
    for subset_size in range(1, min(7, len(candidates)) + 1):
        for active_ids in itertools.combinations(candidates, subset_size):
            active = Matrix(facet_coordinates[list(active_ids)].tolist())
            if active.rank() != subset_size:
                continue
            middle = active * gram * active.T
            right_boundary = Matrix(
                [
                    (active.row(index) * gram * active.row(index).T)[0] / 2
                    for index in range(subset_size)
                ]
            )
            multipliers = middle.inv() * (
                active * gram * point_exact - right_boundary
            )
            if any(value < 0 for value in multipliers):
                continue
            projection = point_exact - active.T * multipliers
            feasible = True
            for index in range(all_facets.rows):
                facet = all_facets.row(index)
                if (
                    (facet * gram * projection)[0]
                    > (facet * gram * facet.T)[0] / 2
                ):
                    feasible = False
                    break
            if not feasible:
                continue
            difference = point_exact - projection
            distance_squared = (
                Rational(4 * (difference.T * gram * difference)[0])
                / denominator
            )
            dykstra = 2.0 * combigeo.dist_to_halfspaces(
                point.tolist(), facets
            )
            return {
                "coordinate": coordinate.astype(int).tolist(),
                "distance_squared": fraction_text(distance_squared),
                "distance_squared_float": float(distance_squared),
                "active_facets": [
                    facet_coordinates[index].astype(int).tolist()
                    for index in active_ids
                ],
                "multipliers": [
                    fraction_text(value) for value in multipliers
                ],
                "dykstra_distance": float(dykstra),
                "exact_distance": math.sqrt(float(distance_squared)),
                "solver_difference": float(
                    dykstra - math.sqrt(float(distance_squared))
                ),
            }
    raise AssertionError(
        f"no exact KKT certificate for vector {coordinate.tolist()}, "
        f"active candidates={candidates}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--denominator", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "metric_deform_e7_1323_certificate.json",
    )
    args = parser.parse_args(argv)
    if args.denominator <= 0:
        parser.error("denominator must be positive")

    source = json.loads(args.candidate.read_text())
    source_record = source["source_record"]
    float_gram = np.asarray(source["best"]["gram"], dtype=np.float64)
    integer_gram = np.rint(float_gram * args.denominator).astype(np.int64)
    rational_gram = integer_gram.astype(np.float64) / args.denominator
    leading_minors = exact_positive_definite(integer_gram)
    basis = np.linalg.cholesky(rational_gram)

    rows = [
        np.asarray(row, dtype=np.int64)
        for row in source_record["rows"]
    ]
    moduli = [int(value) for value in source_record["moduli"]]
    kernel = hnf_columns(kernel_basis(rows, moduli, 7))
    exact_image = image_size(rows, moduli, 7)
    exact_determinant = abs(int(Matrix(kernel.tolist()).det()))
    if exact_image != exact_determinant or exact_image != 1323:
        raise AssertionError(
            f"index mismatch: image={exact_image}, det={exact_determinant}"
        )
    print(
        f"rational Gram 1/{args.denominator}: positive definite; "
        f"kernel image=det={exact_image}",
        flush=True,
    )

    facets, facet_normals, hull, numerical_radius_squared = voronoi_data(
        basis
    )
    facet_coordinates = np.rint(
        facet_normals @ np.linalg.inv(basis)
    ).astype(np.int64)
    if not np.allclose(
        facet_coordinates @ basis, facet_normals, atol=1e-7
    ):
        raise AssertionError("could not recover integer facet coordinates")
    if any(len(active) != 7 for active in hull.dual_facets):
        raise AssertionError("Voronoi polytope is not simple")
    print(
        f"Voronoi: facets={len(facets)} vertices={len(hull.intersections)}; "
        "starting exact vertex audit",
        flush=True,
    )
    exact_radius_squared, farthest_active, singular = exact_vertex_radius(
        integer_gram,
        args.denominator,
        facet_coordinates,
        hull,
    )
    radius = math.sqrt(float(exact_radius_squared))
    diameter = 2.0 * radius
    if abs(float(exact_radius_squared) - numerical_radius_squared) > 1e-9:
        raise AssertionError("exact and numerical covering radii disagree")
    print(
        f"covering radius^2={fraction_text(exact_radius_squared)} "
        f"({float(exact_radius_squared):.12f}); "
        f"farthest vertices={len(farthest_active)}",
        flush=True,
    )

    reduced_rows = lll_coordinate_basis(basis, kernel)
    short_vectors, coefficient_bounds = exact_short_sublattice_vectors(
        integer_gram,
        args.denominator,
        reduced_rows,
        exact_radius_squared,
    )
    # Cross-check the exact coefficient-box enumeration against the independent
    # C++ Fincke-Pohst enumeration.
    cpp_vectors = combigeo._vectors_near(
        (kernel.T @ basis).tolist(),
        [0.0] * 7,
        2.0 * diameter + 1e-8,
    )
    cpp_coordinates = {
        tuple(
            np.rint(np.asarray(vector) @ np.linalg.inv(basis))
            .astype(np.int64)
            .tolist()
        )
        for vector in cpp_vectors
        if np.linalg.norm(vector) > 1e-10
    }
    exact_coordinates = {tuple(vector.tolist()) for vector in short_vectors}
    if cpp_coordinates != exact_coordinates:
        raise AssertionError(
            f"short-vector enumerators disagree: "
            f"exact-only={exact_coordinates-cpp_coordinates}, "
            f"cpp-only={cpp_coordinates-exact_coordinates}"
        )
    print(
        f"short coloring vectors: exact={len(short_vectors)} "
        f"C++={len(cpp_coordinates)}; starting exact KKT audit",
        flush=True,
    )

    projection_certificates = [
        exact_projection_certificate(
            coordinate,
            basis,
            facets,
            facet_coordinates,
            integer_gram,
            args.denominator,
        )
        for coordinate in short_vectors
    ]
    exact_distances_squared = [
        Rational(certificate["distance_squared"])
        for certificate in projection_certificates
    ]
    minimum_distance_squared = min(exact_distances_squared)
    diameter_squared = 4 * exact_radius_squared
    squared_margin = minimum_distance_squared - diameter_squared
    if squared_margin <= 0:
        raise AssertionError(
            f"color separation failed: squared margin={squared_margin}"
        )
    minimum_certificates = [
        certificate
        for certificate, distance_squared in zip(
            projection_certificates, exact_distances_squared
        )
        if distance_squared == minimum_distance_squared
    ]
    print(
        f"exact separation: Dmin^2={fraction_text(minimum_distance_squared)} "
        f"diam^2={fraction_text(diameter_squared)} "
        f"margin={float(squared_margin):.12f} "
        f"ratio={math.sqrt(float(minimum_distance_squared/diameter_squared)):.12f}",
        flush=True,
    )

    forbidden = np.asarray(
        combigeo.forbidden_coords(basis.tolist(), diameter, 1.0),
        dtype=np.int64,
    )
    kernel_conflicts = int(killed_mask(forbidden, rows, moduli).sum())
    if kernel_conflicts:
        raise AssertionError(
            f"independent forbidden-set route found {kernel_conflicts} conflicts"
        )
    print(
        f"independent forbidden set: |F|={len(forbidden)}, "
        "kernel conflicts=0",
        flush=True,
    )

    payload = {
        "method": "rational Gram + exact vertices + exact KKT projections",
        "source_candidate": str(args.candidate),
        "denominator": args.denominator,
        "integer_gram": integer_gram.astype(int).tolist(),
        "exact_positive_definite_leading_minors": leading_minors,
        "moduli": moduli,
        "rows": [row.astype(int).tolist() for row in rows],
        "image_index": exact_image,
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": exact_determinant,
        "kernel_smith": smith_diagonal(kernel),
        "lll_kernel_basis_rows": reduced_rows.astype(int).tolist(),
        "voronoi": {
            "facets": len(facets),
            "vertices": len(hull.intersections),
            "simple": True,
            "singular_vertex_systems": singular,
            "covering_radius_squared": fraction_text(
                exact_radius_squared
            ),
            "covering_radius_squared_float": float(
                exact_radius_squared
            ),
            "diameter": diameter,
            "farthest_vertex_count": len(farthest_active),
            "farthest_active_facet_ids": farthest_active,
        },
        "short_vector_certificate": {
            "length_cutoff_squared": fraction_text(
                16 * exact_radius_squared
            ),
            "coefficient_bounds_squared": [
                fraction_text(value) for value in coefficient_bounds
            ],
            "coefficient_box": [-1, 1],
            "exact_vector_count": len(short_vectors),
            "cpp_vector_count": len(cpp_coordinates),
        },
        "separation": {
            "minimum_distance_squared": fraction_text(
                minimum_distance_squared
            ),
            "diameter_squared": fraction_text(diameter_squared),
            "squared_margin": fraction_text(squared_margin),
            "squared_margin_float": float(squared_margin),
            "distance_ratio": math.sqrt(
                float(minimum_distance_squared / diameter_squared)
            ),
            "minimum_witnesses": minimum_certificates,
            "all_projection_certificates": projection_certificates,
        },
        "independent_forbidden_check": {
            "forbidden_count": int(len(forbidden)),
            "kernel_conflicts": kernel_conflicts,
        },
        "certified_upper_bound": 1323,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"certificate saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
