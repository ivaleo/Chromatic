"""Fully exact, independent audit of a five-dimensional coloring certificate.

This verifier deliberately does not use ``combigeo.relevant_facets``, Qhull,
or a floating-point short-vector enumerator.  Starting from the rational Gram
matrix stored in the main certificate, it:

1. enumerates every nonzero class of Z^n / 2 Z^n and applies Voronoi's exact
   relevant-vector criterion;
2. traverses the complete one-skeleton of the resulting rational H-polytope
   with exact edge pivots;
3. recomputes the covering radius at every exact vertex;
4. proves and enumerates the complete short-vector box of the coloring
   sublattice; and
5. rechecks the exact KKT witnesses for all potentially dangerous translates.

SciPy's LP solver is used only to obtain one starting active set.  That set is
immediately checked exactly.  Completeness then follows from exact graph
traversal and connectedness of the graph of a bounded convex polytope.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linprog
from sympy import Matrix, Rational


def fraction_text(value: Rational | int) -> str:
    value = Rational(value)
    return f"{value.p}/{value.q}" if value.q != 1 else str(value.p)


def rational_floor_sqrt(value: Rational) -> int:
    """Return floor(sqrt(value)) without a floating conversion."""
    value = Rational(value)
    if value < 0:
        raise ValueError("square-root argument must be nonnegative")
    result = math.isqrt(int(value.p // value.q))
    while Rational((result + 1) ** 2) <= value:
        result += 1
    while Rational(result**2) > value:
        result -= 1
    return result


def canonical_sign(vector: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in vector)
    for value in result:
        if value:
            return result if value > 0 else tuple(-entry for entry in result)
    return result


def exact_relevant_vectors(
    gram: Matrix,
) -> tuple[list[tuple[int, ...]], list[dict]]:
    """Apply Voronoi's parity-class criterion with a proved finite box."""
    n = gram.rows
    inverse = gram.inv()
    relevant: list[tuple[int, ...]] = []
    classes: list[dict] = []

    for parity in itertools.product((0, 1), repeat=n):
        if not any(parity):
            continue
        seed = Matrix(parity)
        upper_bound = int((seed.T * gram * seed)[0])
        bounds = [
            rational_floor_sqrt(Rational(upper_bound) * inverse[index, index])
            for index in range(n)
        ]
        ranges = [
            [
                value
                for value in range(-bound, bound + 1)
                if value % 2 == parity[index]
            ]
            for index, bound in enumerate(bounds)
        ]
        minimum: int | None = None
        minimizers: list[tuple[int, ...]] = []
        enumerated = 0
        for raw in itertools.product(*ranges):
            enumerated += 1
            vector = Matrix(raw)
            norm = int((vector.T * gram * vector)[0])
            if minimum is None or norm < minimum:
                minimum = norm
                minimizers = [tuple(int(value) for value in raw)]
            elif norm == minimum:
                minimizers.append(tuple(int(value) for value in raw))
        if minimum is None:
            raise AssertionError(f"empty parity box for {parity}")
        minimizers = sorted(set(minimizers))
        is_relevant = (
            len(minimizers) == 2
            and minimizers[0] == tuple(-value for value in minimizers[1])
        )
        if is_relevant:
            relevant.append(canonical_sign(minimizers[0]))
        classes.append(
            {
                "parity": list(parity),
                "initial_upper_bound": upper_bound,
                "coordinate_bounds": bounds,
                "enumerated": enumerated,
                "minimum_norm_integer": minimum,
                "minimizers": [list(vector) for vector in minimizers],
                "relevant": is_relevant,
            }
        )

    relevant = sorted(set(relevant))
    if len(relevant) * 2 == 0:
        raise AssertionError("no relevant vectors found")
    return relevant, classes


def facet_system(
    gram: Matrix,
    relevant: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, ...]], list[Matrix], list[Rational]]:
    coordinates: list[tuple[int, ...]] = []
    for vector in relevant:
        canonical = tuple(int(value) for value in vector)
        coordinates.extend(
            [canonical, tuple(-value for value in canonical)]
        )
    coordinates = sorted(set(coordinates))
    normals: list[Matrix] = []
    offsets: list[Rational] = []
    for coordinate in coordinates:
        row = Matrix([coordinate])
        normals.append(row * gram)
        offsets.append(Rational((row * gram * row.T)[0], 2))
    return coordinates, normals, offsets


def solve_vertex(
    active: Sequence[int],
    normals: Sequence[Matrix],
    offsets: Sequence[Rational],
) -> Matrix:
    matrix = Matrix.vstack(*(normals[index] for index in active))
    if matrix.rank() != matrix.cols:
        raise AssertionError(f"singular active set {tuple(active)}")
    return matrix.inv() * Matrix([offsets[index] for index in active])


def active_facets(
    vertex: Matrix,
    normals: Sequence[Matrix],
    offsets: Sequence[Rational],
) -> tuple[int, ...]:
    active: list[int] = []
    for index, (normal, offset) in enumerate(zip(normals, offsets)):
        value = (normal * vertex)[0]
        if value > offset:
            raise AssertionError(
                f"infeasible exact vertex at facet {index}: "
                f"{value} > {offset}"
            )
        if value == offset:
            active.append(index)
    return tuple(active)


def numerical_start(
    normals: Sequence[Matrix],
    offsets: Sequence[Rational],
    n: int,
) -> tuple[int, ...]:
    """Find one active set numerically and certify it immediately."""
    raw_a = np.asarray(
        [[float(value) for value in normal] for normal in normals],
        dtype=np.float64,
    )
    raw_b = np.asarray([float(value) for value in offsets], dtype=np.float64)
    if np.any(raw_b <= 0):
        raise AssertionError("Voronoi facet offsets must be positive")
    # Integer Gram matrices with denominators 10^5 or 10^6 make both sides of
    # the inequalities large.  HiGHS can then return its uninitialised
    # ``Status 0`` before presolve, even though the same rational polytope at
    # denominator 10^4 is handled correctly.  Dividing every inequality by
    # its positive offset leaves the polytope unchanged and makes the LP
    # scale-free: every right-hand side is exactly one in floating arithmetic.
    a = raw_a / raw_b[:, None]
    b = np.ones_like(raw_b)
    objective = np.sqrt(
        np.asarray([2, 3, 5, 7, 11][:n], dtype=np.float64)
    )
    failures: list[str] = []
    for method in ("highs", "highs-ds", "highs-ipm"):
        result = linprog(
            -objective,
            A_ub=a,
            b_ub=b,
            bounds=[(None, None)] * n,
            method=method,
        )
        if not result.success:
            failures.append(f"{method}: {result.message}")
            continue
        slacks = b - a @ result.x
        ordered = np.argsort(np.abs(slacks))
        pool = ordered[: min(len(ordered), 2 * n + 4)].tolist()
        for active in itertools.combinations(pool, n):
            try:
                vertex = solve_vertex(active, normals, offsets)
                exact_active = active_facets(vertex, normals, offsets)
            except AssertionError:
                continue
            if len(exact_active) == n:
                return exact_active
        failures.append(f"{method}: no exactly simple active set")
    raise AssertionError(
        "could not obtain an exactly certified starting vertex; "
        + "; ".join(failures)
    )


def exact_vertex_graph(
    gram: Matrix,
    normals: Sequence[Matrix],
    offsets: Sequence[Rational],
) -> tuple[dict[tuple[int, ...], Matrix], set[tuple[tuple[int, ...], tuple[int, ...]]]]:
    """Traverse all vertices by exact simple-polytope edge pivots."""
    n = gram.rows
    start = numerical_start(normals, offsets, n)
    vertices: dict[tuple[int, ...], Matrix] = {
        start: solve_vertex(start, normals, offsets)
    }
    queue: deque[tuple[int, ...]] = deque([start])
    edges: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    while queue:
        active = queue.popleft()
        vertex = vertices[active]
        if active_facets(vertex, normals, offsets) != active:
            raise AssertionError(f"stored active set is not exact at {active}")

        for dropped in active:
            retained = tuple(index for index in active if index != dropped)
            retained_matrix = Matrix.vstack(
                *(normals[index] for index in retained)
            )
            nullspace = retained_matrix.nullspace()
            if len(nullspace) != 1:
                raise AssertionError(
                    f"edge nullspace has dimension {len(nullspace)} at {active}"
                )
            direction = nullspace[0]
            dropped_rate = (normals[dropped] * direction)[0]
            if dropped_rate == 0:
                raise AssertionError("dropped facet is constant along its edge")
            if dropped_rate > 0:
                direction = -direction

            step: Rational | None = None
            for index, (normal, offset) in enumerate(
                zip(normals, offsets)
            ):
                rate = (normal * direction)[0]
                if rate <= 0:
                    continue
                slack = offset - (normal * vertex)[0]
                if slack < 0:
                    raise AssertionError("negative exact slack")
                candidate = Rational(slack, rate)
                if candidate <= 0:
                    continue
                if step is None or candidate < step:
                    step = candidate
            if step is None:
                raise AssertionError("unbounded edge in a Voronoi polytope")

            neighbor_vertex = vertex + step * direction
            neighbor = active_facets(
                neighbor_vertex, normals, offsets
            )
            if len(neighbor) != n:
                raise AssertionError(
                    f"non-simple neighbor with {len(neighbor)} facets: {neighbor}"
                )
            if set(retained) - set(neighbor):
                raise AssertionError("edge pivot lost a retained facet")
            if neighbor == active:
                raise AssertionError("zero exact edge step")
            canonical_edge = tuple(sorted((active, neighbor)))
            edges.add(canonical_edge)
            if neighbor not in vertices:
                vertices[neighbor] = neighbor_vertex
                queue.append(neighbor)

    degrees = {active: 0 for active in vertices}
    for first, second in edges:
        degrees[first] += 1
        degrees[second] += 1
    bad_degrees = {
        active: degree for active, degree in degrees.items() if degree != n
    }
    if bad_degrees:
        raise AssertionError(f"unexpected vertex degrees: {bad_degrees}")
    return vertices, edges


def exact_covering_radius(
    gram: Matrix,
    denominator: int,
    vertices: dict[tuple[int, ...], Matrix],
) -> tuple[Rational, list[tuple[int, ...]]]:
    maximum: Rational | None = None
    farthest: list[tuple[int, ...]] = []
    for active, vertex in vertices.items():
        norm = Rational(
            (vertex.T * gram * vertex)[0],
            denominator,
        )
        if maximum is None or norm > maximum:
            maximum = norm
            farthest = [active]
        elif norm == maximum:
            farthest.append(active)
    if maximum is None:
        raise AssertionError("no vertices were traversed")
    return maximum, sorted(farthest)


def enumerate_short_kernel_vectors(
    gram: Matrix,
    denominator: int,
    covering_radius_squared: Rational,
    reduced_rows: Matrix,
) -> tuple[list[tuple[int, ...]], list[Rational]]:
    reduced_gram = reduced_rows * gram * reduced_rows.T
    inverse_physical = reduced_gram.inv() * denominator
    norm_bound = 16 * covering_radius_squared
    coefficient_bounds = [
        Rational(norm_bound * inverse_physical[index, index])
        for index in range(reduced_rows.rows)
    ]
    if any(value >= 4 for value in coefficient_bounds):
        raise AssertionError("the exact [-1,1]^n coefficient bound failed")

    vectors: list[tuple[int, ...]] = []
    for coefficient in itertools.product((-1, 0, 1), repeat=reduced_rows.rows):
        if not any(coefficient):
            continue
        coordinate = Matrix([coefficient]) * reduced_rows
        norm = Rational(
            (coordinate * gram * coordinate.T)[0],
            denominator,
        )
        if norm < norm_bound:
            vectors.append(tuple(int(value) for value in coordinate))
    return sorted(set(vectors)), coefficient_bounds


def projection_distance_from_witness(
    coordinate: Sequence[int],
    active_coordinates: Sequence[Sequence[int]],
    all_facet_coordinates: Sequence[Sequence[int]],
    gram: Matrix,
    denominator: int,
) -> tuple[Rational, list[Rational]]:
    point = Matrix([list(map(int, coordinate))]).T / 2
    active = Matrix([list(map(int, vector)) for vector in active_coordinates])
    if not active.rows:
        raise AssertionError("projection witness has no active facets")
    middle = active * gram * active.T
    if middle.det() == 0:
        raise AssertionError("singular projection active set")
    boundary = Matrix(
        [
            (active.row(index) * gram * active.row(index).T)[0] / 2
            for index in range(active.rows)
        ]
    )
    multipliers = middle.inv() * (
        active * gram * point - boundary
    )
    if any(value < 0 for value in multipliers):
        raise AssertionError("negative exact KKT multiplier")
    projection = point - active.T * multipliers
    for vector in all_facet_coordinates:
        facet = Matrix([list(map(int, vector))])
        if (
            (facet * gram * projection)[0]
            > (facet * gram * facet.T)[0] / 2
        ):
            raise AssertionError("KKT projection is outside the Voronoi cell")
    difference = point - projection
    distance_squared = Rational(
        4 * (difference.T * gram * difference)[0],
        denominator,
    )
    return distance_squared, [Rational(value) for value in multipliers]


def verify_projection_witnesses(
    certificate: dict,
    short_vectors: Sequence[Sequence[int]],
    facet_coordinates: Sequence[Sequence[int]],
    gram: Matrix,
    denominator: int,
) -> tuple[Rational, list[dict]]:
    stored = {
        tuple(int(value) for value in item["coordinate"]): item
        for item in certificate["separation"]["all_projection_certificates"]
    }
    if set(stored) != {tuple(vector) for vector in short_vectors}:
        raise AssertionError(
            "stored KKT witness coordinates do not equal the exact short set"
        )
    distances: list[Rational] = []
    checks: list[dict] = []
    for coordinate in short_vectors:
        witness = stored[tuple(coordinate)]
        distance, multipliers = projection_distance_from_witness(
            coordinate,
            witness["active_facets"],
            facet_coordinates,
            gram,
            denominator,
        )
        if distance != Rational(witness["distance_squared"]):
            raise AssertionError(
                f"projection distance mismatch for {coordinate}: "
                f"{distance} != {witness['distance_squared']}"
            )
        stored_multipliers = [
            Rational(value) for value in witness["multipliers"]
        ]
        if multipliers != stored_multipliers:
            raise AssertionError(
                f"KKT multiplier mismatch for {coordinate}"
            )
        distances.append(distance)
        checks.append(
            {
                "coordinate": list(coordinate),
                "distance_squared": fraction_text(distance),
                "active_facets": witness["active_facets"],
                "multipliers": [
                    fraction_text(value) for value in multipliers
                ],
            }
        )
    return min(distances), checks


def determinant_abs(matrix: Matrix) -> int:
    return abs(int(matrix.det()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "finish the independent exact audit when the separation or "
            "interval margin is nonpositive; never record an invalid upper "
            "bound"
        ),
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    certificate = json.loads(args.certificate.read_text())
    gram = Matrix(certificate["integer_gram"])
    denominator = int(certificate["denominator"])
    n = gram.rows
    if gram.cols != n:
        raise AssertionError("Gram matrix is not square")
    if any(gram[:size, :size].det() <= 0 for size in range(1, n + 1)):
        raise AssertionError("Gram matrix is not positive definite")

    relevant, parity_classes = exact_relevant_vectors(gram)
    facet_coordinates, normals, offsets = facet_system(gram, relevant)
    print(
        f"exact parity audit: classes={len(parity_classes)} "
        f"relevant-pairs={len(relevant)} facets={len(facet_coordinates)}",
        flush=True,
    )
    if len(facet_coordinates) != certificate["voronoi"]["facets"]:
        raise AssertionError("exact facet count differs from the main certificate")

    vertices, edges = exact_vertex_graph(gram, normals, offsets)
    radius_squared, farthest = exact_covering_radius(
        gram, denominator, vertices
    )
    print(
        f"exact graph audit: vertices={len(vertices)} edges={len(edges)} "
        f"R^2={fraction_text(radius_squared)}",
        flush=True,
    )
    if len(vertices) != certificate["voronoi"]["vertices"]:
        raise AssertionError("exact vertex count differs from the main certificate")
    if radius_squared != Rational(
        certificate["voronoi"]["covering_radius_squared"]
    ):
        raise AssertionError("exact covering radius differs from main certificate")
    incident = set(itertools.chain.from_iterable(vertices))
    if incident != set(range(len(facet_coordinates))):
        raise AssertionError("some exact relevant facet is not vertex-incident")

    kernel = Matrix(certificate["kernel_basis_columns"])
    reduced_rows = Matrix(certificate["lll_kernel_basis_rows"])
    rows = certificate["rows"]
    moduli = certificate["moduli"]
    if determinant_abs(kernel) != certificate["image_index"]:
        raise AssertionError("kernel determinant does not equal image index")
    if determinant_abs(reduced_rows) != determinant_abs(kernel):
        raise AssertionError("reduced rows do not generate an equal-index lattice")
    for reduced_row in reduced_rows.tolist():
        for row, modulus in zip(rows, moduli):
            dot = sum(
                int(left) * int(right)
                for left, right in zip(row, reduced_row)
            )
            if dot % int(modulus):
                raise AssertionError("LLL row is not in the modular kernel")

    short_vectors, coefficient_bounds = enumerate_short_kernel_vectors(
        gram, denominator, radius_squared, reduced_rows
    )
    print(
        f"exact kernel audit: short-vectors={len(short_vectors)} "
        f"coefficient-box=[-1,1]^{n}",
        flush=True,
    )
    if len(short_vectors) != certificate["short_vector_certificate"][
        "exact_vector_count"
    ]:
        raise AssertionError("short-vector count differs from main certificate")
    for coordinate in short_vectors:
        for row, modulus in zip(rows, moduli):
            if sum(a * b for a, b in zip(row, coordinate)) % int(modulus):
                raise AssertionError("enumerated short vector is outside the kernel")

    minimum_distance_squared, projection_checks = (
        verify_projection_witnesses(
            certificate,
            short_vectors,
            facet_coordinates,
            gram,
            denominator,
        )
    )
    diameter_squared = 4 * radius_squared
    squared_margin = minimum_distance_squared - diameter_squared
    separation_valid = bool(squared_margin > 0)
    if not separation_valid and not args.diagnostic:
        raise AssertionError("exact separation margin is not positive")
    interval = Rational(certificate["certified_interval"]["upper_endpoint"])
    interval_margin = (
        minimum_distance_squared - interval**2 * diameter_squared
    )
    interval_valid = bool(interval_margin > 0)
    if not interval_valid and not args.diagnostic:
        raise AssertionError("stored interval endpoint is not certified")
    if minimum_distance_squared != Rational(
        certificate["separation"]["minimum_distance_squared"]
    ):
        raise AssertionError("minimum separation differs from main certificate")

    payload = {
        "method": (
            "exact parity classes + exact one-skeleton traversal + "
            "exact short-vector/KKT audit"
        ),
        "source_certificate": str(args.certificate),
        "diagnostic_mode": bool(args.diagnostic),
        "diagnostic_status": (
            "valid-certificate" if interval_valid else "invalid-candidate"
        ),
        "dimension": n,
        "denominator": denominator,
        "positive_definite": True,
        "voronoi_relevant_audit": {
            "parity_classes": len(parity_classes),
            "relevant_pairs": len(relevant),
            "facet_count": len(facet_coordinates),
            "relevant_vectors_one_per_sign_pair": [
                list(vector) for vector in relevant
            ],
            "classes": parity_classes,
        },
        "exact_polytope_graph": {
            "vertices": len(vertices),
            "edges": len(edges),
            "degree": n,
            "connected": True,
            "all_facets_incident": True,
            "farthest_vertex_count": len(farthest),
            "covering_radius_squared": fraction_text(radius_squared),
        },
        "kernel": {
            "index": determinant_abs(kernel),
            "lll_index": determinant_abs(reduced_rows),
            "all_lll_rows_in_modular_kernel": True,
            "coefficient_bounds_squared": [
                fraction_text(value) for value in coefficient_bounds
            ],
            "short_vector_count": len(short_vectors),
        },
        "projection_audit": {
            "certificate_count": len(projection_checks),
            "minimum_distance_squared": fraction_text(
                minimum_distance_squared
            ),
            "diameter_squared": fraction_text(diameter_squared),
            "squared_margin": fraction_text(squared_margin),
            "valid": separation_valid,
            "distance_ratio": math.sqrt(
                float(minimum_distance_squared / diameter_squared)
            ),
            "all_kkt_checks": projection_checks,
        },
        "interval_audit": {
            "upper_endpoint": fraction_text(interval),
            "squared_margin": fraction_text(interval_margin),
            "valid": interval_valid,
        },
        "certified_upper_bound": (
            certificate["image_index"] if interval_valid else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if interval_valid:
        label = (
            f"INDEPENDENT EXACT AUDIT PASSED: "
            f"colors={certificate['image_index']}"
        )
    else:
        label = "INDEPENDENT EXACT DIAGNOSTIC PASSED: no bound recorded"
    print(
        f"{label} "
        f"ratio={payload['projection_audit']['distance_ratio']:.15f} "
        f"elapsed={payload['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
