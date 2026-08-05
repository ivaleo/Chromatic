"""Independent numerical audit for A9-star secondary-cone candidates.

The search optimizer uses a specialized permutation archive and subset-sum
facets.  This verifier deliberately recomputes the main ingredients through
separate paths:

* exact SymPy determinant and Smith data for the integer coloring kernel;
* actual Voronoi-relevant facets from ``combigeo``;
* the complete 10! specialized covering scan;
* direct active-facet linear solves for the worst and random permutations;
* all-facet feasibility of the reported worst Voronoi vertex; and
* complete kernel separation using the generic actual-facet oracle.

This remains a floating-point audit, not an exact rational certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from sympy import Matrix

import combigeo
from chromatic_research.core.lazy_prime_campaign import separate_kernel
from chromatic_research.campaigns.permutohedral_cover import (
    covering_radius,
    permutohedral_facet_coordinates,
    radii_for_orders,
    superbase_from_astar_basis,
)
from chromatic_research.core.prime_radon import smith_diagonal


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direct_order_vertex(
    basis: np.ndarray, order: Sequence[int]
) -> tuple[float, np.ndarray, float]:
    """Solve the prefix-facet equations for one Delone permutation."""
    basis = np.asarray(basis, dtype=np.float64)
    superbase = superbase_from_astar_basis(basis)
    order_array = np.asarray(order, dtype=np.int64)
    n = len(basis)
    if order_array.shape != (n + 1,) or sorted(order_array.tolist()) != list(
        range(n + 1)
    ):
        raise ValueError("order must be a permutation of 0,...,n")
    prefixes = np.cumsum(superbase[order_array[:-1]], axis=0)
    rhs = 0.5 * np.einsum("ij,ij->i", prefixes, prefixes)
    center = np.linalg.solve(prefixes, rhs)
    residual = float(np.max(np.abs(prefixes @ center - rhs)))
    return float(np.linalg.norm(center)), center, residual


def actual_facet_coordinates(
    basis: np.ndarray,
    facets: Sequence[tuple[Sequence[float], float]],
) -> tuple[set[tuple[int, ...]], float]:
    inverse = np.linalg.inv(basis)
    coordinates: set[tuple[int, ...]] = set()
    maximum_residual = 0.0
    for normal, _ in facets:
        raw = np.asarray(normal, dtype=np.float64) @ inverse
        rounded = np.rint(raw).astype(np.int64)
        maximum_residual = max(
            maximum_residual, float(np.max(np.abs(raw - rounded)))
        )
        coordinates.add(tuple(rounded.tolist()))
    return coordinates, maximum_residual


def audit_candidate(
    candidate_path: Path,
    *,
    spot_checks: int,
    seed: int,
) -> dict:
    payload = json.loads(candidate_path.read_text())
    best = payload["best"]
    basis = np.asarray(best["basis"], dtype=np.float64)
    kernel = np.asarray(payload["kernel_basis_columns"], dtype=np.int64)
    if basis.shape != (9, 9) or kernel.shape != (9, 9):
        raise ValueError("candidate must contain 9 by 9 basis and kernel")

    exact_determinant = abs(int(Matrix(kernel.tolist()).det()))
    exact_smith = smith_diagonal(kernel)
    actual_facets = combigeo.relevant_facets(basis.tolist())
    observed, coordinate_residual = actual_facet_coordinates(
        basis, actual_facets
    )
    expected = {
        tuple(row)
        for row in permutohedral_facet_coordinates(9).tolist()
    }

    radius, vertex_count, witness = covering_radius(
        basis, with_witness=True
    )
    assert witness is not None
    worst_order = witness["permutation"]
    direct_radius, direct_center, direct_residual = direct_order_vertex(
        basis, worst_order
    )

    # Cross-check the specialized batched formula against independently
    # solved prefix-facet systems on reproducible random permutations.
    rng = np.random.default_rng(seed)
    orders = np.asarray(
        [rng.permutation(10) for _ in range(spot_checks)],
        dtype=np.uint8,
    )
    specialized_spots = radii_for_orders(basis, orders)
    direct_spots = np.asarray(
        [direct_order_vertex(basis, order)[0] for order in orders]
    )
    spot_difference = float(
        np.max(np.abs(specialized_spots - direct_spots))
    )

    normals = np.asarray(
        [normal for normal, _ in actual_facets], dtype=np.float64
    )
    facet_rhs = 0.5 * np.einsum("ij,ij->i", normals, normals)
    witness_slacks = facet_rhs - normals @ direct_center
    minimum_witness_slack = float(np.min(witness_slacks))
    tight_facets = int(np.count_nonzero(np.abs(witness_slacks) <= 2e-8))

    recorded_diameter = float(best["diameter"])
    recorded_ratio = float(best["min_ratio"])
    recorded_min_distance = float(best["min_distance"])
    independent_diameter = 2.0 * radius
    separation = separate_kernel(
        basis,
        independent_diameter,
        actual_facets,
        kernel,
    )
    independent_ratio = float(separation["minimum_distance_ratio"])
    independent_min_distance = independent_ratio * independent_diameter

    gram = basis @ basis.T
    recorded_gram = np.asarray(best.get("gram"), dtype=np.float64)
    gram_difference = (
        float(np.max(np.abs(gram - recorded_gram)))
        if recorded_gram.shape == (9, 9)
        else float("inf")
    )

    checks = {
        "kernel_determinant_exact": (
            exact_determinant == int(payload["kernel_determinant"])
        ),
        "kernel_smith_exact": exact_smith == payload["kernel_smith"],
        "actual_facets_are_all_subset_sums": observed == expected,
        "facet_coordinate_residual_small": coordinate_residual <= 2e-6,
        "complete_vertex_count": vertex_count == math.factorial(10),
        "recorded_diameter_reproduced": math.isclose(
            independent_diameter,
            recorded_diameter,
            rel_tol=2e-11,
            abs_tol=2e-11,
        ),
        "worst_vertex_direct_solve_reproduced": math.isclose(
            direct_radius,
            radius,
            rel_tol=2e-11,
            abs_tol=2e-11,
        ),
        "worst_vertex_equations_hold": direct_residual <= 2e-10,
        "worst_vertex_satisfies_all_actual_facets": (
            minimum_witness_slack >= -2e-9
        ),
        "random_direct_solve_crosscheck": spot_difference <= 2e-10,
        "recorded_separation_reproduced": math.isclose(
            independent_ratio,
            recorded_ratio,
            rel_tol=2e-10,
            abs_tol=2e-10,
        ),
        "recorded_min_distance_reproduced": math.isclose(
            independent_min_distance,
            recorded_min_distance,
            rel_tol=2e-10,
            abs_tol=2e-10,
        ),
        "recorded_gram_reproduced": gram_difference <= 2e-12,
    }
    return {
        "method": "independent A9-star secondary-cone numerical audit",
        "candidate": str(candidate_path),
        "candidate_sha256": sha256(candidate_path),
        "verifier_sha256": sha256(Path(__file__)),
        "spot_checks": spot_checks,
        "seed": seed,
        "exact_kernel_determinant": exact_determinant,
        "exact_kernel_smith": exact_smith,
        "actual_facet_count": len(actual_facets),
        "expected_subset_facet_count": len(expected),
        "maximum_facet_coordinate_residual": coordinate_residual,
        "complete_vertex_count": vertex_count,
        "recorded_diameter": recorded_diameter,
        "independent_diameter": independent_diameter,
        "worst_permutation": worst_order,
        "direct_worst_radius": direct_radius,
        "direct_worst_equation_residual": direct_residual,
        "minimum_worst_vertex_facet_slack": minimum_witness_slack,
        "tight_actual_facets_at_worst_vertex": tight_facets,
        "maximum_random_direct_radius_difference": spot_difference,
        "recorded_minimum_distance_ratio": recorded_ratio,
        "independent_minimum_distance_ratio": independent_ratio,
        "recorded_minimum_distance": recorded_min_distance,
        "independent_minimum_distance": independent_min_distance,
        "independent_checked_kernel_vectors": separation[
            "checked_kernel_vectors"
        ],
        "independent_conflict_count_with_sign": separation[
            "conflict_count_with_sign"
        ],
        "maximum_recorded_gram_difference": gram_difference,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--spot-checks", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.spot_checks < 1:
        parser.error("spot checks must be positive")

    audit = audit_candidate(
        args.candidate,
        spot_checks=args.spot_checks,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"audit all={audit['all_checks_passed']} "
        f"index={audit['exact_kernel_determinant']} "
        f"facets={audit['actual_facet_count']} "
        f"vertices={audit['complete_vertex_count']} "
        f"ratio={audit['independent_minimum_distance_ratio']:.12f} "
        f"saved={args.output}",
        flush=True,
    )
    return 0 if audit["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
