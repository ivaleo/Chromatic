"""Fourier/Hoffman--Delsarte bounds for finite toroidal conflict graphs.

For a finite abelian Cayley graph ``G = Cay(Gamma, S)`` all irreducible
characters are one-dimensional.  Averaging the Lovasz-theta SDP over
translations therefore turns it into a linear program.  In Fourier
coordinates the primal program used here is

    maximize    a_trivial
    subject to  sum_chi a_chi = |Gamma|,
                sum_chi a_chi chi(s) = 0       for s in S,
                a_chi >= 0.

It upper-bounds the independence number.  Averaging a character and its
inverse makes the program real.  The Schrijver strengthening additionally
requires the inverse Fourier transform to be entrywise nonnegative.

The module also computes the ordinary Hoffman ratio bound from the exact
connection set.  HiGHS solves both Fourier LPs through
``scipy.optimize.linprog``.  The quotient and connection arithmetic is exact;
the trigonometric character table and LP certificates are numerical, so a
candidate theorem still needs interval/rational certification or a discrete
exact oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from d6_periodic_lift_highs import (
    quotient_key,
    quotient_map,
    quotient_representatives,
)
from d6_torus_column_generation import (
    CayleyConflictGraph,
    _json_profile,
    build_cayley_graph,
    character_extension_profiles,
    refinement_period,
    source_extension_coordinates,
)
from determinant_repair import exact_det, load_preset
from prime_radon import projective_forms
from prime_row_opt import _forbidden_with_weights


@dataclass(frozen=True)
class RealFourierBasis:
    """Characters of a finite abelian quotient modulo inversion."""

    determinant: int
    representatives: np.ndarray
    multiplicities: np.ndarray

    @property
    def orbit_count(self) -> int:
        return int(len(self.representatives))

    def evaluate(self, quotient_keys: Sequence[Sequence[int]]) -> np.ndarray:
        """Evaluate inverse-paired real characters on quotient keys.

        Columns correspond to character inversion orbits.  A two-element
        orbit contributes ``chi(g) + conjugate(chi(g)) = 2 cos(theta)``;
        a self-inverse character contributes just ``chi(g)``.
        """
        keys = np.asarray(quotient_keys, dtype=np.int64)
        if keys.ndim != 2 or keys.shape[1] != self.representatives.shape[1]:
            raise ValueError("quotient keys and character basis are incompatible")
        numerators = np.remainder(
            self.representatives @ keys.T,
            self.determinant,
        )
        values = np.cos(
            (2.0 * np.pi / self.determinant)
            * numerators.astype(np.float64)
        )
        values *= self.multiplicities[:, None]
        return values.T


def real_fourier_basis(period: np.ndarray) -> RealFourierBasis:
    """Enumerate dual characters and quotient them by complex conjugation."""
    period = np.asarray(period, dtype=np.int64)
    determinant, _ = quotient_map(period)
    dual_period = period.T.copy()
    dual_determinant, dual_adjugate = quotient_map(dual_period)
    if dual_determinant != determinant:
        raise AssertionError("primal and dual quotient orders differ")
    dual_representatives, dual_index = quotient_representatives(dual_period)

    seen: set[int] = set()
    representatives: list[np.ndarray] = []
    multiplicities: list[int] = []
    for index, representative in enumerate(dual_representatives):
        if index in seen:
            continue
        inverse_key = quotient_key(
            -representative,
            dual_adjugate,
            determinant,
        )
        inverse_index = int(dual_index[inverse_key])
        seen.add(index)
        seen.add(inverse_index)
        representative_index = min(index, inverse_index)
        representatives.append(
            np.asarray(
                dual_representatives[representative_index],
                dtype=np.int64,
            )
        )
        multiplicities.append(1 if index == inverse_index else 2)

    if len(seen) != determinant:
        raise AssertionError("character inversion orbits do not partition the dual")
    representatives_array = np.asarray(representatives, dtype=np.int64)
    multiplicities_array = np.asarray(multiplicities, dtype=np.int64)
    if (
        not np.all(representatives_array[0] == 0)
        or multiplicities_array[0] != 1
        or int(multiplicities_array.sum()) != determinant
    ):
        raise AssertionError("the trivial character was not normalized")
    return RealFourierBasis(
        determinant=determinant,
        representatives=representatives_array,
        multiplicities=multiplicities_array,
    )


def inverse_orbit_keys(
    keys: Sequence[Sequence[int]],
    determinant: int,
) -> list[tuple[int, ...]]:
    """Choose one lexicographic representative from every ``{g,-g}`` pair."""
    normalized = {
        tuple(int(value) % determinant for value in key) for key in keys
    }
    representatives: set[tuple[int, ...]] = set()
    for key in normalized:
        inverse = tuple((-value) % determinant for value in key)
        if inverse not in normalized:
            raise ValueError("key set is not closed under inversion")
        representatives.add(min(key, inverse))
    return sorted(representatives)


def cayley_eigenvalues(
    graph: CayleyConflictGraph,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    """Return all adjacency eigenvalues from the abelian character table."""
    if not graph.loop_free:
        raise ValueError("a looped quotient has no simple conflict graph")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    dual_representatives, _ = quotient_representatives(graph.period.T)
    characters = np.asarray(dual_representatives, dtype=np.int64)
    connections = np.asarray(
        sorted(graph.connection_keys),
        dtype=np.int64,
    )
    if not len(connections):
        return np.zeros(graph.vertex_count, dtype=np.float64)
    eigenvalues = np.empty(graph.vertex_count, dtype=np.float64)
    scale = 2.0 * np.pi / graph.determinant
    for start in range(0, graph.vertex_count, batch_size):
        stop = min(start + batch_size, graph.vertex_count)
        numerators = np.remainder(
            characters[start:stop] @ connections.T,
            graph.determinant,
        )
        eigenvalues[start:stop] = np.cos(
            scale * numerators.astype(np.float64)
        ).sum(axis=1)
    eigenvalues[np.abs(eigenvalues) < 1e-11] = 0.0
    return eigenvalues


def hoffman_ratio_bound(
    graph: CayleyConflictGraph,
    *,
    batch_size: int = 512,
) -> dict:
    """Compute the Hoffman upper bound on ``alpha(G)``."""
    started = time.perf_counter()
    eigenvalues = cayley_eigenvalues(graph, batch_size=batch_size)
    degree = float(graph.degree)
    minimum = float(eigenvalues.min(initial=0.0))
    if degree == 0.0:
        bound = float(graph.vertex_count)
    elif minimum < 0.0:
        bound = (
            graph.vertex_count
            * (-minimum)
            / (degree - minimum)
        )
    else:
        bound = float(graph.vertex_count)
    return {
        "success": True,
        "degree": int(graph.degree),
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": float(eigenvalues.max(initial=0.0)),
        "upper_bound": float(bound),
        "rounded_integer_upper_bound": int(
            min(graph.vertex_count, math.floor(bound + 1e-8))
        ),
        "distinct_eigenvalues_1e-8": int(
            len(np.unique(np.round(eigenvalues, decimals=8)))
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "solver": "abelian character spectrum and Hoffman ratio bound",
    }


def abelian_theta_lp(
    graph: CayleyConflictGraph,
    *,
    nonnegative: bool = False,
    time_limit: float = 60.0,
) -> dict:
    """Solve the translation-reduced Lovasz or Schrijver theta LP.

    With ``nonnegative=False`` this is the ordinary Lovasz theta number.
    With ``nonnegative=True`` the inverse Fourier transform is constrained
    entrywise nonnegative, giving Schrijver's ``theta-prime`` strengthening.
    """
    if not graph.loop_free:
        return {
            "success": False,
            "optimal": False,
            "status": "LOOPED",
            "upper_bound": None,
        }
    if time_limit <= 0:
        raise ValueError("time limit must be positive")
    started = time.perf_counter()
    basis = real_fourier_basis(graph.period)
    edge_keys = inverse_orbit_keys(
        graph.connection_keys,
        graph.determinant,
    )
    edge_values = basis.evaluate(edge_keys)
    equality = np.vstack(
        [
            basis.multiplicities.astype(np.float64),
            edge_values,
        ]
    )
    equality_rhs = np.zeros(len(edge_values) + 1, dtype=np.float64)
    equality_rhs[0] = float(graph.vertex_count)

    inequality = None
    inequality_rhs = None
    compatible_keys: list[tuple[int, ...]] = []
    if nonnegative:
        compatible_keys = [
            key
            for key in graph.keys
            if key not in graph.connection_keys
        ]
        compatible_values = basis.evaluate(compatible_keys)
        inequality = csc_matrix(-compatible_values)
        inequality_rhs = np.zeros(
            len(compatible_values),
            dtype=np.float64,
        )

    objective = np.zeros(basis.orbit_count, dtype=np.float64)
    objective[0] = -1.0
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=csc_matrix(equality),
        b_eq=equality_rhs,
        bounds=(0.0, None),
        method="highs",
        options={
            "presolve": True,
            "time_limit": float(time_limit),
            "dual_feasibility_tolerance": 1e-8,
            "primal_feasibility_tolerance": 1e-8,
        },
    )
    optimal = bool(result.status == 0 and result.x is not None)
    upper_bound = float(-result.fun) if optimal else None
    normalization_residual = None
    edge_residual = None
    minimum_function_value = None
    if result.x is not None:
        normalization_residual = float(
            abs(
                basis.multiplicities.astype(np.float64) @ result.x
                - graph.vertex_count
            )
        )
        edge_residual = float(
            np.max(np.abs(edge_values @ result.x), initial=0.0)
            / graph.vertex_count
        )
        if nonnegative:
            minimum_function_value = float(
                np.min(
                    basis.evaluate(compatible_keys) @ result.x,
                    initial=0.0,
                )
                / graph.vertex_count
            )
    return {
        "success": optimal,
        "optimal": optimal,
        "status": int(result.status),
        "message": str(result.message),
        "upper_bound": upper_bound,
        "rounded_integer_upper_bound": (
            int(math.floor(upper_bound + 1e-7))
            if upper_bound is not None
            else None
        ),
        "character_orbits": basis.orbit_count,
        "edge_inversion_orbits": len(edge_keys),
        "nonnegative_rows": len(compatible_keys) if nonnegative else 0,
        "normalization_residual": normalization_residual,
        "maximum_edge_residual": edge_residual,
        "minimum_function_value": minimum_function_value,
        "iterations": int(getattr(result, "nit", 0) or 0),
        "elapsed_seconds": time.perf_counter() - started,
        "solver": (
            "HiGHS Schrijver theta-prime LP in abelian Fourier coordinates"
            if nonnegative
            else "HiGHS Lovasz theta LP in abelian Fourier coordinates"
        ),
    }


def _parse_primes(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one prime is required")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primes", type=_parse_primes, default=[3, 5, 7])
    parser.add_argument("--sparsest", type=int, default=0)
    parser.add_argument("--per-profile", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument(
        "--theta-prime",
        action="store_true",
        help="also solve the entrywise-nonnegative Schrijver strengthening",
    )
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.sparsest < 0
        or args.per_profile < 0
        or args.batch_size < 1
        or args.time_limit <= 0
        or args.target_colors < 1
    ):
        parser.error("budgets, time limit, and target must be positive")

    lattice, basis, diameter, _, source_kernel = load_preset("d6")
    forbidden, _, _ = _forbidden_with_weights(basis, diameter)
    coordinates, class_ids, source_order = source_extension_coordinates(
        source_kernel,
        forbidden,
    )
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "Hoffman and translation-reduced Lovasz/Schrijver theta "
            "bounds for prime covers of the exact d6 index-343 kernel"
        ),
        "lattice": lattice,
        "dimension": len(source_kernel),
        "source_index": source_order,
        "source_kernel_basis_columns": source_kernel.astype(int).tolist(),
        "source_kernel_determinant": abs(exact_det(source_kernel)),
        "forbidden_projective_pairs": len(forbidden),
        "target_colors": args.target_colors,
        "settings": {
            "primes": args.primes,
            "sparsest": args.sparsest,
            "per_profile": args.per_profile,
            "batch_size": args.batch_size,
            "time_limit": args.time_limit,
            "theta_prime": args.theta_prime,
        },
        "prime_screens": [],
        "records": [],
        "candidate_periods": [],
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for prime in args.primes:
        characters = projective_forms(len(source_kernel), prime)
        profile = character_extension_profiles(
            coordinates,
            class_ids,
            source_order,
            characters,
            prime,
            batch_size=args.batch_size,
        )
        public_profile = _json_profile(profile)
        payload["prime_screens"].append(public_profile)
        ranked = sorted(
            profile["incomplete_character_indices"],
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        selected = set(ranked[: args.sparsest])
        by_profile: dict[tuple[int, int], list[int]] = {}
        for character_index in ranked:
            key = (
                int(profile["_connection_counts"][character_index]),
                int(profile["_minimum_coverages"][character_index]),
            )
            by_profile.setdefault(key, []).append(character_index)
        for indices in by_profile.values():
            selected.update(indices[: args.per_profile])
        selected_ordered = sorted(
            selected,
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        public_profile["spectral_selected"] = len(selected_ordered)
        public_profile["spectral_unscreened"] = (
            len(ranked) - len(selected_ordered)
        )
        save()

        for character_index in selected_ordered:
            period = refinement_period(
                source_kernel,
                characters[character_index],
                prime,
            )
            graph = build_cayley_graph(period, forbidden)
            necessary_alpha = int(
                math.ceil(graph.vertex_count / args.target_colors)
            )
            record = {
                "prime": prime,
                "character_index": int(character_index),
                "character": characters[character_index].astype(int).tolist(),
                "period_basis_columns": period.astype(int).tolist(),
                "period_index": graph.vertex_count,
                "connection_keys": len(graph.connection_keys),
                "necessary_independence_number_for_target": necessary_alpha,
                "hoffman": hoffman_ratio_bound(
                    graph,
                    batch_size=args.batch_size,
                ),
                "theta": abelian_theta_lp(
                    graph,
                    nonnegative=False,
                    time_limit=args.time_limit,
                ),
            }
            if args.theta_prime:
                record["theta_prime"] = abelian_theta_lp(
                    graph,
                    nonnegative=True,
                    time_limit=args.time_limit,
                )
            strongest = record.get("theta_prime", record["theta"])
            record["numerically_excludes_target_alpha"] = bool(
                strongest["optimal"]
                and strongest["upper_bound"] < necessary_alpha - 1e-7
            )
            payload["records"].append(record)
            if not record["numerically_excludes_target_alpha"]:
                payload["candidate_periods"].append(record)
            save()
            print(
                f"p={prime} character={character_index} "
                f"degree={graph.degree} "
                f"Hoffman={record['hoffman']['upper_bound']:.9g} "
                f"theta={record['theta']['upper_bound']}",
                flush=True,
            )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
