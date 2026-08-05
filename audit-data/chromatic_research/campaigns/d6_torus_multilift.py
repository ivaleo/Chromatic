"""Screen composite non-coset covers of the exact E6*/343 coloring.

For every rank-``r`` subspace of ``F_p^6`` this campaign refines the known
index-343 color kernel by ``ker(A)``.  The resulting period has
``343 * p^r`` cells.  Exact residue masks first identify complete multipartite
graphs.  Exceptional profiles are then tested for an independent set of size

    ceil(343 * p^r / target_colors).

The decision oracle is CP-SAT; a negative result is exact because a source
fiber supplies an independent set of size ``p^r``.  If the next size is
feasible, the generic HiGHS column-generation machinery can be applied to the
saved period.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sympy import isprime

from chromatic_research.campaigns.d6_torus_column_generation import (
    build_cayley_graph,
    independent_set_target_cpsat,
    source_extension_coordinates,
    source_fiber_columns,
    subspace_extension_profiles,
    subspace_refinement_period,
)
from chromatic_research.core.determinant_repair import exact_det, load_preset
from chromatic_research.core.prime_radon import gaussian_binomial, rref_subspaces
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


def parse_cases(text: str) -> list[tuple[int, int]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("expected [[prime, rank], ...]")
    cases: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise argparse.ArgumentTypeError("each case must be [prime, rank]")
        prime, rank = (int(item[0]), int(item[1]))
        if not isprime(prime) or not 1 <= rank <= 6:
            raise argparse.ArgumentTypeError("invalid prime/rank case")
        if prime**rank > 16:
            raise argparse.ArgumentTypeError(
                "current exact bit-mask implementation requires p^rank <= 16"
            )
        cases.append((prime, rank))
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=parse_cases, default=[(2, 2), (2, 3), (3, 2)]
    )
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--decision-sparsest", type=int, default=0)
    parser.add_argument("--decision-per-profile", type=int, default=0)
    parser.add_argument("--decision-time-limit", type=float, default=60.0)
    parser.add_argument("--decision-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.target_colors < 1:
        parser.error("target colors must be positive")
    if args.decision_sparsest < 0 or args.decision_per_profile < 0:
        parser.error("decision budgets must be nonnegative")
    if args.decision_time_limit <= 0 or args.decision_workers < 1:
        parser.error("invalid decision solver settings")

    lattice, basis, diameter, _, source_kernel = load_preset("d6")
    forbidden, _, _ = _forbidden_with_weights(basis, diameter)
    coordinates, class_ids, source_order = source_extension_coordinates(
        source_kernel, forbidden
    )
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "all RREF prime-field composite covers of the exact 343 kernel; "
            "exact residue profiles plus CP-SAT target-independent-set screens"
        ),
        "lattice": lattice,
        "dimension": len(source_kernel),
        "source_index": source_order,
        "source_kernel_determinant": abs(exact_det(source_kernel)),
        "source_kernel_basis_columns": source_kernel.astype(int).tolist(),
        "target_colors": args.target_colors,
        "forbidden_projective_pairs": len(forbidden),
        "settings": {
            "cases": [list(case) for case in args.cases],
            "decision_sparsest": args.decision_sparsest,
            "decision_per_profile": args.decision_per_profile,
            "decision_time_limit": args.decision_time_limit,
            "decision_workers": args.decision_workers,
        },
        "case_screens": [],
        "exceptional_graphs": [],
        "promising_periods": [],
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"source={source_order} target={args.target_colors} "
        f"signed-forbidden={len(coordinates)}",
        flush=True,
    )
    for prime, rank in args.cases:
        expected = gaussian_binomial(6, rank, prime)
        subspaces = list(rref_subspaces(6, prime, rank))
        if len(subspaces) != expected:
            raise AssertionError("RREF subspace enumeration count mismatch")
        profile = subspace_extension_profiles(
            coordinates,
            class_ids,
            source_order,
            subspaces,
            prime,
        )
        incomplete = profile["incomplete_subspace_indices"]
        public = {
            key: value
            for key, value in profile.items()
            if not key.startswith("_")
            and key != "incomplete_subspace_indices"
        }
        public["incomplete_subspace_count"] = len(incomplete)
        public["incomplete_subspace_indices_preview"] = incomplete[:40]
        necessary_alpha = math.ceil(
            profile["period_order"] / args.target_colors
        )
        public["necessary_independence_number_for_target"] = necessary_alpha

        ranked = sorted(
            incomplete,
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        selected = set(ranked[: args.decision_sparsest])
        if args.decision_per_profile:
            by_profile: dict[tuple[int, int], list[int]] = {}
            for index in ranked:
                key = (
                    int(profile["_connection_counts"][index]),
                    int(profile["_minimum_coverages"][index]),
                )
                by_profile.setdefault(key, []).append(index)
            for indices in by_profile.values():
                selected.update(indices[: args.decision_per_profile])
        selected_ordered = sorted(
            selected,
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        public["decision_selected"] = len(selected_ordered)
        public["decision_unscreened"] = len(incomplete) - len(selected_ordered)
        payload["case_screens"].append(public)
        save()
        print(
            f"p={prime} rank={rank} q={prime**rank} "
            f"subspaces={len(subspaces)} "
            f"connections=[{profile['connection_count_minimum']},"
            f"{profile['connection_count_maximum']}] "
            f"complete={profile['complete_multipartite_subspaces']} "
            f"selected={len(selected_ordered)}",
            flush=True,
        )

        for subspace_index in selected_ordered:
            rows = subspaces[subspace_index]
            period = subspace_refinement_period(
                source_kernel, rows, prime
            )
            graph = build_cayley_graph(period, forbidden)
            fibers = source_fiber_columns(graph, source_kernel)
            fiber_sizes = {len(column) for column in fibers}
            if fiber_sizes != {prime**rank}:
                raise AssertionError("unexpected composite source-fiber size")
            decision = independent_set_target_cpsat(
                graph,
                necessary_alpha,
                time_limit=args.decision_time_limit,
                workers=args.decision_workers,
            )
            record: dict = {
                "prime": prime,
                "rank": rank,
                "quotient_size": prime**rank,
                "subspace_index": int(subspace_index),
                "rows": rows.astype(int).tolist(),
                "period_basis_columns": period.astype(int).tolist(),
                "period_index": graph.vertex_count,
                "connection_keys": len(graph.connection_keys),
                "minimum_residues_per_source_class": int(
                    profile["_minimum_coverages"][subspace_index]
                ),
                "source_fiber_size": prime**rank,
                "target_independence_number": necessary_alpha,
                "target_independent_set": decision,
            }
            if decision["proven_infeasible"]:
                record["independence_number"] = prime**rank
                record["fractional_chromatic_number"] = float(source_order)
            elif decision["feasible"]:
                record["independence_number_lower_bound"] = len(
                    decision["vertices"]
                )
                record["fractional_chromatic_upper_bound"] = (
                    graph.vertex_count / len(decision["vertices"])
                )
                payload["promising_periods"].append(record)
            payload["exceptional_graphs"].append(record)
            save()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
