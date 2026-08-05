"""HiGHS neighborhood portfolio for a cyclic affine coloring.

The cyclic coordinate descent often stops at a row whose bad modular images
cannot be repaired by changing one coordinate.  For target difference one,
the exact constraints are an integer-linear interval system.  This script
therefore fixes all but two, three, or another requested number of row
coordinates and lets HiGHS search every such coordinate neighborhood exactly.

The metric checkpoint is independently rechecked.  Modular arithmetic and
period indices are exact; the forbidden catalogue and its distance ratios are
numerical until a final candidate is rationalized and independently verified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.d6_cyclic_hole_search import (
    cyclic_target_highs_interval_mip,
    load_metric_checkpoint,
    parse_indices,
    parse_row,
    primitive_cyclic_row,
    target_violation_mask,
)
from chromatic_research.campaigns.d6_torus_period_portfolio import (
    quotient_matching_coloring,
    signed_connection_images,
)
from chromatic_research.core.determinant_repair import exact_det, load_preset
from chromatic_research.core.prime_radon import hnf_columns, kernel_basis, smith_diagonal
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric_checkpoint", type=Path)
    parser.add_argument("--modulus", type=int, default=684)
    parser.add_argument("--target-difference", type=int, default=1)
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--initial-row", type=parse_row, required=True)
    parser.add_argument(
        "--free-counts",
        type=parse_indices,
        default=[2, 3],
    )
    parser.add_argument("--search-min-ratio", type=float, default=1.0)
    parser.add_argument("--time-per-model", type=float, default=3.0)
    parser.add_argument(
        "--max-models-per-count",
        type=int,
        default=0,
        help="zero tests every coordinate subset",
    )
    parser.add_argument("--matching-time-limit", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.modulus < 4
        or args.target_difference % args.modulus
        not in {1, args.modulus - 1}
        or args.target_colors < 1
        or not 0 < args.search_min_ratio <= 1
        or args.time_per_model <= 0
        or args.max_models_per_count < 0
        or args.matching_time_limit <= 0
    ):
        parser.error("invalid modulus, target, ratio, or budget")

    lattice, preset_basis, _, _, _ = load_preset("d6")
    try:
        basis, diameter, metric_payload = load_metric_checkpoint(
            args.metric_checkpoint,
            preset_basis.shape,
        )
    except ValueError as error:
        parser.error(str(error))
    dimension = len(basis)
    if (
        len(args.initial_row) != dimension
        or not primitive_cyclic_row(args.initial_row, args.modulus)
        or any(
            free_count < 1 or free_count > dimension
            for free_count in args.free_counts
        )
    ):
        parser.error("invalid initial row or free-coordinate count")

    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    search_mask = ratios < args.search_min_ratio - 1e-12
    search_forbidden = forbidden[search_mask]
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "exact HiGHS interval-MIP coordinate-neighborhood portfolio"
        ),
        "lattice": lattice,
        "metric_checkpoint": str(args.metric_checkpoint),
        "metric_checkpoint_method": metric_payload.get("method"),
        "parent_basis": basis.tolist(),
        "parent_diameter": diameter,
        "modulus": args.modulus,
        "target_difference": args.target_difference,
        "target_colors": args.target_colors,
        "initial_row": args.initial_row,
        "forbidden_projective_pairs": len(forbidden),
        "search_core_ratio": args.search_min_ratio,
        "search_core_vectors": len(search_forbidden),
        "settings": {
            "free_counts": args.free_counts,
            "time_per_model": args.time_per_model,
            "max_models_per_count": args.max_models_per_count,
            "matching_time_limit": args.matching_time_limit,
        },
        "runs": [],
        "best": None,
        "coloring": None,
        "valid_combinatorial_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    def candidate_key(record: dict) -> tuple[float, int]:
        minimum = record.get("full_minimum_conflict_ratio")
        return (
            1.0 if minimum is None else float(minimum),
            -int(record.get("full_conflict_count", len(forbidden) + 1)),
        )

    initial = np.asarray(args.initial_row, dtype=np.int64)
    for free_count in args.free_counts:
        subsets = list(itertools.combinations(range(dimension), free_count))
        if args.max_models_per_count:
            subsets = subsets[: args.max_models_per_count]
        print(
            f"free={free_count} models={len(subsets)} "
            f"core={len(search_forbidden)}",
            flush=True,
        )
        for model_number, free_coordinates in enumerate(subsets, start=1):
            result = cyclic_target_highs_interval_mip(
                search_forbidden,
                args.modulus,
                args.target_difference,
                time_limit=args.time_per_model,
                fixed_row=initial,
                free_coordinates=free_coordinates,
            )
            record: dict = {
                "free_count": free_count,
                "model_number": model_number,
                "free_coordinates": list(free_coordinates),
                "highs_mip": result,
            }
            if result["feasible"]:
                row = np.asarray(result["row"], dtype=np.int64)
                conflict_mask = target_violation_mask(
                    forbidden,
                    row,
                    args.modulus,
                    args.target_difference,
                )
                conflict_indices = np.flatnonzero(conflict_mask)
                record.update(
                    {
                        "row": row.astype(int).tolist(),
                        "full_conflict_count": int(len(conflict_indices)),
                        "full_minimum_conflict_ratio": (
                            float(ratios[conflict_indices].min())
                            if len(conflict_indices)
                            else None
                        ),
                        "full_conflicts": [
                            {
                                "coordinate": forbidden[index]
                                .astype(int)
                                .tolist(),
                                "distance_ratio": float(ratios[index]),
                                "quotient_residue": int(
                                    forbidden[index] @ row % args.modulus
                                ),
                            }
                            for index in conflict_indices
                        ],
                    }
                )
                if (
                    payload["best"] is None
                    or candidate_key(record) > candidate_key(payload["best"])
                ):
                    payload["best"] = record
                if not len(conflict_indices):
                    connections = signed_connection_images(
                        forbidden,
                        [row],
                        [args.modulus],
                    )
                    matching = quotient_matching_coloring(
                        connections,
                        [args.modulus],
                        args.target_colors,
                        time_limit=args.matching_time_limit,
                    )
                    period = hnf_columns(
                        kernel_basis(
                            [row],
                            [args.modulus],
                            dimension,
                        )
                    )
                    if abs(exact_det(period)) != args.modulus:
                        raise AssertionError(
                            "valid cyclic row has the wrong period index"
                        )
                    record.update(
                        {
                            "period_basis_columns": period.astype(int).tolist(),
                            "period_smith": smith_diagonal(period),
                            "matching_coloring": matching,
                        }
                    )
                    if matching["success"]:
                        payload["coloring"] = record
                        payload["valid_combinatorial_witness"] = True
            payload["runs"].append(record)
            save()
            print(
                f"  {model_number:2d}/{len(subsets)} "
                f"free={list(free_coordinates)} "
                f"status={result['status']} "
                f"feasible={result['feasible']} "
                f"full={record.get('full_conflict_count')}",
                flush=True,
            )
            if payload["valid_combinatorial_witness"]:
                break
        if payload["valid_combinatorial_witness"]:
            break
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
