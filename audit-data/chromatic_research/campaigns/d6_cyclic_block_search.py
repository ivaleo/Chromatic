"""Search consecutive-block cyclic colorings beyond the pair construction.

For ``C`` desired colors and a block size ``b``, choose a primitive cyclic
row modulo ``N=b*C`` and color quotient residues by

    color(r) = floor(r / b).

Two residues in one block differ by one of
``0, +/-1, ..., +/-(b-1)``.  Thus this is a valid periodic coloring whenever
every forbidden displacement has residue in the single interval

    b <= row*f <= N-b  (mod N).

The discrete stage uses exact modular coordinate descent.  For target
difference one the same constraints form an exact integer-linear interval
model, so an optional open-HiGHS MIP can search or certify the selected core.
This directly tests the nonmonotone idea of moving farther out in period
index while keeping the number of colors fixed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.d6_cyclic_hole_search import (
    cyclic_target_highs_interval_mip,
    load_metric_checkpoint,
    parse_indices,
    primitive_cyclic_row,
)
from chromatic_research.core.determinant_repair import exact_det, load_preset
from chromatic_research.core.prime_radon import hnf_columns, kernel_basis, smith_diagonal
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


def block_violation_mask(
    forbidden: np.ndarray,
    row: Sequence[int],
    modulus: int,
    block_size: int,
) -> np.ndarray:
    """Vectors mapping to ``0,+/-1,...,+/-(block_size-1)``."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    row = np.asarray(row, dtype=np.int64)
    residues = np.remainder(forbidden @ row, int(modulus))
    return (residues < int(block_size)) | (
        residues > int(modulus) - int(block_size)
    )


def block_coordinate_scores(
    forbidden: np.ndarray,
    row: np.ndarray,
    coordinate: int,
    modulus: int,
    block_size: int,
) -> np.ndarray:
    """Exact block-violation counts for every coordinate value."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    row = np.asarray(row, dtype=np.int64)
    current = np.remainder(forbidden @ row, modulus)
    coefficient = np.remainder(forbidden[:, coordinate], modulus)
    base = np.remainder(
        current - coefficient * int(row[coordinate]),
        modulus,
    )
    values = np.remainder(
        base[:, None]
        + coefficient[:, None]
        * np.arange(modulus, dtype=np.int64)[None, :],
        modulus,
    )
    return np.count_nonzero(
        (values < block_size) | (values > modulus - block_size),
        axis=0,
    ).astype(np.int64)


def block_coordinate_weighted_scores(
    forbidden: np.ndarray,
    row: np.ndarray,
    coordinate: int,
    modulus: int,
    block_size: int,
    weights: Sequence[float],
) -> np.ndarray:
    """Weighted block-violation sums for every coordinate value."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.shape != (len(forbidden),):
        raise ValueError("one weight is required per forbidden vector")
    current = np.remainder(forbidden @ row, modulus)
    coefficient = np.remainder(forbidden[:, coordinate], modulus)
    base = np.remainder(
        current - coefficient * int(row[coordinate]),
        modulus,
    )
    values = np.remainder(
        base[:, None]
        + coefficient[:, None]
        * np.arange(modulus, dtype=np.int64)[None, :],
        modulus,
    )
    violations = (values < block_size) | (
        values > modulus - block_size
    )
    return weights_array @ violations


def block_weighted_score(
    forbidden: np.ndarray,
    row: Sequence[int],
    modulus: int,
    block_size: int,
    weights: Sequence[float],
) -> float:
    mask = block_violation_mask(
        forbidden,
        row,
        modulus,
        block_size,
    )
    return float(np.asarray(weights, dtype=np.float64)[mask].sum())


def cyclic_block_descent(
    forbidden: np.ndarray,
    modulus: int,
    block_size: int,
    *,
    restarts: int = 100,
    sweeps: int = 20,
    top: int = 12,
    weights: Sequence[float] | None = None,
    weighted_first: bool = False,
    seed: int = 0,
) -> dict:
    """Coordinate descent for a primitive block-avoiding cyclic row."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    if (
        forbidden.ndim != 2
        or modulus < 2 * block_size
        or block_size < 2
        or restarts < 1
        or sweeps < 1
        or top < 1
    ):
        raise ValueError("invalid block search instance or budget")
    if weights is not None and np.asarray(weights).shape != (
        len(forbidden),
    ):
        raise ValueError("one weight is required per forbidden vector")
    rng = np.random.default_rng(seed)
    best_score = len(forbidden) + 1
    best_weighted = float("inf")
    best_row = None
    best_restart = -1
    best_sweep = -1
    started = time.perf_counter()

    def key(count: int, weighted: float) -> tuple[float, float]:
        return (
            (float(weighted), float(count))
            if weighted_first
            else (float(count), float(weighted))
        )

    for restart in range(restarts):
        row = rng.integers(
            0,
            modulus,
            size=forbidden.shape[1],
            dtype=np.int64,
        )
        row[int(rng.integers(len(row)))] = 1
        current_score = int(
            block_violation_mask(
                forbidden,
                row,
                modulus,
                block_size,
            ).sum()
        )
        current_weighted = (
            block_weighted_score(
                forbidden,
                row,
                modulus,
                block_size,
                weights,
            )
            if weights is not None
            else float(current_score)
        )
        for sweep in range(sweeps):
            improved = False
            for coordinate in rng.permutation(forbidden.shape[1]):
                scores = block_coordinate_scores(
                    forbidden,
                    row,
                    int(coordinate),
                    modulus,
                    block_size,
                )
                weighted_scores = (
                    block_coordinate_weighted_scores(
                        forbidden,
                        row,
                        int(coordinate),
                        modulus,
                        block_size,
                        weights,
                    )
                    if weights is not None
                    else scores.astype(np.float64)
                )
                order = (
                    np.lexsort((scores, weighted_scores))
                    if weighted_first
                    else np.lexsort((weighted_scores, scores))
                )
                candidates = order[: min(top, modulus)]
                if weighted_first:
                    best_local = float(weighted_scores[candidates[0]])
                    tolerance = max(1e-12, 0.02 * abs(best_local))
                    near = candidates[
                        weighted_scores[candidates]
                        <= best_local + tolerance
                    ]
                else:
                    best_count = int(scores[candidates[0]])
                    same_count = candidates[scores[candidates] == best_count]
                    best_local = float(weighted_scores[same_count].min())
                    tolerance = max(1e-12, 0.02 * abs(best_local))
                    near = same_count[
                        weighted_scores[same_count]
                        <= best_local + tolerance
                    ]
                chosen = int(near[int(rng.integers(len(near)))])
                chosen_score = int(scores[chosen])
                chosen_weighted = float(weighted_scores[chosen])
                if key(chosen_score, chosen_weighted) <= key(
                    current_score,
                    current_weighted,
                ):
                    improved |= (
                        key(chosen_score, chosen_weighted)
                        < key(current_score, current_weighted)
                        or chosen != int(row[coordinate])
                    )
                    row[coordinate] = chosen
                    current_score = chosen_score
                    current_weighted = chosen_weighted

            if (
                primitive_cyclic_row(row, modulus)
                and key(current_score, current_weighted)
                < key(best_score, best_weighted)
            ):
                best_score = current_score
                best_weighted = current_weighted
                best_row = row.copy()
                best_restart = restart
                best_sweep = sweep
            if current_score == 0 and primitive_cyclic_row(row, modulus):
                return {
                    "success": True,
                    "score": 0,
                    "weighted_score": 0.0,
                    "row": row.astype(int).tolist(),
                    "restart": restart,
                    "sweep": sweep,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            if not improved:
                coordinate = int(rng.integers(len(row)))
                row[coordinate] = int(rng.integers(modulus))
                current_score = int(
                    block_violation_mask(
                        forbidden,
                        row,
                        modulus,
                        block_size,
                    ).sum()
                )
                current_weighted = (
                    block_weighted_score(
                        forbidden,
                        row,
                        modulus,
                        block_size,
                        weights,
                    )
                    if weights is not None
                    else float(current_score)
                )
    return {
        "success": False,
        "score": int(best_score),
        "weighted_score": (
            float(best_weighted) if np.isfinite(best_weighted) else None
        ),
        "row": best_row.astype(int).tolist() if best_row is not None else None,
        "restart": best_restart,
        "sweep": best_sweep,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument(
        "--block-sizes",
        type=parse_indices,
        default=[3],
    )
    parser.add_argument("--restarts", type=int, default=80)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--weight-power", type=float, default=0.0)
    parser.add_argument("--weighted-first", action="store_true")
    parser.add_argument("--search-min-ratio", type=float, default=1.0)
    parser.add_argument("--highs-time-limit", type=float, default=0.0)
    parser.add_argument("--metric-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.target_colors < 1
        or any(block_size < 2 for block_size in args.block_sizes)
        or args.restarts < 1
        or args.sweeps < 1
        or args.top < 1
        or args.weight_power < 0
        or not 0 < args.search_min_ratio <= 1
        or args.highs_time_limit < 0
    ):
        parser.error("invalid color count, block size, or search budget")

    lattice, basis, diameter, _, _ = load_preset("d6")
    metric_payload = None
    if args.metric_checkpoint is not None:
        try:
            basis, diameter, metric_payload = load_metric_checkpoint(
                args.metric_checkpoint,
                basis.shape,
            )
        except ValueError as error:
            parser.error(str(error))
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    search_mask = ratios < args.search_min_ratio - 1e-12
    search_forbidden = forbidden[search_mask]
    search_ratios = ratios[search_mask]
    weights = (
        np.power(
            np.maximum(0.0, 1.0 - search_ratios),
            args.weight_power,
        )
        if args.weight_power > 0
        else None
    )
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "cyclic consecutive-block modular descent with optional exact "
            "HiGHS interval MIP"
        ),
        "lattice": lattice,
        "dimension": len(basis),
        "metric_checkpoint": (
            str(args.metric_checkpoint)
            if args.metric_checkpoint is not None
            else None
        ),
        "metric_checkpoint_method": (
            metric_payload.get("method")
            if metric_payload is not None
            else None
        ),
        "parent_basis": basis.tolist(),
        "parent_diameter": diameter,
        "forbidden_projective_pairs": len(forbidden),
        "target_colors": args.target_colors,
        "settings": {
            "block_sizes": args.block_sizes,
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "weight_power": args.weight_power,
            "weighted_first": args.weighted_first,
            "search_min_ratio": args.search_min_ratio,
            "highs_time_limit": args.highs_time_limit,
            "seed": args.seed,
        },
        "records": [],
        "coloring": None,
        "valid_combinatorial_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for case_number, block_size in enumerate(args.block_sizes):
        modulus = block_size * args.target_colors
        print(
            f"[{case_number+1}/{len(args.block_sizes)}] "
            f"block={block_size} N={modulus} "
            f"core={len(search_forbidden)}",
            flush=True,
        )
        search = cyclic_block_descent(
            search_forbidden,
            modulus,
            block_size,
            restarts=args.restarts,
            sweeps=args.sweeps,
            top=args.top,
            weights=weights,
            weighted_first=args.weighted_first,
            seed=args.seed + 1_000_003 * case_number,
        )
        record: dict = {
            "block_size": block_size,
            "period_index": modulus,
            "search_core_ratio": args.search_min_ratio,
            "search_core_vectors": len(search_forbidden),
            "search": search,
        }
        if not search["success"] and args.highs_time_limit > 0:
            highs = cyclic_target_highs_interval_mip(
                search_forbidden,
                modulus,
                1,
                avoid_radius=block_size - 1,
                time_limit=args.highs_time_limit,
            )
            record["highs_mip"] = highs
            if highs["feasible"]:
                search = {
                    **search,
                    "success": True,
                    "score": 0,
                    "row": highs["row"],
                    "completed_by": "HiGHS interval MIP",
                }
                record["search"] = search
        if search["row"] is not None:
            row = np.asarray(search["row"], dtype=np.int64)
            conflict_mask = block_violation_mask(
                forbidden,
                row,
                modulus,
                block_size,
            )
            conflict_indices = np.flatnonzero(conflict_mask)
            record.update(
                {
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
                                forbidden[index] @ row % modulus
                            ),
                        }
                        for index in conflict_indices
                    ],
                }
            )
            if not len(conflict_indices):
                period = hnf_columns(
                    kernel_basis([row], [modulus], len(basis))
                )
                if abs(exact_det(period)) != modulus:
                    raise AssertionError(
                        "block row has the wrong exact period index"
                    )
                record.update(
                    {
                        "row": row.astype(int).tolist(),
                        "period_basis_columns": period.astype(int).tolist(),
                        "period_smith": smith_diagonal(period),
                        "coloring_rule": (
                            "color(x) = floor((row*x mod N)/block_size)"
                        ),
                        "same_color_quotient_blocks": [
                            list(
                                range(
                                    color * block_size,
                                    (color + 1) * block_size,
                                )
                            )
                            for color in range(args.target_colors)
                        ],
                    }
                )
                payload["coloring"] = record
                payload["valid_combinatorial_witness"] = True
        payload["records"].append(record)
        save()
        print(
            f"  score={search['score']} "
            f"core-success={search['success']} "
            f"full={record.get('full_conflict_count')} "
            f"min={record.get('full_minimum_conflict_ratio')} "
            f"coloring={payload['valid_combinatorial_witness']}",
            flush=True,
        )
        if payload["valid_combinatorial_witness"]:
            break
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
