"""Hybrid search for a large-prime block and a small primary block.

The complete projective-row method in ``block_row_metric_opt.py`` is ideal
while every row pool is modest.  It becomes impractical for, for example,

    134 = 67 * 2,

because P^4(F_67) has more than twenty million points.  This module never
materializes that pool.  Instead it alternates:

* an exact all-values coordinate (and optional coordinate-pair) best response
  for the large-prime row; and
* a global best response over the complete projective pool of the small
  prime-power block.

Every coordinate neighborhood is scored exactly in O(p + |F|), and every
reported incumbent is rechecked with exact modular arithmetic and the complete
geometric separation oracle.  Geometry-weighted objectives are supported so
that near misses can seed a subsequent metric deformation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import combigeo
from block_row_metric_opt import candidate_record
from prime_radon import projective_forms
from prime_row_opt import (
    _best_coordinate_values,
    _best_pair_values,
    _cached_forbidden_with_weights,
    _cost_cutoff,
    _is_prime,
    _objective_key,
    _source_lattice,
)


def joint_score(
    forbidden: np.ndarray,
    weights: np.ndarray,
    large_row: np.ndarray,
    large_prime: int,
    small_row: np.ndarray,
    small_modulus: int,
) -> tuple[int, float]:
    mask = (
        ((forbidden @ large_row) % large_prime == 0)
        & ((forbidden @ small_row) % small_modulus == 0)
    )
    return int(np.count_nonzero(mask)), float(weights[mask].sum())


def best_small_response(
    forbidden: np.ndarray,
    weights: np.ndarray,
    large_row: np.ndarray,
    large_prime: int,
    small_modulus: int,
    small_pool: np.ndarray,
    objective: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a globally optimal small-block row for the fixed large row."""
    active = (forbidden @ large_row) % large_prime == 0
    vectors = forbidden[active]
    active_weights = weights[active]
    if not len(vectors):
        return small_pool[int(rng.integers(len(small_pool)))].copy()

    zero = (vectors @ small_pool.T) % small_modulus == 0
    counts = np.count_nonzero(zero, axis=0)
    costs = active_weights @ zero
    if objective == "weighted":
        minimum_cost = float(costs.min())
        candidates = np.flatnonzero(costs <= _cost_cutoff(minimum_cost))
        minimum_count = int(counts[candidates].min())
        candidates = candidates[counts[candidates] == minimum_count]
    else:
        minimum_count = int(counts.min())
        candidates = np.flatnonzero(counts == minimum_count)
        minimum_cost = float(costs[candidates].min())
        candidates = candidates[
            costs[candidates] <= _cost_cutoff(minimum_cost)
        ]
    return small_pool[int(rng.choice(candidates))].copy()


def choose_coordinate_value(
    forbidden: np.ndarray,
    weights: np.ndarray,
    row: np.ndarray,
    dots: np.ndarray,
    coordinate: int,
    prime: int,
    objective: str,
    rng: np.random.Generator,
) -> int:
    """Exact coordinate best response, excluding the all-zero row."""
    candidates, counts, costs = _best_coordinate_values(
        forbidden,
        weights,
        row,
        dots,
        coordinate,
        prime,
        objective,
    )
    other_nonzero = np.any(
        np.delete(row % prime, coordinate) != 0
    )
    if not other_nonzero:
        candidates = candidates[candidates != 0]
        if not len(candidates):
            allowed = np.arange(1, prime, dtype=np.int64)
            if objective == "weighted":
                minimum_cost = float(costs[allowed].min())
                allowed = allowed[
                    costs[allowed] <= _cost_cutoff(minimum_cost)
                ]
                minimum_count = int(counts[allowed].min())
                candidates = allowed[counts[allowed] == minimum_count]
            else:
                minimum_count = int(counts[allowed].min())
                allowed = allowed[counts[allowed] == minimum_count]
                minimum_cost = float(costs[allowed].min())
                candidates = allowed[
                    costs[allowed] <= _cost_cutoff(minimum_cost)
                ]
    return int(rng.choice(candidates))


def normalize_prime_row(row: np.ndarray, prime: int) -> np.ndarray:
    result = np.asarray(row, dtype=np.int64).copy() % prime
    first = next(int(value) for value in result if value)
    result = result * pow(first, -1, prime) % prime
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--large-prime", type=int, required=True)
    parser.add_argument("--small-modulus", type=int, required=True)
    parser.add_argument("--restarts", type=int, default=20_000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    parser.add_argument("--kick", type=int, default=2)
    parser.add_argument("--pair-moves", type=int, default=1)
    parser.add_argument(
        "--objective",
        choices=("lexicographic", "weighted"),
        default="lexicographic",
    )
    parser.add_argument(
        "--weight-power",
        type=float,
        default=2.0,
        help="use (1-distance_ratio)^power as the geometric cost",
    )
    parser.add_argument(
        "--forbidden-cache",
        type=Path,
        help=(
            "optional .npz cache for metric-specific forbidden coordinates "
            "and ratios; weights are recomputed for each objective power"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not _is_prime(args.large_prime):
        parser.error("--large-prime must be prime")
    if args.small_modulus < 2:
        parser.error("--small-modulus must be at least two")
    if math.gcd(args.large_prime, args.small_modulus) != 1:
        parser.error("the two block moduli must be coprime")
    if args.restarts < 1 or args.sweeps < 1:
        parser.error("restarts and sweeps must be positive")
    if args.stall_sweeps < 1 or args.kick < 1 or args.pair_moves < 0:
        parser.error("stall/kick must be positive and pair-moves nonnegative")
    if not math.isfinite(args.weight_power) or args.weight_power <= 0:
        parser.error("--weight-power must be finite and positive")

    metric = json.loads(args.metric.read_text())
    lattice = _source_lattice(args.metric, metric)
    basis = np.asarray(metric["best"]["basis"], dtype=np.float64)
    diameter = float(metric["best"]["diameter"])
    facets = combigeo.relevant_facets(basis.tolist())
    try:
        forbidden, ratios, weights = _cached_forbidden_with_weights(
            basis,
            diameter,
            args.forbidden_cache,
            weight_power=args.weight_power,
        )
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    n = len(basis)
    large_prime = args.large_prime
    small_modulus = args.small_modulus
    small_pool = projective_forms(n, small_modulus)
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    seed_pair: tuple[np.ndarray, np.ndarray] | None = None
    source_record = metric.get("source_record", {})
    if (
        source_record.get("moduli") == [large_prime, small_modulus]
        and len(source_record.get("rows", [])) == 2
    ):
        large_seed = (
            np.asarray(source_record["rows"][0], dtype=np.int64)
            % large_prime
        )
        small_seed = (
            np.asarray(source_record["rows"][1], dtype=np.int64)
            % small_modulus
        )
        if np.any(large_seed) and any(
            np.array_equal(small_seed, candidate)
            for candidate in small_pool
        ):
            seed_pair = (large_seed, small_seed)

    payload: dict = {
        "method": (
            "alternating exact coordinate response for a large prime block "
            "and complete projective response for a small primary block"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "n": n,
        "dimension": n,
        "parent_diameter": diameter,
        "parent_facet_count": len(facets),
        "forbidden_projective_pairs": len(forbidden),
        "structures": [[large_prime, small_modulus]],
        "budget": {
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "stall_sweeps": args.stall_sweeps,
            "kick": args.kick,
            "pair_moves": args.pair_moves,
            "objective": args.objective,
            "weight_power": args.weight_power,
            "forbidden_cache": (
                str(args.forbidden_cache)
                if args.forbidden_cache is not None
                else None
            ),
            "seed": args.seed,
            "small_pool_size": len(small_pool),
        },
        "results": [],
        "objective_best": None,
        "best_by_minimum_ratio": None,
        "valid_candidate": None,
    }
    payload["seeded_from_source_record"] = seed_pair is not None
    best_key: tuple[float | int, float | int] = (float("inf"), float("inf"))
    best_minimum_ratio = float("-inf")
    progress_every = max(100, args.restarts // 10)

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    def consider(
        large_row: np.ndarray,
        small_row: np.ndarray,
        restart: int,
        sweep: int,
    ) -> bool:
        nonlocal best_key, best_minimum_ratio
        score = joint_score(
            forbidden,
            weights,
            large_row,
            large_prime,
            small_row,
            small_modulus,
        )
        key = _objective_key(score, args.objective)
        mask = (
            ((forbidden @ large_row) % large_prime == 0)
            & ((forbidden @ small_row) % small_modulus == 0)
        )
        minimum_ratio = (
            float(ratios[mask].min()) if np.any(mask) else float("inf")
        )
        objective_improves = key < best_key
        ratio_improves = minimum_ratio > best_minimum_ratio + 1e-12
        if not objective_improves and not ratio_improves:
            return False
        large_normalized = normalize_prime_row(large_row, large_prime)
        record = candidate_record(
            label=f"{args.objective}-power-{args.weight_power:g}",
            beta=float(args.weight_power),
            rows=[large_normalized, small_row],
            moduli=[large_prime, small_modulus],
            forbidden=forbidden,
            ratios=ratios,
            weights=weights,
            basis=basis,
            diameter=diameter,
            facets=facets,
            search_seconds=time.perf_counter() - started,
            search_metadata={
                "restart": restart + 1,
                "sweep": sweep,
                "score": [score[0], score[1]],
            },
        )
        if record["image_index"] != large_prime * small_modulus:
            raise AssertionError("hybrid rows do not have the requested image")
        if objective_improves:
            best_key = key
            payload["objective_best"] = record
        if ratio_improves:
            best_minimum_ratio = minimum_ratio
            payload["best_by_minimum_ratio"] = record
        payload["results"] = []
        for candidate in (
            payload["objective_best"],
            payload["best_by_minimum_ratio"],
        ):
            if candidate is not None and candidate not in payload["results"]:
                payload["results"].append(candidate)
        payload["restarts_completed"] = restart + 1
        if record.get("complete_separation", {}).get("valid"):
            payload["valid_candidate"] = record
        save()
        labels = []
        if objective_improves:
            labels.append("objective")
        if ratio_improves:
            labels.append("minimum-ratio")
        print(
            f"  new best ({'+'.join(labels)}) "
            f"restart={restart + 1} sweep={sweep}: "
            f"killed={record['killed']} loss={record['weighted_loss']:.9g} "
            f"min-ratio={record['minimum_conflict_ratio']}",
            flush=True,
        )
        if payload["valid_candidate"] is not None:
            print("*** VALID HYBRID BLOCK KERNEL FOUND ***", flush=True)
            return True
        return False

    for restart in range(args.restarts):
        if restart == 0 and seed_pair is not None:
            large_row = seed_pair[0].copy()
            small_row = seed_pair[1].copy()
        else:
            large_row = rng.integers(
                0, large_prime, size=n, dtype=np.int64
            )
            while not np.any(large_row):
                large_row = rng.integers(
                    0, large_prime, size=n, dtype=np.int64
                )
            small_row = small_pool[
                int(rng.integers(len(small_pool)))
            ].copy()
        if restart == 0 and seed_pair is not None:
            # Keep the exact control in the dual archive before the small-row
            # surrogate response is allowed to move to another basin.
            if consider(large_row, small_row, restart, -1):
                return 0
        small_row = best_small_response(
            forbidden,
            weights,
            large_row,
            large_prime,
            small_modulus,
            small_pool,
            args.objective,
            rng,
        )
        if consider(large_row, small_row, restart, 0):
            return 0
        local_key = _objective_key(
            joint_score(
                forbidden,
                weights,
                large_row,
                large_prime,
                small_row,
                small_modulus,
            ),
            args.objective,
        )
        stalled = 0

        for sweep in range(1, args.sweeps + 1):
            active = (forbidden @ small_row) % small_modulus == 0
            active_forbidden = forbidden[active]
            active_weights = weights[active]
            dots = (active_forbidden @ large_row) % large_prime
            for coordinate in rng.permutation(n):
                coordinate = int(coordinate)
                value = choose_coordinate_value(
                    active_forbidden,
                    active_weights,
                    large_row,
                    dots,
                    coordinate,
                    large_prime,
                    args.objective,
                    rng,
                )
                delta = value - int(large_row[coordinate])
                if delta:
                    dots = (
                        dots
                        + delta * active_forbidden[:, coordinate]
                    ) % large_prime
                    large_row[coordinate] = value

            for _ in range(args.pair_moves):
                first, second = sorted(
                    int(value)
                    for value in rng.choice(n, size=2, replace=False)
                )
                candidates, counts, costs = _best_pair_values(
                    active_forbidden,
                    active_weights,
                    large_row,
                    dots,
                    first,
                    second,
                    large_prime,
                    args.objective,
                )
                other = np.delete(large_row % large_prime, [first, second])
                if not np.any(other):
                    candidates = candidates[
                        np.any(candidates != 0, axis=1)
                    ]
                    if not len(candidates):
                        mask = np.ones_like(counts, dtype=bool)
                        mask[0, 0] = False
                        if args.objective == "weighted":
                            minimum_cost = float(costs[mask].min())
                            mask &= costs <= _cost_cutoff(minimum_cost)
                            minimum_count = int(counts[mask].min())
                            mask &= counts == minimum_count
                        else:
                            minimum_count = int(counts[mask].min())
                            mask &= counts == minimum_count
                            minimum_cost = float(costs[mask].min())
                            mask &= costs <= _cost_cutoff(minimum_cost)
                        candidates = np.argwhere(mask)
                first_value, second_value = candidates[
                    int(rng.integers(len(candidates)))
                ]
                first_delta = int(first_value) - int(large_row[first])
                second_delta = int(second_value) - int(large_row[second])
                if first_delta or second_delta:
                    dots = (
                        dots
                        + first_delta * active_forbidden[:, first]
                        + second_delta * active_forbidden[:, second]
                    ) % large_prime
                    large_row[first] = int(first_value)
                    large_row[second] = int(second_value)

            small_row = best_small_response(
                forbidden,
                weights,
                large_row,
                large_prime,
                small_modulus,
                small_pool,
                args.objective,
                rng,
            )
            score = joint_score(
                forbidden,
                weights,
                large_row,
                large_prime,
                small_row,
                small_modulus,
            )
            current_key = _objective_key(score, args.objective)
            if consider(large_row, small_row, restart, sweep):
                return 0
            if current_key < local_key:
                local_key = current_key
                stalled = 0
            else:
                stalled += 1
            if score[0] == 0:
                break
            if stalled >= args.stall_sweeps:
                kick = min(n, args.kick)
                for coordinate in rng.choice(
                    n, size=kick, replace=False
                ):
                    large_row[int(coordinate)] = int(
                        rng.integers(large_prime)
                    )
                if not np.any(large_row):
                    large_row[int(rng.integers(n))] = int(
                        rng.integers(1, large_prime)
                    )
                small_row = small_pool[
                    int(rng.integers(len(small_pool)))
                ].copy()
                stalled = 0

        if (restart + 1) % progress_every == 0:
            print(
                f"  progress {restart + 1}/{args.restarts}: "
                f"best-key={best_key} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

    best_record = (
        payload["best_by_minimum_ratio"] or payload["objective_best"]
    )
    if best_record is None:
        raise AssertionError("search evaluated no candidate")
    payload["restarts_completed"] = args.restarts
    save()
    print(
        f"FINAL killed={best_record['killed']} "
        f"loss={best_record['weighted_loss']:.9g} "
        f"min-ratio={best_record['minimum_conflict_ratio']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
