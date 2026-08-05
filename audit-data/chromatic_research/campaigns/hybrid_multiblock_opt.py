"""Hybrid search for one large-prime row and several small primary blocks.

This is the multi-block extension of ``hybrid_block_row_opt.py``.  It avoids
materializing the projective space of the large prime, while retaining exact
global best responses for every small prime-power row.  It is intended for
quotients such as

    342 = 19 * 9 * 2       and       342 = 19 * 3 * 3 * 2.

One sweep alternates:

* all-values coordinate responses (and optional pair responses) for the large
  prime row;
* complete projective-pool responses for each small block;
* a small random kick after a coordinatewise stall.

Every incumbent is checked for the exact image size and by the complete
geometric separation oracle.
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
from chromatic_research.core.block_row_metric_opt import candidate_record
from chromatic_research.campaigns.hybrid_block_row_opt import choose_coordinate_value, normalize_prime_row
from chromatic_research.core.prime_radon import (
    _prime_power,
    killed_mask,
    projective_forms,
    rank_mod,
    score_forms,
)
from chromatic_research.core.prime_row_opt import (
    _best_coordinate_values,
    _best_pair_values,
    _cached_forbidden_with_weights,
    _cost_cutoff,
    _is_prime,
    _objective_key,
    _source_lattice,
)


def parse_moduli(text: str) -> list[int]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("small moduli must be a nonempty list")
    return [int(value) for value in raw]


def parse_rows(text: str) -> list[list[int]]:
    raw = json.loads(text)
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(row, list) or not row for row in raw)
    ):
        raise argparse.ArgumentTypeError(
            "initial rows must be a nonempty JSON list of nonempty rows"
        )
    try:
        return [[int(value) for value in row] for row in raw]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "initial rows must contain integers"
        ) from error


def joint_score(
    forbidden: np.ndarray,
    weights: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
) -> tuple[int, float]:
    mask = killed_mask(forbidden, rows, moduli)
    return int(mask.sum()), float(weights[mask].sum())


def _independent_candidate(
    candidate: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
    row_index: int,
) -> bool:
    prime, _ = _prime_power(moduli[row_index])
    peers = [
        np.asarray(row, dtype=np.int64) % prime
        for index, (row, modulus) in enumerate(zip(rows, moduli))
        if index != row_index and _prime_power(modulus)[0] == prime
    ]
    if not peers:
        return True
    before = rank_mod(np.asarray(peers), prime)
    after = rank_mod(
        np.vstack(peers + [np.asarray(candidate, dtype=np.int64) % prime]),
        prime,
    )
    return after == before + 1


def best_small_response(
    forbidden: np.ndarray,
    weights: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
    row_index: int,
    pool: np.ndarray,
    objective: str,
    rng: np.random.Generator,
) -> np.ndarray:
    active = killed_mask(forbidden, rows, moduli, skip=row_index)
    vectors = forbidden[active]
    active_weights = weights[active]
    counts = score_forms(vectors, moduli[row_index], pool)
    costs = score_forms(
        vectors,
        moduli[row_index],
        pool,
        active_weights,
    )
    if objective == "weighted":
        order = np.lexsort((counts, costs))
    else:
        order = np.lexsort((costs, counts))
    best_key: tuple[float | int, float | int] | None = None
    candidates: list[np.ndarray] = []
    for candidate_index in order:
        candidate = pool[int(candidate_index)]
        if not _independent_candidate(candidate, rows, moduli, row_index):
            continue
        key = _objective_key(
            (int(round(float(counts[candidate_index]))),
             float(costs[candidate_index])),
            objective,
        )
        if best_key is None:
            best_key = key
        elif float(key[0]) > _cost_cutoff(float(best_key[0])):
            break
        elif float(key[1]) > _cost_cutoff(float(best_key[1])):
            continue
        candidates.append(candidate)
        if len(candidates) >= 32:
            break
    if not candidates:
        raise RuntimeError("no independent small-block response")
    return candidates[int(rng.integers(len(candidates)))].copy()


def random_rows(
    n: int,
    large_prime: int,
    small_moduli: Sequence[int],
    pools: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    large = rng.integers(0, large_prime, size=n, dtype=np.int64)
    while not np.any(large):
        large = rng.integers(0, large_prime, size=n, dtype=np.int64)
    rows = [large]
    moduli = [large_prime] + list(small_moduli)
    for row_index, pool in enumerate(pools, start=1):
        for _ in range(10_000):
            candidate = pool[int(rng.integers(len(pool)))].copy()
            provisional = rows + [candidate]
            if _independent_candidate(
                candidate,
                provisional,
                moduli[: len(provisional)],
                len(provisional) - 1,
            ):
                rows.append(candidate)
                break
        else:
            raise RuntimeError("failed to initialize independent rows")
    return rows


def validated_seed_rows(
    raw_rows: Sequence[Sequence[int]],
    moduli: Sequence[int],
    n: int,
) -> list[np.ndarray]:
    """Normalize and validate a full-rank family of quotient rows.

    A portfolio bridge deliberately has no distinguished ``source_record``.
    Explicit seed rows let the alternating search retain a nearby discrete
    basin on such a metric.  Validation is exact: every prime-power row must
    be primitive, and rows over the same residue characteristic must be
    independent after reduction modulo that prime.
    """

    if len(raw_rows) != len(moduli):
        raise ValueError(
            f"expected {len(moduli)} initial rows, got {len(raw_rows)}"
        )
    rows: list[np.ndarray] = []
    for row_index, (raw_row, modulus) in enumerate(zip(raw_rows, moduli)):
        row = np.asarray(raw_row, dtype=np.int64)
        if row.shape != (n,):
            raise ValueError(
                f"initial row {row_index} has shape {row.shape}, "
                f"expected {(n,)}"
            )
        row = row % modulus
        if math.gcd(int(modulus), *(int(value) for value in row)) != 1:
            raise ValueError(
                f"initial row {row_index} is not primitive modulo {modulus}"
            )
        rows.append(row)
    for row_index, row in enumerate(rows):
        if not _independent_candidate(
            row,
            rows,
            moduli,
            row_index,
        ):
            prime, _ = _prime_power(moduli[row_index])
            raise ValueError(
                f"initial row {row_index} is dependent modulo {prime}"
            )
    return rows


def projective_hamming_distance(
    row: Sequence[int],
    anchor: Sequence[int],
    modulus: int,
) -> int:
    """Hamming distance between two rows modulo multiplication by a unit."""

    candidate = np.asarray(row, dtype=np.int64) % modulus
    reference = np.asarray(anchor, dtype=np.int64) % modulus
    if candidate.shape != reference.shape:
        raise ValueError("projective rows must have the same shape")
    units = [
        value
        for value in range(1, modulus)
        if math.gcd(value, modulus) == 1
    ]
    return min(
        int(np.count_nonzero(candidate != value * reference % modulus))
        for value in units
    )


def projective_trust_pool(
    pool: np.ndarray,
    anchor: Sequence[int],
    modulus: int,
    radius: int,
) -> np.ndarray:
    """Restrict canonical forms to a projective Hamming ball."""

    forms = np.asarray(pool, dtype=np.int64)
    reference = np.asarray(anchor, dtype=np.int64) % modulus
    if forms.ndim != 2 or reference.shape != (forms.shape[1],):
        raise ValueError("trust-pool dimension mismatch")
    best = np.full(len(forms), forms.shape[1] + 1, dtype=np.int64)
    for value in range(1, modulus):
        if math.gcd(value, modulus) == 1:
            best = np.minimum(
                best,
                np.count_nonzero(
                    forms % modulus != value * reference % modulus,
                    axis=1,
                ),
            )
    result = forms[best <= radius]
    if not len(result):
        raise ValueError(
            f"projective trust ball of radius {radius} modulo {modulus} "
            "is empty"
        )
    return result


def _choose_allowed(
    allowed: np.ndarray,
    counts: np.ndarray,
    costs: np.ndarray,
    objective: str,
    rng: np.random.Generator,
) -> int:
    if not len(allowed):
        raise RuntimeError("discrete trust region has no admissible move")
    if objective == "weighted":
        minimum_cost = float(costs[allowed].min())
        candidates = allowed[
            costs[allowed] <= _cost_cutoff(minimum_cost)
        ]
        minimum_count = int(counts[candidates].min())
        candidates = candidates[counts[candidates] == minimum_count]
    else:
        minimum_count = int(counts[allowed].min())
        candidates = allowed[counts[allowed] == minimum_count]
        minimum_cost = float(costs[candidates].min())
        candidates = candidates[
            costs[candidates] <= _cost_cutoff(minimum_cost)
        ]
    return int(rng.choice(candidates))


def trusted_coordinate_value(
    forbidden: np.ndarray,
    weights: np.ndarray,
    row: np.ndarray,
    dots: np.ndarray,
    coordinate: int,
    prime: int,
    objective: str,
    rng: np.random.Generator,
    anchor: np.ndarray,
    radius: int,
) -> int:
    """Exact best coordinate response inside a projective Hamming ball."""

    _, counts, costs = _best_coordinate_values(
        forbidden,
        weights,
        row,
        dots,
        coordinate,
        prime,
        objective,
    )
    allowed: list[int] = []
    for value in range(prime):
        candidate = row.copy()
        candidate[coordinate] = value
        if not np.any(candidate % prime):
            continue
        if projective_hamming_distance(
            candidate,
            anchor,
            prime,
        ) <= radius:
            allowed.append(value)
    return _choose_allowed(
        np.asarray(allowed, dtype=np.int64),
        counts,
        costs,
        objective,
        rng,
    )


def trusted_pair_values(
    row: np.ndarray,
    first: int,
    second: int,
    prime: int,
    anchor: np.ndarray,
    radius: int,
) -> np.ndarray:
    """All admissible simultaneous values for a trusted coordinate pair."""

    candidates: list[tuple[int, int]] = []
    for first_value in range(prime):
        for second_value in range(prime):
            trial = row.copy()
            trial[first] = first_value
            trial[second] = second_value
            if not np.any(trial % prime):
                continue
            if projective_hamming_distance(
                trial,
                anchor,
                prime,
            ) <= radius:
                candidates.append((first_value, second_value))
    return np.asarray(candidates, dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--large-prime", type=int, required=True)
    parser.add_argument("--small-moduli", type=parse_moduli, required=True)
    parser.add_argument("--restarts", type=int, default=2000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--stall-sweeps", type=int, default=3)
    parser.add_argument("--kick", type=int, default=2)
    parser.add_argument("--pair-moves", type=int, default=1)
    parser.add_argument(
        "--source-restarts",
        type=int,
        default=0,
        help=(
            "number of initial restarts seeded from source_record rows; "
            "restart zero is exact and later ones perturb the large row"
        ),
    )
    parser.add_argument(
        "--source-kick",
        type=int,
        default=1,
        help="large-row coordinates changed in each perturbed source restart",
    )
    parser.add_argument(
        "--initial-rows",
        type=parse_rows,
        help=(
            "explicit JSON rows used for the seeded restarts; overrides "
            "source_record and enables local continuation from a portfolio "
            "bridge metric"
        ),
    )
    parser.add_argument(
        "--trust-radius",
        type=int,
        help=(
            "for seeded restarts, constrain every quotient row to this "
            "projective Hamming radius from the explicit/source rows"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("lexicographic", "weighted"),
        default="weighted",
    )
    parser.add_argument("--weight-power", type=float, default=2.0)
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
    try:
        for value in args.small_moduli:
            _prime_power(value)
    except ValueError as error:
        parser.error(str(error))
    if any(
        math.gcd(args.large_prime, modulus) != 1
        for modulus in args.small_moduli
    ):
        parser.error("large prime must be coprime to every small block")
    if args.restarts < 1 or args.sweeps < 1 or args.stall_sweeps < 1:
        parser.error("restart/sweep/stall budgets must be positive")
    if (
        args.kick < 1
        or args.pair_moves < 0
        or args.source_restarts < 0
        or args.source_kick < 1
        or (args.trust_radius is not None and args.trust_radius < 0)
    ):
        parser.error("kick must be positive and pair moves nonnegative")
    if not math.isfinite(args.weight_power) or args.weight_power <= 0:
        parser.error("weight power must be finite and positive")

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
    moduli = [args.large_prime] + args.small_moduli
    pools = [projective_forms(n, modulus) for modulus in args.small_moduli]
    rng = np.random.default_rng(args.seed)
    source_record = metric.get("source_record", {})
    source_rows: list[np.ndarray] | None = None
    seed_rows_source: str | None = None
    if args.initial_rows is not None:
        try:
            source_rows = validated_seed_rows(args.initial_rows, moduli, n)
        except ValueError as error:
            parser.error(str(error))
        seed_rows_source = "explicit"
    elif source_record.get("moduli") == moduli:
        try:
            source_rows = validated_seed_rows(
                source_record.get("rows", []),
                moduli,
                n,
            )
        except ValueError:
            # A source record is only an optional warm start.  Historical
            # payloads with incomplete metadata must not make a fresh search
            # unusable.
            source_rows = None
        else:
            seed_rows_source = "source_record"
    if args.trust_radius is not None and source_rows is None:
        parser.error("--trust-radius requires usable initial/source rows")
    trust_pools: list[np.ndarray] | None = None
    if args.trust_radius is not None and source_rows is not None:
        try:
            trust_pools = [
                projective_trust_pool(
                    pool,
                    source_rows[row_index],
                    moduli[row_index],
                    args.trust_radius,
                )
                for row_index, pool in enumerate(pools, start=1)
            ]
        except ValueError as error:
            parser.error(str(error))
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "hybrid exact large-prime coordinate and complete small-block "
            "best-response search"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "n": n,
        "dimension": n,
        "parent_diameter": diameter,
        "parent_facet_count": len(facets),
        "forbidden_projective_pairs": len(forbidden),
        "structures": [moduli],
        "budget": {
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "stall_sweeps": args.stall_sweeps,
            "kick": args.kick,
            "pair_moves": args.pair_moves,
            "source_restarts": args.source_restarts,
            "source_kick": args.source_kick,
            "initial_rows": args.initial_rows,
            "trust_radius": args.trust_radius,
            "trust_pool_sizes": (
                [len(pool) for pool in trust_pools]
                if trust_pools is not None
                else None
            ),
            "objective": args.objective,
            "weight_power": args.weight_power,
            "forbidden_cache": (
                str(args.forbidden_cache)
                if args.forbidden_cache is not None
                else None
            ),
            "seed": args.seed,
            "small_pool_sizes": [len(pool) for pool in pools],
        },
        "results": [],
        "objective_best": None,
        "best_by_minimum_ratio": None,
        "valid_candidate": None,
    }
    payload["seeded_from_source_record"] = (
        seed_rows_source == "source_record"
    )
    payload["seed_rows_source"] = seed_rows_source
    best_key: tuple[float | int, float | int] = (
        float("inf"),
        float("inf"),
    )
    best_minimum_ratio = float("-inf")

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    def consider(
        rows: Sequence[np.ndarray], restart: int, sweep: int
    ) -> bool:
        nonlocal best_key, best_minimum_ratio
        score = joint_score(forbidden, weights, rows, moduli)
        key = _objective_key(score, args.objective)
        mask = killed_mask(forbidden, rows, moduli)
        minimum_ratio = (
            float(ratios[mask].min()) if np.any(mask) else float("inf")
        )
        objective_improves = key < best_key
        ratio_improves = minimum_ratio > best_minimum_ratio + 1e-12
        if not objective_improves and not ratio_improves:
            return False
        normalized = [
            normalize_prime_row(rows[0], args.large_prime),
            *[np.asarray(row, dtype=np.int64) for row in rows[1:]],
        ]
        record = candidate_record(
            label=f"{args.objective}-power-{args.weight_power:g}",
            beta=float(args.weight_power),
            rows=normalized,
            moduli=moduli,
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
        if record["image_index"] != math.prod(moduli):
            raise AssertionError("hybrid rows lost the requested image size")
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
            f"new best ({'+'.join(labels)}) restart={restart + 1} "
            f"sweep={sweep}: "
            f"killed={record['killed']} loss={record['weighted_loss']:.8g} "
            f"min-ratio={record['minimum_conflict_ratio']}",
            flush=True,
        )
        return payload["valid_candidate"] is not None

    progress_every = max(10, args.restarts // 10)
    for restart in range(args.restarts):
        use_source = (
            source_rows is not None
            and restart < max(1, args.source_restarts)
        )
        if use_source:
            rows = [row.copy() for row in source_rows]
            active_pools = trust_pools if trust_pools is not None else pools
            if restart > 0:
                maximum_kick = (
                    min(args.source_kick, args.trust_radius)
                    if args.trust_radius is not None
                    else args.source_kick
                )
                if maximum_kick:
                    for coordinate in rng.choice(
                        n,
                        size=min(maximum_kick, n),
                        replace=False,
                    ):
                        rows[0][int(coordinate)] = int(
                            rng.integers(args.large_prime)
                        )
                if not np.any(rows[0]):
                    rows[0][int(rng.integers(n))] = int(
                        rng.integers(1, args.large_prime)
                    )
            # Preserve the exact control before any surrogate best response.
            if consider(rows, restart, -1):
                return 0
        else:
            active_pools = pools
            rows = random_rows(
                n,
                args.large_prime,
                args.small_moduli,
                pools,
                rng,
            )
        for row_index, pool in enumerate(active_pools, start=1):
            rows[row_index] = best_small_response(
                forbidden,
                weights,
                rows,
                moduli,
                row_index,
                pool,
                args.objective,
                rng,
            )
        if consider(rows, restart, 0):
            return 0
        local_key = _objective_key(
            joint_score(forbidden, weights, rows, moduli),
            args.objective,
        )
        stalled = 0
        for sweep in range(1, args.sweeps + 1):
            old_key = local_key
            active = killed_mask(forbidden, rows[1:], moduli[1:])
            vectors = forbidden[active]
            active_weights = weights[active]
            dots = (vectors @ rows[0]) % args.large_prime
            for coordinate in rng.permutation(n):
                if use_source and args.trust_radius is not None:
                    value = trusted_coordinate_value(
                        vectors,
                        active_weights,
                        rows[0],
                        dots,
                        int(coordinate),
                        args.large_prime,
                        args.objective,
                        rng,
                        source_rows[0],
                        args.trust_radius,
                    )
                else:
                    value = choose_coordinate_value(
                        vectors,
                        active_weights,
                        rows[0],
                        dots,
                        int(coordinate),
                        args.large_prime,
                        args.objective,
                        rng,
                    )
                if value != rows[0][coordinate]:
                    dots = (
                        dots
                        + (value - rows[0][coordinate])
                        * vectors[:, coordinate]
                    ) % args.large_prime
                    rows[0][coordinate] = value
            for _ in range(args.pair_moves):
                first, second = rng.choice(n, size=2, replace=False)
                candidates, counts, costs = _best_pair_values(
                    vectors,
                    active_weights,
                    rows[0],
                    dots,
                    int(first),
                    int(second),
                    args.large_prime,
                    args.objective,
                )
                if use_source and args.trust_radius is not None:
                    allowed = trusted_pair_values(
                        rows[0],
                        int(first),
                        int(second),
                        args.large_prime,
                        source_rows[0],
                        args.trust_radius,
                    )
                    allowed_mask = np.zeros(
                        (args.large_prime, args.large_prime),
                        dtype=bool,
                    )
                    allowed_mask[allowed[:, 0], allowed[:, 1]] = True
                    if args.objective == "weighted":
                        minimum_cost = float(costs[allowed_mask].min())
                        mask = allowed_mask & (
                            costs <= _cost_cutoff(minimum_cost)
                        )
                        minimum_count = int(counts[mask].min())
                        mask &= counts == minimum_count
                    else:
                        minimum_count = int(counts[allowed_mask].min())
                        mask = allowed_mask & (counts == minimum_count)
                        minimum_cost = float(costs[mask].min())
                        mask &= costs <= _cost_cutoff(minimum_cost)
                    candidates = np.argwhere(mask)
                if not np.any(
                    np.delete(rows[0] % args.large_prime, [first, second])
                ):
                    candidates = candidates[np.any(candidates != 0, axis=1)]
                    if not len(candidates):
                        continue
                selected = candidates[int(rng.integers(len(candidates)))]
                old_first, old_second = rows[0][first], rows[0][second]
                rows[0][first], rows[0][second] = map(int, selected)
                dots = (
                    dots
                    + (rows[0][first] - old_first) * vectors[:, first]
                    + (rows[0][second] - old_second) * vectors[:, second]
                ) % args.large_prime
            for row_index in rng.permutation(
                np.arange(1, len(rows), dtype=np.int64)
            ):
                row_index = int(row_index)
                rows[row_index] = best_small_response(
                    forbidden,
                    weights,
                    rows,
                    moduli,
                    row_index,
                    active_pools[row_index - 1],
                    args.objective,
                    rng,
                )
            local_key = _objective_key(
                joint_score(forbidden, weights, rows, moduli),
                args.objective,
            )
            if consider(rows, restart, sweep):
                return 0
            if local_key < old_key:
                stalled = 0
            else:
                stalled += 1
            if stalled >= args.stall_sweeps:
                for coordinate in rng.choice(
                    n, size=min(args.kick, n), replace=False
                ):
                    proposed = rows[0].copy()
                    proposed[coordinate] = int(
                        rng.integers(args.large_prime)
                    )
                    if (
                        not use_source
                        or args.trust_radius is None
                        or (
                            np.any(proposed % args.large_prime)
                            and projective_hamming_distance(
                                proposed,
                                source_rows[0],
                                args.large_prime,
                            )
                            <= args.trust_radius
                        )
                    ):
                        rows[0] = proposed
                stalled = 0
        if (restart + 1) % progress_every == 0:
            print(
                f"progress {restart + 1}/{args.restarts} best={best_key}",
                flush=True,
            )
            payload["restarts_completed"] = restart + 1
            save()

    payload["restarts_completed"] = args.restarts
    save()
    print(f"FINAL best={best_key} saved={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
