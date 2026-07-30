"""Fast one-row prime-quotient search on a deformed parent lattice.

For a prime index p, every coloring kernel is

    ker(a : Z^n -> F_p),  a != 0.

A forbidden vector f conflicts exactly when ``a.f = 0 mod p``.  Generic
min-conflicts code spends most of its time proposing scalar coefficient
changes.  Here one coordinate update scores *all p values at once*: for every
constraint with f_j != 0 there is exactly one value of a_j that creates a
conflict.  A pair of ``bincount`` operations therefore gives the exact
coordinate neighborhood in O(|F|+p).

Two objectives are available.  ``lexicographic`` first minimizes the number
of conflicting projective +/- pairs and then their squared geometric
deficits.  ``weighted`` reverses those priorities, which is useful during
alternating continuation: several shallow conflicts can be much easier for
the metric oracle than one deep conflict.  Random kicks escape coordinatewise
local minima.  With ``--pareto-size`` the search additionally retains a
nondominated archive in three coordinates: conflict count and total deficit
are minimized, while the worst conflict ratio is maximized.  This exposes
geometrically different kernels to the subsequent metric-deformation stage
instead of collapsing the campaign to one scalar objective.  Any zero-conflict
row is subjected to the complete sublattice separation oracle before it is
reported.
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
from lazy_prime_campaign import canonical_rows, separate_kernel
from prime_radon import hnf_columns, kernel_basis, smith_diagonal


_INVERSE_CACHE: dict[int, np.ndarray] = {}


def _cost_cutoff(minimum: float) -> float:
    """Scale-aware upper cutoff for floating weighted-cost ties.

    A fixed absolute tolerance incorrectly merges distinct objectives when
    high powers of a small geometric deficit make every cost much smaller
    than 1e-15.  The accumulated costs are nonnegative, so a relative
    tolerance at the scale of the actual minimum preserves numerical ties
    without erasing the ordering.
    """
    minimum = float(minimum)
    tolerance = max(
        1e-12 * abs(minimum),
        16.0 * float(np.spacing(abs(minimum))),
    )
    return minimum + tolerance


def _is_prime(value: int) -> bool:
    return value >= 2 and all(
        value % divisor
        for divisor in range(2, int(value**0.5) + 1)
    )


def _source_lattice(metric_path: Path, metric: dict) -> str:
    direct = metric.get("lattice")
    if direct:
        return str(direct)
    source_text = metric.get("source_campaign")
    if source_text is None:
        raise ValueError("metric JSON has no source_campaign")
    source = Path(source_text)
    if not source.exists():
        source = metric_path.resolve().parent / source.name
    payload = json.loads(source.read_text())
    lattice = payload.get("lattice")
    if not lattice:
        raise ValueError(f"source campaign {source} has no lattice")
    return str(lattice)


def _forbidden_with_weights(
    basis: np.ndarray,
    diameter: float,
    interval: float = 1.0,
    weight_power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        combigeo.forbidden_coords(
            basis.tolist(), diameter, float(interval)
        ),
        dtype=np.int64,
    )
    forbidden = canonical_rows(raw)
    facets = combigeo.relevant_facets(basis.tolist())
    ratios: list[float] = []
    for coordinate in forbidden:
        vector = coordinate @ basis
        distance = 2.0 * combigeo.dist_to_halfspaces(
            (0.5 * vector).tolist(), facets
        )
        ratios.append(float(distance / diameter))
    ratio_array = np.asarray(ratios, dtype=np.float64)
    weights = np.power(
        np.maximum(0.0, 1.0 - ratio_array), float(weight_power)
    )
    return forbidden, ratio_array, weights


def _cached_forbidden_with_weights(
    basis: np.ndarray,
    diameter: float,
    cache: Path | None,
    *,
    weight_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the expensive metric forbidden set across objective powers."""
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as payload:
            cached_basis = np.asarray(payload["basis"], dtype=np.float64)
            cached_diameter = float(payload["diameter"])
            forbidden = np.asarray(payload["forbidden"], dtype=np.int64)
            ratios = np.asarray(payload["ratios"], dtype=np.float64)
        if cached_basis.shape != basis.shape or not np.allclose(
            cached_basis, basis, rtol=2e-13, atol=2e-13
        ):
            raise ValueError(
                f"forbidden cache {cache} belongs to a different basis"
            )
        if not math.isclose(
            cached_diameter, diameter, rel_tol=2e-13, abs_tol=2e-13
        ):
            raise ValueError(
                f"forbidden cache {cache} has a different diameter"
            )
        if forbidden.ndim != 2 or len(forbidden) != len(ratios):
            raise ValueError(f"forbidden cache {cache} is malformed")
        weights = np.power(
            np.maximum(0.0, 1.0 - ratios), float(weight_power)
        )
        return forbidden, ratios, weights

    forbidden, ratios, weights = _forbidden_with_weights(
        basis, diameter, weight_power=weight_power
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            version=np.asarray([1], dtype=np.int64),
            basis=np.asarray(basis, dtype=np.float64),
            diameter=np.asarray(diameter, dtype=np.float64),
            forbidden=forbidden,
            ratios=ratios,
        )
    return forbidden, ratios, weights


def _score(
    dots: np.ndarray, weights: np.ndarray
) -> tuple[int, float]:
    mask = dots == 0
    return int(mask.sum()), float(weights[mask].sum())


def _objective_key(
    score: tuple[int, float], objective: str
) -> tuple[float | int, float | int]:
    count, weight = score
    if objective == "weighted":
        return weight, count
    return count, weight


def _canonical_prime_row(row: np.ndarray, prime: int) -> np.ndarray:
    """Choose the unique projective representative whose first nonzero is 1."""
    canonical = np.asarray(row, dtype=np.int64).copy() % prime
    nonzero = np.flatnonzero(canonical)
    if not len(nonzero):
        raise ValueError("zero row has no projective representative")
    scale = pow(int(canonical[int(nonzero[0])]), -1, prime)
    return canonical * scale % prime


def _pareto_dominates(first: dict, second: dict) -> bool:
    """Return whether ``first`` weakly improves all three archive objectives."""
    first_count = int(first["conflict_projective_pairs"])
    second_count = int(second["conflict_projective_pairs"])
    first_weight = float(first["conflict_weight"])
    second_weight = float(second["conflict_weight"])
    first_ratio = float(first["minimum_forbidden_ratio"])
    second_ratio = float(second["minimum_forbidden_ratio"])
    weight_tolerance = max(
        1e-12 * max(abs(first_weight), abs(second_weight)),
        16.0 * float(np.spacing(max(abs(first_weight), abs(second_weight)))),
    )
    ratio_tolerance = 1e-12
    weak = (
        first_count <= second_count
        and first_weight <= second_weight + weight_tolerance
        and first_ratio + ratio_tolerance >= second_ratio
    )
    strict = (
        first_count < second_count
        or first_weight + weight_tolerance < second_weight
        or first_ratio > second_ratio + ratio_tolerance
    )
    return weak and strict


def _update_pareto_archive(
    archive: list[dict],
    row: np.ndarray,
    prime: int,
    score: tuple[int, float],
    minimum_ratio: float,
    maximum_size: int,
) -> bool:
    """Insert a projectively unique nondominated lightweight row record."""
    if maximum_size <= 0 or score[0] <= 0 or not math.isfinite(minimum_ratio):
        return False
    canonical = _canonical_prime_row(row, prime)
    key = tuple(int(value) for value in canonical)
    if any(tuple(entry["row"]) == key for entry in archive):
        return False
    candidate = {
        "row": list(key),
        "conflict_projective_pairs": int(score[0]),
        "conflict_weight": float(score[1]),
        "minimum_forbidden_ratio": float(minimum_ratio),
    }
    if any(_pareto_dominates(entry, candidate) for entry in archive):
        return False
    archive[:] = [
        entry
        for entry in archive
        if not _pareto_dominates(candidate, entry)
    ]
    archive.append(candidate)
    if len(archive) > maximum_size:
        preserve: set[int] = set()
        counts = sorted(
            {int(entry["conflict_projective_pairs"]) for entry in archive}
        )
        for count in counts:
            indices = [
                index
                for index, entry in enumerate(archive)
                if int(entry["conflict_projective_pairs"]) == count
            ]
            preserve.add(
                min(indices, key=lambda index: archive[index]["conflict_weight"])
            )
            preserve.add(
                max(
                    indices,
                    key=lambda index: archive[index][
                        "minimum_forbidden_ratio"
                    ],
                )
            )
        preserve.add(
            max(
                range(len(archive)),
                key=lambda index: archive[index]["minimum_forbidden_ratio"],
            )
        )
        priority = sorted(
            range(len(archive)),
            key=lambda index: (
                int(archive[index]["conflict_projective_pairs"]),
                -float(archive[index]["minimum_forbidden_ratio"]),
                float(archive[index]["conflict_weight"]),
                tuple(archive[index]["row"]),
            ),
        )
        selected = list(sorted(preserve, key=priority.index))
        selected.extend(
            index
            for index in priority
            if index not in preserve
        )
        selected = selected[:maximum_size]
        archive[:] = [archive[index] for index in selected]
    archive.sort(
        key=lambda entry: (
            int(entry["conflict_projective_pairs"]),
            -float(entry["minimum_forbidden_ratio"]),
            float(entry["conflict_weight"]),
            tuple(entry["row"]),
        )
    )
    return any(tuple(entry["row"]) == key for entry in archive)


def _best_coordinate_values(
    forbidden: np.ndarray,
    weights: np.ndarray,
    row: np.ndarray,
    dots: np.ndarray,
    coordinate: int,
    prime: int,
    objective: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coefficients = forbidden[:, coordinate] % prime
    base = (dots - row[coordinate] * coefficients) % prime
    nonzero = coefficients != 0
    inverse = _INVERSE_CACHE.get(prime)
    if inverse is None:
        inverse = np.zeros(prime, dtype=np.int64)
        inverse[1:] = [pow(value, -1, prime) for value in range(1, prime)]
        _INVERSE_CACHE[prime] = inverse
    forbidden_values = (
        -base[nonzero] * inverse[coefficients[nonzero]]
    ) % prime
    fixed = (not np.all(nonzero)) and np.any((~nonzero) & (base == 0))
    fixed_count = int(np.count_nonzero((~nonzero) & (base == 0)))
    fixed_weight = float(weights[(~nonzero) & (base == 0)].sum())
    counts = np.bincount(forbidden_values, minlength=prime).astype(np.int64)
    costs = np.bincount(
        forbidden_values,
        weights=weights[nonzero],
        minlength=prime,
    ).astype(np.float64, copy=False)
    if fixed:
        counts += fixed_count
        costs += fixed_weight
    if objective == "weighted":
        minimum_cost = float(costs.min())
        candidates = np.flatnonzero(
            costs <= _cost_cutoff(minimum_cost)
        )
        minimum_count = int(counts[candidates].min())
        candidates = candidates[counts[candidates] == minimum_count]
    else:
        minimum_count = int(counts.min())
        candidates = np.flatnonzero(counts == minimum_count)
        minimum_cost = float(costs[candidates].min())
        candidates = candidates[
            costs[candidates] <= _cost_cutoff(minimum_cost)
        ]
    return candidates, counts, costs


def _best_pair_values(
    forbidden: np.ndarray,
    weights: np.ndarray,
    row: np.ndarray,
    dots: np.ndarray,
    first: int,
    second: int,
    prime: int,
    objective: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score all p^2 simultaneous values of two row coordinates.

    Every constraint contributes one affine line in F_p^2.  Accumulating
    those lines costs O(p|F|), versus O(p^2|F|) for direct rescoring.
    """
    first_coefficients = forbidden[:, first] % prime
    second_coefficients = forbidden[:, second] % prime
    base = (
        dots
        - row[first] * first_coefficients
        - row[second] * second_coefficients
    ) % prime
    inverse = _INVERSE_CACHE.get(prime)
    if inverse is None:
        inverse = np.zeros(prime, dtype=np.int64)
        inverse[1:] = [
            pow(value, -1, prime) for value in range(1, prime)
        ]
        _INVERSE_CACHE[prime] = inverse
    counts = np.zeros((prime, prime), dtype=np.int64)
    costs = np.zeros((prime, prime), dtype=np.float64)
    values = np.arange(prime, dtype=np.int64)
    fixed_count = 0
    fixed_weight = 0.0

    for first_coefficient, second_coefficient, constant, weight in zip(
        first_coefficients,
        second_coefficients,
        base,
        weights,
    ):
        first_coefficient = int(first_coefficient)
        second_coefficient = int(second_coefficient)
        constant = int(constant)
        weight = float(weight)
        if second_coefficient:
            second_values = (
                -constant - first_coefficient * values
            ) * inverse[second_coefficient] % prime
            counts[values, second_values] += 1
            costs[values, second_values] += weight
        elif first_coefficient:
            first_value = (
                -constant * int(inverse[first_coefficient])
            ) % prime
            counts[first_value, :] += 1
            costs[first_value, :] += weight
        elif constant == 0:
            fixed_count += 1
            fixed_weight += weight

    if fixed_count:
        counts += fixed_count
        costs += fixed_weight
    if objective == "weighted":
        minimum_cost = float(costs.min())
        mask = costs <= _cost_cutoff(minimum_cost)
        minimum_count = int(counts[mask].min())
        mask &= counts == minimum_count
    else:
        minimum_count = int(counts.min())
        mask = counts == minimum_count
        minimum_cost = float(costs[mask].min())
        mask &= costs <= _cost_cutoff(minimum_cost)
    candidates = np.argwhere(mask)
    return candidates, counts, costs


def _record(
    row: np.ndarray,
    prime: int,
    forbidden: np.ndarray,
    ratios: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    diameter: float,
    facets: list[tuple[list[float], float]],
) -> dict:
    dots = (forbidden @ row) % prime
    mask = dots == 0
    kernel = hnf_columns(kernel_basis([row], [prime], len(row)))
    separation = separate_kernel(basis, diameter, facets, kernel)
    return {
        "row": row.astype(int).tolist(),
        "modulus": prime,
        "conflict_projective_pairs": int(mask.sum()),
        "conflict_weight": float(weights[mask].sum()),
        "minimum_forbidden_ratio": (
            float(ratios[mask].min()) if np.any(mask) else None
        ),
        "conflicts": [
            {
                "coordinate": coordinate.astype(int).tolist(),
                "distance_ratio": float(ratio),
            }
            for coordinate, ratio in zip(forbidden[mask], ratios[mask])
        ],
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "complete_separation": {
            key: value
            for key, value in separation.items()
            if key != "conflict_coordinates"
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--prime", type=int, required=True)
    parser.add_argument("--restarts", type=int, default=20_000)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--stall-sweeps", type=int, default=2)
    parser.add_argument("--kick", type=int, default=2)
    parser.add_argument(
        "--pair-moves",
        type=int,
        default=0,
        help=(
            "number of exact all-p^2 pair-coordinate moves per sweep "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--pareto-size",
        type=int,
        default=0,
        help=(
            "maximum number of nondominated kernels retained across conflict "
            "count, total deficit and minimum ratio (default: disabled)"
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("lexicographic", "weighted"),
        default="lexicographic",
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
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not _is_prime(args.prime):
        parser.error("--prime must be prime")
    if args.restarts < 1 or args.sweeps < 1:
        parser.error("restarts and sweeps must be positive")
    if args.pair_moves < 0:
        parser.error("--pair-moves must be nonnegative")
    if args.pareto_size < 0:
        parser.error("--pareto-size must be nonnegative")
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
    prime = args.prime
    rng = np.random.default_rng(args.seed)
    seed_rows: list[np.ndarray] = []
    source_record = metric.get("source_record", {})
    if (
        source_record.get("rows")
        and source_record.get("moduli") == [prime]
    ):
        seed_rows.append(
            np.asarray(source_record["rows"][0], dtype=np.int64) % prime
        )

    payload: dict = {
        "method": "all-values coordinate descent for one prime quotient row",
        "source_metric": str(args.metric),
        "lattice": lattice,
        "n": n,
        "dimension": n,
        "prime": prime,
        "forbidden_projective_pairs": len(forbidden),
        "parent_diameter": diameter,
        "parent_facet_count": len(facets),
        "budget": {
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "stall_sweeps": args.stall_sweeps,
            "kick": args.kick,
            "pair_moves": args.pair_moves,
            "pareto_size": args.pareto_size,
            "seed": args.seed,
            "objective": args.objective,
            "weight_power": args.weight_power,
            "forbidden_cache": (
                str(args.forbidden_cache)
                if args.forbidden_cache is not None
                else None
            ),
        },
        "best": None,
        "best_by_minimum_ratio": None,
        "results": [],
        "valid_candidate": None,
    }
    start = time.perf_counter()
    best_score = (len(forbidden) + 1, float("inf"))
    best_row: np.ndarray | None = None
    best_minimum_ratio = float("-inf")
    objective_campaign_record: dict | None = None
    ratio_campaign_record: dict | None = None
    pareto_archive: list[dict] = []
    progress = max(100, args.restarts // 10)

    def as_campaign_record(record: dict) -> dict:
        return {
            "moduli": [prime],
            "rows": [record["row"]],
            "image_index": prime,
            "killed": record["conflict_projective_pairs"],
            "beta": float(args.weight_power),
            "minimum_conflict_ratio": record[
                "minimum_forbidden_ratio"
            ],
            "conflicts": record["conflicts"],
        }

    def consider(
        row: np.ndarray,
        dots: np.ndarray,
        restart: int,
    ) -> bool:
        nonlocal best_score, best_row, best_minimum_ratio
        nonlocal objective_campaign_record, ratio_campaign_record
        score = _score(dots, weights)
        objective_improves = _objective_key(
            score, args.objective
        ) < _objective_key(
            best_score, args.objective
        )
        mask = dots == 0
        minimum_ratio = (
            float(ratios[mask].min()) if np.any(mask) else float("inf")
        )
        pareto_changed = _update_pareto_archive(
            pareto_archive,
            row,
            prime,
            score,
            minimum_ratio,
            args.pareto_size,
        )
        if pareto_changed:
            payload["pareto_archive_summary"] = pareto_archive
        ratio_improves = minimum_ratio > best_minimum_ratio + 1e-12
        if not objective_improves and not ratio_improves:
            return False
        record = _record(
            row,
            prime,
            forbidden,
            ratios,
            weights,
            basis,
            diameter,
            facets,
        )
        campaign_record = as_campaign_record(record)
        if objective_improves:
            best_score = score
            best_row = row.copy()
            payload["best"] = record
            objective_campaign_record = campaign_record
        if ratio_improves:
            best_minimum_ratio = minimum_ratio
            payload["best_by_minimum_ratio"] = record
            ratio_campaign_record = campaign_record
        payload["results"] = []
        for candidate in (objective_campaign_record, ratio_campaign_record):
            if candidate is not None and candidate not in payload["results"]:
                payload["results"].append(candidate)
        payload["restarts_completed"] = restart + 1
        payload["elapsed_seconds"] = time.perf_counter() - start
        if score[0] == 0 and record["complete_separation"]["valid"]:
            payload["valid_candidate"] = record
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        labels = []
        if objective_improves:
            labels.append("objective")
        if ratio_improves:
            labels.append("minimum-ratio")
        print(
            f"  new best ({'+'.join(labels)}) restart={restart + 1}: "
            f"conflicts={score[0]} weight={score[1]:.6g} "
            f"min-ratio={record['minimum_forbidden_ratio']}",
            flush=True,
        )
        if score[0] == 0 and record["complete_separation"]["valid"]:
            print("*** VALID PRIME-INDEX KERNEL FOUND ***", flush=True)
            return True
        return False

    for restart in range(args.restarts):
        if restart < len(seed_rows):
            row = seed_rows[restart].copy()
        else:
            row = rng.integers(0, prime, size=n, dtype=np.int64)
            while not np.any(row):
                row = rng.integers(0, prime, size=n, dtype=np.int64)
        dots = (forbidden @ row) % prime
        local_best = _score(dots, weights)
        stalled = 0
        if consider(row, dots, restart):
            return 0

        for _ in range(args.sweeps):
            before = _score(dots, weights)
            for coordinate in rng.permutation(n):
                candidates, _, _ = _best_coordinate_values(
                    forbidden,
                    weights,
                    row,
                    dots,
                    int(coordinate),
                    prime,
                    args.objective,
                )
                value = int(rng.choice(candidates))
                delta = value - int(row[coordinate])
                if delta:
                    dots = (
                        dots + delta * forbidden[:, coordinate]
                    ) % prime
                    row[coordinate] = value
            for _ in range(args.pair_moves):
                first, second = sorted(
                    int(value)
                    for value in rng.choice(n, size=2, replace=False)
                )
                candidates, _, _ = _best_pair_values(
                    forbidden,
                    weights,
                    row,
                    dots,
                    first,
                    second,
                    prime,
                    args.objective,
                )
                first_value, second_value = candidates[
                    int(rng.integers(0, len(candidates)))
                ]
                first_delta = int(first_value) - int(row[first])
                second_delta = int(second_value) - int(row[second])
                if first_delta or second_delta:
                    dots = (
                        dots
                        + first_delta * forbidden[:, first]
                        + second_delta * forbidden[:, second]
                    ) % prime
                    row[first] = int(first_value)
                    row[second] = int(second_value)
            current = _score(dots, weights)
            if _objective_key(
                current, args.objective
            ) < _objective_key(local_best, args.objective):
                local_best = current
                stalled = 0
            elif _objective_key(
                current, args.objective
            ) >= _objective_key(before, args.objective):
                stalled += 1
            if consider(row, dots, restart):
                return 0
            if current[0] == 0:
                break
            if stalled >= args.stall_sweeps:
                kick = min(n, max(1, args.kick))
                for coordinate in rng.choice(n, size=kick, replace=False):
                    value = int(rng.integers(0, prime))
                    delta = value - int(row[coordinate])
                    if delta:
                        dots = (
                            dots + delta * forbidden[:, coordinate]
                        ) % prime
                        row[coordinate] = value
                stalled = 0

        if (restart + 1) % progress == 0:
            print(
                f"  progress {restart + 1}/{args.restarts}: "
                f"best-conflicts={best_score[0]} "
                f"best-weight={best_score[1]:.6g} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )

    if best_row is None:
        raise AssertionError("search evaluated no row")
    payload["restarts_completed"] = args.restarts
    payload["elapsed_seconds"] = time.perf_counter() - start
    if pareto_archive:
        full_archive: list[dict] = []
        for entry in pareto_archive:
            full_archive.append(
                _record(
                    np.asarray(entry["row"], dtype=np.int64),
                    prime,
                    forbidden,
                    ratios,
                    weights,
                    basis,
                    diameter,
                    facets,
                )
            )
        payload["pareto_archive"] = full_archive
        payload["pareto_archive_summary"] = pareto_archive
        for record in full_archive:
            candidate = as_campaign_record(record)
            if candidate not in payload["results"]:
                payload["results"].append(candidate)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL conflicts={best_score[0]} weight={best_score[1]:.9g} "
        f"best-min-ratio={best_minimum_ratio:.9g} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
