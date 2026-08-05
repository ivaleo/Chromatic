"""Arithmetic continuation from the 343-color E6* certificate to index 336.

The known E6* coloring has quotient

    (Z/7 Z)^3

and three exact characters stored in ``interval_fast_results.json``.  A
generic determinant repair from 343 to 342 or 340 destroys all three
characters.  The arithmetically favourable target

    336 = 7 * 48 = 7 * 16 * 3

can instead retain one character exactly.  Once a fixed row ``a`` modulo 7 is
chosen, only forbidden vectors satisfying ``a.f = 0 (mod 7)`` remain relevant.
The residual order-48 quotient is searched on that smaller exact constraint
set using complete primary-block best responses.

All five abelian 2-primary structures of order 16 are supported:

    [16], [8,2], [4,4], [4,2,2], [2,2,2,2].

An independent modulo-3 row is appended to each.  Count and geometry-weighted
objectives are retained because a kernel with several shallow conflicts may
be a better input to continuous metric deformation than a kernel with one
deep conflict.  Every saved incumbent is checked on the full forbidden set,
for exact image size 336, and with the complete geometric separation oracle.
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
from chromatic_research.core.lazy_prime_campaign import canonical_rows
from chromatic_research.core.prime_radon import (
    PrimarySearch,
    SearchResult,
    WeightedSearchResult,
    _prime_power,
    image_size,
    killed_mask,
    load_forbidden,
    rank_mod,
    weighted_close,
    weighted_improves,
)
from chromatic_research.core.prime_row_opt import _forbidden_with_weights
from chromatic_research.paths import results_path


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = results_path("interval_fast_results.json")
DEFAULT_STRUCTURES = [
    [16, 3],
    [8, 2, 3],
    [4, 4, 3],
    [4, 2, 2, 3],
    [2, 2, 2, 2, 3],
]


class CoordinatePrimarySearch:
    """Exact coefficient-neighbourhood search for large coprime blocks.

    Materializing all 2,064,384 projective rows modulo 16 is unnecessary for
    the residual ``[16,3]`` structure; the same issue occurs for prime blocks
    such as 47 in lower nonmonotonic targets.  For one coordinate and fixed
    remaining entries there are only ``q`` exact values.  This class
    alternates all such coefficient best responses for pairwise-coprime
    prime-power rows, while retaining only primitive rows.  It has the same
    ``run``/``run_weighted`` surface as :class:`PrimarySearch`, so final
    verification is shared.
    """

    def __init__(
        self,
        forbidden: Sequence[Sequence[int]] | np.ndarray,
        moduli: Sequence[int],
        *,
        seed: int = 0,
    ) -> None:
        self.forbidden = np.asarray(forbidden, dtype=np.int64)
        if self.forbidden.ndim != 2:
            raise ValueError("forbidden must be a two-dimensional array")
        self.n = self.forbidden.shape[1]
        self.moduli = [int(value) for value in moduli]
        if not self.moduli:
            raise ValueError("coordinate primary search needs a modulus")
        for modulus in self.moduli:
            _prime_power(modulus)
        if any(
            math.gcd(self.moduli[left], self.moduli[right]) != 1
            for left in range(len(self.moduli))
            for right in range(left)
        ):
            raise ValueError(
                "coordinate primary blocks must be pairwise coprime"
            )
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _primitive(row: np.ndarray, modulus: int) -> bool:
        common = int(modulus)
        for value in np.asarray(row, dtype=np.int64):
            common = math.gcd(common, int(value))
        return common == 1

    def random_rows(self) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for modulus in self.moduli:
            while True:
                row = self.rng.integers(
                    0, modulus, size=self.n, dtype=np.int64
                )
                if self._primitive(row, modulus):
                    rows.append(row)
                    break
        return rows

    def _score(
        self, rows: Sequence[np.ndarray], weights: np.ndarray | None
    ) -> float:
        mask = killed_mask(self.forbidden, rows, self.moduli)
        if weights is None:
            return float(mask.sum())
        return float(np.asarray(weights, dtype=np.float64)[mask].sum())

    def coordinate_candidates(
        self,
        rows: Sequence[np.ndarray],
        row_index: int,
        coordinate: int,
        *,
        top: int,
        weights: np.ndarray | None,
    ) -> list[tuple[float, np.ndarray]]:
        active = killed_mask(
            self.forbidden, rows, self.moduli, skip=row_index
        )
        vectors = self.forbidden[active]
        row = np.asarray(rows[row_index], dtype=np.int64)
        modulus = self.moduli[row_index]
        coefficients = vectors[:, coordinate]
        base = (vectors @ row - row[coordinate] * coefficients) % modulus
        values = np.arange(modulus, dtype=np.int64)
        zero = (
            base[:, None] + coefficients[:, None] * values[None, :]
        ) % modulus == 0
        if weights is None:
            scores = zero.sum(axis=0).astype(np.float64)
        else:
            scores = np.asarray(weights, dtype=np.float64)[active] @ zero
        order = np.argsort(scores, kind="stable")
        result: list[tuple[float, np.ndarray]] = []
        for value_index in order:
            candidate = row.copy()
            candidate[coordinate] = int(value_index)
            if not self._primitive(candidate, modulus):
                continue
            result.append((float(scores[value_index]), candidate))
            if len(result) >= top:
                break
        return result

    def descend(
        self,
        initial: Sequence[np.ndarray] | None = None,
        *,
        max_sweeps: int = 20,
        top: int = 8,
        kick_probability: float = 0.15,
        temperature: float = 0.35,
        weights: np.ndarray | None = None,
    ) -> tuple[float, list[np.ndarray], int]:
        rows = (
            [np.asarray(row, dtype=np.int64).copy() for row in initial]
            if initial is not None
            else self.random_rows()
        )
        if len(rows) != len(self.moduli) or any(
            row.shape != (self.n,) for row in rows
        ):
            raise ValueError("initial rows have incompatible shape")
        if any(
            not self._primitive(row, modulus)
            for row, modulus in zip(rows, self.moduli)
        ):
            raise ValueError("initial rows must be primitive")
        current = self._score(rows, weights)
        best = current
        best_rows = [row.copy() for row in rows]
        stale = 0
        sweeps_done = 0
        moves = [
            (row_index, coordinate)
            for row_index in range(len(rows))
            for coordinate in range(self.n)
        ]
        for sweep in range(max_sweeps):
            sweeps_done = sweep + 1
            before_sweep = current
            for move_index in self.rng.permutation(len(moves)):
                row_index, coordinate = moves[int(move_index)]
                candidates = self.coordinate_candidates(
                    rows,
                    row_index,
                    coordinate,
                    top=top,
                    weights=weights,
                )
                if not candidates:
                    continue
                choice = 0
                if (
                    len(candidates) > 1
                    and stale > 0
                    and self.rng.random() < kick_probability
                ):
                    ranks = np.arange(len(candidates), dtype=np.float64)
                    probabilities = np.exp(
                        -ranks / max(float(temperature), 1e-9)
                    )
                    probabilities /= probabilities.sum()
                    choice = int(
                        self.rng.choice(len(candidates), p=probabilities)
                    )
                score, candidate = candidates[choice]
                if score <= current or choice != 0:
                    rows[row_index] = candidate
                    current = score
                improves = (
                    current < best
                    if weights is None
                    else weighted_improves(current, best)
                )
                if improves:
                    best = current
                    best_rows = [row.copy() for row in rows]
                if best == 0.0:
                    return best, best_rows, sweeps_done
            sweep_improves = (
                current < before_sweep
                if weights is None
                else weighted_improves(current, before_sweep)
            )
            if sweep_improves:
                stale = 0
            else:
                stale += 1
            if stale >= 3:
                break
        return best, best_rows, sweeps_done

    def run(
        self,
        *,
        restarts: int,
        max_sweeps: int,
        top: int,
        progress_every: int,
        initial_rows: Sequence[Sequence[int]] | None = None,
    ) -> SearchResult:
        started = time.perf_counter()
        starts: list[Sequence[np.ndarray] | None] = []
        if initial_rows is not None:
            starts.append(
                [
                    np.asarray(row, dtype=np.int64)
                    for row in initial_rows
                ]
            )
        starts.extend([None] * restarts)
        best = len(self.forbidden) + 1.0
        best_rows: list[np.ndarray] | None = None
        total_sweeps = 0
        for restart, initial in enumerate(starts):
            score, rows, sweeps = self.descend(
                initial, max_sweeps=max_sweeps, top=top
            )
            total_sweeps += sweeps
            if score < best:
                best = score
                best_rows = [row.copy() for row in rows]
                print(
                    f"  cyclic-coordinate best={int(best)} "
                    f"restart={restart} sweeps={sweeps}",
                    flush=True,
                )
            if best <= 0:
                break
            if progress_every and (restart + 1) % progress_every == 0:
                print(
                    f"  cyclic-coordinate progress "
                    f"{restart + 1}/{len(starts)} best={int(best)}",
                    flush=True,
                )
        if best_rows is None:
            raise RuntimeError("coordinate search received no starts")
        exact = int(
            killed_mask(
                self.forbidden, best_rows, self.moduli
            ).sum()
        )
        if exact != int(best):
            raise AssertionError(
                f"coordinate score mismatch: search={best}, exact={exact}"
            )
        return SearchResult(
            killed=exact,
            rows=best_rows,
            moduli=self.moduli.copy(),
            image_index=image_size(best_rows, self.moduli, self.n),
            restarts=len(starts),
            sweeps=total_sweeps,
            seconds=time.perf_counter() - started,
        )

    def run_weighted(
        self,
        weights: Sequence[float] | np.ndarray,
        *,
        restarts: int,
        max_sweeps: int,
        top: int,
        progress_every: int,
        initial_rows: Sequence[Sequence[int]] | None = None,
    ) -> WeightedSearchResult:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != (len(self.forbidden),):
            raise ValueError("weights have incompatible shape")
        if not np.all(np.isfinite(weights_array)) or np.any(
            weights_array < 0
        ):
            raise ValueError("weights must be finite and nonnegative")
        started = time.perf_counter()
        starts: list[Sequence[np.ndarray] | None] = []
        if initial_rows is not None:
            starts.append(
                [
                    np.asarray(row, dtype=np.int64)
                    for row in initial_rows
                ]
            )
        starts.extend([None] * restarts)
        best = float("inf")
        best_rows: list[np.ndarray] | None = None
        total_sweeps = 0
        for restart, initial in enumerate(starts):
            score, rows, sweeps = self.descend(
                initial,
                max_sweeps=max_sweeps,
                top=top,
                weights=weights_array,
            )
            total_sweeps += sweeps
            if weighted_improves(score, best):
                best = score
                best_rows = [row.copy() for row in rows]
                killed = int(
                    killed_mask(
                        self.forbidden, best_rows, self.moduli
                    ).sum()
                )
                print(
                    f"  cyclic-weighted best={best:.9g} "
                    f"killed={killed} restart={restart}",
                    flush=True,
                )
            if best == 0.0:
                break
            if progress_every and (restart + 1) % progress_every == 0:
                print(
                    f"  cyclic-weighted progress "
                    f"{restart + 1}/{len(starts)} best={best:.9g}",
                    flush=True,
                )
        if best_rows is None:
            raise RuntimeError("coordinate search received no starts")
        final_mask = killed_mask(
            self.forbidden, best_rows, self.moduli
        )
        exact = float(weights_array[final_mask].sum())
        if not weighted_close(exact, best):
            raise AssertionError(
                f"weighted coordinate mismatch: search={best}, exact={exact}"
            )
        return WeightedSearchResult(
            weighted_loss=exact,
            killed=int(final_mask.sum()),
            rows=best_rows,
            moduli=self.moduli.copy(),
            image_index=image_size(best_rows, self.moduli, self.n),
            restarts=len(starts),
            sweeps=total_sweeps,
            seconds=time.perf_counter() - started,
        )


def parse_structures(text: str) -> list[list[int]]:
    """Parse and validate residual primary structures of order 48."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("structures must be a non-empty list")
    structures: list[list[int]] = []
    for raw_structure in raw:
        if not isinstance(raw_structure, list) or not raw_structure:
            raise argparse.ArgumentTypeError(
                "each structure must be a non-empty list"
            )
        structure = [int(value) for value in raw_structure]
        if math.prod(structure) != 48:
            raise argparse.ArgumentTypeError(
                f"residual structure {structure} has product "
                f"{math.prod(structure)}, expected 48"
            )
        try:
            prime_powers = [_prime_power(value) for value in structure]
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        prime_products: dict[int, int] = {}
        for value, (prime, _) in zip(structure, prime_powers):
            prime_products[prime] = prime_products.get(prime, 1) * value
        if prime_products != {2: 16, 3: 3}:
            raise argparse.ArgumentTypeError(
                f"structure {structure} must split 48 into a 2-primary "
                "part of order 16 and a modulo-3 block"
            )
        structures.append(structure)
    return structures


def parse_powers(text: str) -> list[float]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list):
        raise argparse.ArgumentTypeError("weight powers must be a list")
    powers = [float(value) for value in raw]
    if any(not math.isfinite(value) or value <= 0 for value in powers):
        raise argparse.ArgumentTypeError(
            "weight powers must be finite and positive"
        )
    return powers


def load_e6_source_rows(path: Path) -> list[np.ndarray]:
    """Load the three exact modulo-7 rows of the index-343 certificate."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list")
    matches = [
        entry
        for entry in payload
        if (
            isinstance(entry, dict)
            and entry.get("name") == "E6*"
            and int(entry.get("k", -1)) == 343
            and entry.get("e_list") == [7, 7, 7]
            and isinstance(entry.get("phi"), list)
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one E6* index-343 character record in {path}, "
            f"found {len(matches)}"
        )
    rows = [np.asarray(row, dtype=np.int64) % 7 for row in matches[0]["phi"]]
    if len(rows) != 3 or any(row.shape != (6,) for row in rows):
        raise ValueError("the E6* certificate must contain three length-6 rows")
    if rank_mod(np.asarray(rows), 7) != 3:
        raise ValueError("the E6* modulo-7 rows are not independent")
    if image_size(rows, [7, 7, 7], 6) != 343:
        raise ValueError("the E6* rows do not have image size 343")
    return rows


def residual_mask(forbidden: np.ndarray, fixed_row: np.ndarray) -> np.ndarray:
    """Constraints not already separated by the retained modulo-7 row."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    fixed_row = np.asarray(fixed_row, dtype=np.int64)
    if forbidden.ndim != 2 or fixed_row.shape != (forbidden.shape[1],):
        raise ValueError("incompatible forbidden array and fixed row")
    return (forbidden @ fixed_row) % 7 == 0


def full_rows(
    fixed_row: np.ndarray, residual_rows: Sequence[np.ndarray]
) -> list[np.ndarray]:
    return [
        np.asarray(fixed_row, dtype=np.int64).copy(),
        *[
            np.asarray(row, dtype=np.int64).copy()
            for row in residual_rows
        ],
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="JSON containing the E6* index-343 character certificate",
    )
    parser.add_argument(
        "--structures",
        type=parse_structures,
        default=DEFAULT_STRUCTURES,
        help="JSON list of residual order-48 primary structures",
    )
    parser.add_argument("--count-restarts", type=int, default=40)
    parser.add_argument("--weighted-restarts", type=int, default=40)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--weight-powers",
        type=parse_powers,
        default=[1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument("--pair-top", type=int, default=0)
    parser.add_argument("--weighted-pair-top", type=int, default=0)
    parser.add_argument("--seed", type=int, default=6336001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count_restarts < 1 or args.weighted_restarts < 0:
        parser.error(
            "count restarts must be positive and weighted restarts nonnegative"
        )
    if (
        args.sweeps < 1
        or args.top < 1
        or args.pair_top < 0
        or args.weighted_pair_top < 0
    ):
        parser.error(
            "sweeps/top must be positive and pair budgets nonnegative"
        )

    basis, catalog_forbidden, diameter = load_forbidden("E6*")
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    # Both loaders must describe the same canonical projective +/- constraint
    # set, even if their enumeration order changes.
    weighted_set = {tuple(int(value) for value in row) for row in forbidden}
    catalog_set = {
        tuple(int(value) for value in row)
        for row in canonical_rows(catalog_forbidden)
    }
    if weighted_set != catalog_set:
        raise RuntimeError(
            "the weighted and catalog E6* forbidden sets differ: "
            f"weighted={len(weighted_set)}, catalog={len(catalog_set)}"
        )
    source_rows = load_e6_source_rows(args.source)
    if np.any(killed_mask(forbidden, source_rows, [7, 7, 7])):
        raise RuntimeError("the source index-343 characters fail exact separation")

    facets = combigeo.relevant_facets(basis.tolist())
    deficits = np.maximum(0.0, 1.0 - ratios)
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "fixed modulo-7 arithmetic continuation from index 343, "
            "followed by complete primary-block search of residual order 48"
        ),
        "lattice": "E6*",
        "n": 6,
        "dimension": 6,
        "source_certificate": str(args.source),
        "source_index": 343,
        "source_moduli": [7, 7, 7],
        "source_rows": [row.astype(int).tolist() for row in source_rows],
        "target_index": 336,
        "target_factorization": [7, 16, 3],
        "parent_diameter": float(diameter),
        "parent_facet_count": len(facets),
        "forbidden_projective_pairs": len(forbidden),
        "structures": args.structures,
        "budget": {
            "count_restarts": args.count_restarts,
            "weighted_restarts": args.weighted_restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "weight_powers": args.weight_powers,
            "pair_top": args.pair_top,
            "weighted_pair_top": args.weighted_pair_top,
            "seed": args.seed,
        },
        "fixed_rows": [],
        "results": [],
        "valid_candidate": None,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    def add_record(
        *,
        label: str,
        beta: float,
        fixed_index: int,
        structure: list[int],
        rows: Sequence[np.ndarray],
        weights: np.ndarray,
        seconds: float,
        metadata: dict,
    ) -> dict:
        moduli = [7, *structure]
        all_rows = full_rows(source_rows[fixed_index], rows)
        record = candidate_record(
            label=label,
            beta=beta,
            rows=all_rows,
            moduli=moduli,
            forbidden=forbidden,
            ratios=ratios,
            weights=weights,
            basis=basis,
            diameter=diameter,
            facets=facets,
            search_seconds=seconds,
            search_metadata={
                "fixed_row_index": fixed_index,
                "fixed_row": source_rows[fixed_index].astype(int).tolist(),
                "residual_moduli": structure,
                **metadata,
            },
        )
        record["fixed_row_index"] = fixed_index
        record["residual_moduli"] = structure
        payload["results"].append(record)
        if record.get("complete_separation", {}).get("valid"):
            payload["valid_candidate"] = record
        save()
        return record

    for fixed_index, fixed_row in enumerate(source_rows):
        mask = residual_mask(forbidden, fixed_row)
        residual_forbidden = forbidden[mask]
        payload["fixed_rows"].append(
            {
                "index": fixed_index,
                "row": fixed_row.astype(int).tolist(),
                "residual_projective_pairs": int(mask.sum()),
                "removed_projective_pairs": int((~mask).sum()),
            }
        )
        save()
        print(
            f"\nfixed={fixed_index} row={fixed_row.tolist()} "
            f"residual={len(residual_forbidden)}/{len(forbidden)}",
            flush=True,
        )
        for structure_index, structure in enumerate(args.structures):
            print(
                f"  structure={structure} full=[7,{','.join(map(str, structure))}] "
                "target=336",
                flush=True,
            )
            local_seed = (
                args.seed
                + 100_003 * fixed_index
                + 1_009 * structure_index
            )
            search_type = (
                CoordinatePrimarySearch
                if structure == [16, 3]
                else PrimarySearch
            )
            count_search = search_type(
                residual_forbidden, structure, seed=local_seed
            )
            count_started = time.perf_counter()
            count = count_search.run(
                restarts=args.count_restarts,
                max_sweeps=args.sweeps,
                top=args.top,
                progress_every=max(1, args.count_restarts // 4),
            )
            count_record = add_record(
                label=f"fixed-{fixed_index}-count",
                beta=0.0,
                fixed_index=fixed_index,
                structure=structure,
                rows=count.rows,
                weights=np.ones(len(forbidden), dtype=np.float64),
                seconds=time.perf_counter() - count_started,
                metadata=count.as_json(),
            )
            print(
                f"    count killed={count_record['killed']} "
                f"min-ratio={count_record['minimum_conflict_ratio']}",
                flush=True,
            )
            if payload["valid_candidate"] is not None:
                print("*** VALID INDEX-336 KERNEL FOUND ***", flush=True)
                return 0

            if (
                args.pair_top
                and len(structure) >= 2
                and isinstance(count_search, PrimarySearch)
            ):
                pair_started = time.perf_counter()
                pair_killed, pair_rows = count_search.pair_polish(
                    count.rows, first_top=args.pair_top
                )
                pair_record = add_record(
                    label=f"fixed-{fixed_index}-count-pair",
                    beta=-1.0,
                    fixed_index=fixed_index,
                    structure=structure,
                    rows=pair_rows,
                    weights=np.ones(len(forbidden), dtype=np.float64),
                    seconds=time.perf_counter() - pair_started,
                    metadata={
                        "pair_top": args.pair_top,
                        "starting_killed": count.killed,
                        "final_killed": pair_killed,
                    },
                )
                print(
                    f"    pair killed={pair_record['killed']} "
                    f"min-ratio={pair_record['minimum_conflict_ratio']}",
                    flush=True,
                )
                if payload["valid_candidate"] is not None:
                    print("*** VALID INDEX-336 KERNEL FOUND ***", flush=True)
                    return 0

            for power_index, power in enumerate(args.weight_powers):
                weights = np.power(deficits, power)
                weighted_search = search_type(
                    residual_forbidden,
                    structure,
                    seed=local_seed + 104_729 * (power_index + 1),
                )
                weighted_started = time.perf_counter()
                weighted = weighted_search.run_weighted(
                    weights[mask],
                    restarts=args.weighted_restarts,
                    max_sweeps=args.sweeps,
                    top=args.top,
                    progress_every=max(
                        1, max(1, args.weighted_restarts) // 4
                    ),
                    initial_rows=count.rows,
                )
                weighted_record = add_record(
                    label=f"fixed-{fixed_index}-deficit-{power:g}",
                    beta=float(power),
                    fixed_index=fixed_index,
                    structure=structure,
                    rows=weighted.rows,
                    weights=weights,
                    seconds=time.perf_counter() - weighted_started,
                    metadata=weighted.as_json(),
                )
                print(
                    f"    power={power:g} killed={weighted_record['killed']} "
                    f"loss={weighted_record['weighted_loss']:.9g} "
                    f"min-ratio={weighted_record['minimum_conflict_ratio']}",
                    flush=True,
                )
                if payload["valid_candidate"] is not None:
                    print("*** VALID INDEX-336 KERNEL FOUND ***", flush=True)
                    return 0

                if (
                    args.weighted_pair_top
                    and len(structure) >= 2
                    and isinstance(weighted_search, PrimarySearch)
                ):
                    pair_started = time.perf_counter()
                    pair_loss, pair_rows = (
                        weighted_search.pair_polish_weighted(
                            weighted.rows,
                            weights[mask],
                            first_top=args.weighted_pair_top,
                        )
                    )
                    pair_record = add_record(
                        label=(
                            f"fixed-{fixed_index}-deficit-{power:g}-pair"
                        ),
                        beta=float(power),
                        fixed_index=fixed_index,
                        structure=structure,
                        rows=pair_rows,
                        weights=weights,
                        seconds=time.perf_counter() - pair_started,
                        metadata={
                            "weighted_pair_top": args.weighted_pair_top,
                            "starting_loss": weighted.weighted_loss,
                            "final_loss": pair_loss,
                        },
                    )
                    print(
                        f"    weighted-pair power={power:g} "
                        f"killed={pair_record['killed']} "
                        f"loss={pair_record['weighted_loss']:.9g} "
                        f"min-ratio={pair_record['minimum_conflict_ratio']}",
                        flush=True,
                    )
                    if payload["valid_candidate"] is not None:
                        print("*** VALID INDEX-336 KERNEL FOUND ***", flush=True)
                        return 0

    save()
    best = max(
        payload["results"],
        key=lambda record: float(
            record["complete_separation"]["minimum_distance_ratio"]
        ),
    )
    print(
        "\nFINAL "
        f"best-fixed={best['fixed_row_index']} "
        f"moduli={best['moduli']} killed={best['killed']} "
        "complete-ratio="
        f"{best['complete_separation']['minimum_distance_ratio']:.12g} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
