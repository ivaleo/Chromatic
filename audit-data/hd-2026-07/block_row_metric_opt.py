"""Block-character search on an arbitrary deformed lattice metric.

For a quotient

    G = Z/q_1 Z x ... x Z/q_r Z

the coloring homomorphism is represented by one primitive row for each cyclic
prime-power block.  A forbidden vector is a conflict only when every row kills
it.  Updating one whole row is therefore an exact best response over the full
projective row pool for that block.

This script extends ``prime_row_opt.py`` in two directions that matter directly
below 139 colors:

* composite orders are split into CRT/primary blocks;
* all abelian quotient types of a fixed order can be compared, e.g.
  136 = 17*8 via [17,8], [17,4,2], or [17,2,2,2].

Count and geometry-weighted searches are both retained.  Every incumbent is
checked by exact modular arithmetic and the complete geometric separation
oracle before being written.
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
from lazy_prime_campaign import separate_kernel
from prime_radon import (
    PrimarySearch,
    hnf_columns,
    image_size,
    kernel_basis,
    killed_mask,
    smith_diagonal,
)
from prime_row_opt import _forbidden_with_weights, _source_lattice


def parse_structures(text: str) -> list[list[int]]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("structures must be a non-empty list")
    structures: list[list[int]] = []
    for structure in raw:
        if not isinstance(structure, list) or not structure:
            raise argparse.ArgumentTypeError(
                "each quotient structure must be a non-empty list"
            )
        values = [int(value) for value in structure]
        if any(value < 2 for value in values):
            raise argparse.ArgumentTypeError("all moduli must be at least two")
        structures.append(values)
    return structures


def parse_powers(text: str) -> list[float]:
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise argparse.ArgumentTypeError("weight powers must be a list")
    powers = [float(value) for value in raw]
    if any(not math.isfinite(value) or value <= 0 for value in powers):
        raise argparse.ArgumentTypeError(
            "weight powers must be finite and positive"
        )
    return powers


def metric_source_rows(
    metric: dict,
    moduli: Sequence[int],
    n: int,
) -> list[np.ndarray] | None:
    """Return an exact full-image warm start embedded in a metric checkpoint."""
    record = metric.get("source_record")
    if not isinstance(record, dict) or record.get("moduli") != list(moduli):
        return None
    raw_rows = record.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(moduli):
        return None
    rows = [np.asarray(row, dtype=np.int64) for row in raw_rows]
    if any(row.shape != (n,) for row in rows):
        return None
    if image_size(rows, moduli, n) != math.prod(moduli):
        return None
    return rows


def candidate_record(
    *,
    label: str,
    beta: float,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
    forbidden: np.ndarray,
    ratios: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    diameter: float,
    facets: list[tuple[list[float], float]],
    search_seconds: float,
    search_metadata: dict,
) -> dict:
    rows = [np.asarray(row, dtype=np.int64) for row in rows]
    mask = killed_mask(forbidden, rows, moduli)
    exact_index = image_size(rows, moduli, basis.shape[0])
    target = math.prod(moduli)
    record = {
        "label": label,
        "beta": float(beta),
        "moduli": [int(value) for value in moduli],
        "rows": [row.astype(int).tolist() for row in rows],
        "target_product": target,
        "image_index": exact_index,
        "killed": int(mask.sum()),
        "weighted_loss": float(weights[mask].sum()),
        "minimum_conflict_ratio": (
            float(ratios[mask].min()) if np.any(mask) else None
        ),
        "conflicts": [
            {
                "coordinate": coordinate.astype(int).tolist(),
                "distance_ratio": float(ratio),
                "weight": float(weight),
            }
            for coordinate, ratio, weight in zip(
                forbidden[mask], ratios[mask], weights[mask]
            )
        ],
        "search_seconds": float(search_seconds),
        "search": search_metadata,
    }
    if exact_index != target:
        record["kernel_error"] = (
            f"image index {exact_index} differs from target product {target}"
        )
        return record
    kernel = hnf_columns(kernel_basis(rows, moduli, basis.shape[0]))
    separation = separate_kernel(
        basis, diameter, facets, kernel
    )
    record.update(
        {
            "kernel_basis_columns": kernel.astype(int).tolist(),
            "kernel_determinant": abs(
                int(round(np.linalg.det(kernel)))
            ),
            "kernel_smith": smith_diagonal(kernel),
            "complete_separation": {
                key: value
                for key, value in separation.items()
                if key != "conflict_coordinates"
            },
        }
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("structures", type=parse_structures)
    parser.add_argument("--count-restarts", type=int, default=20)
    parser.add_argument("--weighted-restarts", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=16)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--weight-powers",
        type=parse_powers,
        default=[1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--pair-top",
        type=int,
        default=0,
        help=(
            "if positive, retain this many first-row candidates in an "
            "unweighted exact two-block polish"
        ),
    )
    parser.add_argument(
        "--weighted-pair-top",
        type=int,
        default=0,
        help=(
            "if positive, retain this many first-row candidates in a "
            "geometry-weighted two-block look-ahead"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--ignore-source-seed",
        action="store_true",
        help=(
            "do not warm-start from metric.source_record even when its "
            "quotient structure matches"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count_restarts < 0 or args.weighted_restarts < 0:
        parser.error("restart counts must be nonnegative")
    if (
        args.sweeps < 1
        or args.top < 1
        or args.pair_top < 0
        or args.weighted_pair_top < 0
    ):
        parser.error(
            "sweeps/top must be positive and pair budgets nonnegative"
        )

    metric = json.loads(args.metric.read_text())
    basis = np.asarray(metric["best"]["basis"], dtype=np.float64)
    diameter = float(metric["best"]["diameter"])
    lattice = _source_lattice(args.metric, metric)
    facets = combigeo.relevant_facets(basis.tolist())
    forbidden, ratios, squared_deficits = _forbidden_with_weights(
        basis, diameter
    )
    deficits = np.maximum(0.0, 1.0 - ratios)
    started = time.perf_counter()
    payload = {
        "method": (
            "complete primary-block best responses on a deformed metric"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "n": int(basis.shape[0]),
        "dimension": int(basis.shape[0]),
        "parent_diameter": diameter,
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
            "ignore_source_seed": args.ignore_source_seed,
        },
        "results": [],
        "valid_candidate": None,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for structure_number, moduli in enumerate(args.structures):
        print(
            f"\nstructure={moduli} target={math.prod(moduli)}",
            flush=True,
        )
        search = PrimarySearch(
            forbidden,
            moduli,
            seed=args.seed + 1009 * structure_number,
        )
        source_rows = (
            None
            if args.ignore_source_seed
            else metric_source_rows(metric, moduli, basis.shape[0])
        )
        if source_rows is not None:
            source_record = candidate_record(
                label="metric-source-control",
                beta=-2.0,
                rows=source_rows,
                moduli=moduli,
                forbidden=forbidden,
                ratios=ratios,
                weights=np.ones(len(forbidden), dtype=np.float64),
                basis=basis,
                diameter=diameter,
                facets=facets,
                search_seconds=0.0,
                search_metadata={
                    "source": "metric.source_record",
                    "optimization": "none",
                },
            )
            payload["results"].append(source_record)
            save()
            print(
                f"  source control killed={source_record['killed']} "
                f"min-ratio={source_record['minimum_conflict_ratio']}",
                flush=True,
            )
            if source_record.get("complete_separation", {}).get("valid"):
                payload["valid_candidate"] = source_record
                save()
                print("*** VALID SOURCE KERNEL FOUND ***", flush=True)
                return 0
        count_started = time.perf_counter()
        count = search.run(
            restarts=args.count_restarts,
            max_sweeps=args.sweeps,
            top=args.top,
            progress_every=max(1, (args.count_restarts + 1) // 4),
            initial_rows=source_rows,
        )
        count_record = candidate_record(
            label="count",
            beta=0.0,
            rows=count.rows,
            moduli=moduli,
            forbidden=forbidden,
            ratios=ratios,
            weights=np.ones(len(forbidden), dtype=np.float64),
            basis=basis,
            diameter=diameter,
            facets=facets,
            search_seconds=time.perf_counter() - count_started,
            search_metadata=count.as_json(),
        )
        payload["results"].append(count_record)
        save()
        print(
            f"  count result killed={count_record['killed']} "
            f"min-ratio={count_record['minimum_conflict_ratio']}",
            flush=True,
        )
        if count_record.get("complete_separation", {}).get("valid"):
            payload["valid_candidate"] = count_record
            save()
            print("*** VALID BLOCK KERNEL FOUND ***", flush=True)
            return 0

        if args.pair_top and len(moduli) >= 2:
            pair_started = time.perf_counter()
            pair_killed, pair_rows = search.pair_polish(
                count.rows, first_top=args.pair_top
            )
            pair_record = candidate_record(
                label="count-pair-polish",
                beta=-1.0,
                rows=pair_rows,
                moduli=moduli,
                forbidden=forbidden,
                ratios=ratios,
                weights=np.ones(len(forbidden), dtype=np.float64),
                basis=basis,
                diameter=diameter,
                facets=facets,
                search_seconds=time.perf_counter() - pair_started,
                search_metadata={
                    "pair_top": args.pair_top,
                    "starting_killed": count.killed,
                    "final_killed": pair_killed,
                },
            )
            payload["results"].append(pair_record)
            save()
            print(
                f"  pair result killed={pair_record['killed']} "
                f"min-ratio={pair_record['minimum_conflict_ratio']}",
                flush=True,
            )
            if pair_record.get("complete_separation", {}).get("valid"):
                payload["valid_candidate"] = pair_record
                save()
                print("*** VALID BLOCK KERNEL FOUND ***", flush=True)
                return 0

        for power_number, power in enumerate(args.weight_powers):
            weights = np.power(deficits, power)
            weighted_started = time.perf_counter()
            weighted = PrimarySearch(
                forbidden,
                moduli,
                seed=(
                    args.seed
                    + 1009 * structure_number
                    + 104729 * (power_number + 1)
                ),
            ).run_weighted(
                weights,
                restarts=args.weighted_restarts,
                max_sweeps=args.sweeps,
                top=args.top,
                progress_every=max(
                    1, (args.weighted_restarts + 1) // 4
                ),
                initial_rows=(
                    source_rows if source_rows is not None else count.rows
                ),
            )
            weighted_record = candidate_record(
                label=f"deficit-power-{power:g}",
                beta=float(power),
                rows=weighted.rows,
                moduli=moduli,
                forbidden=forbidden,
                ratios=ratios,
                weights=weights,
                basis=basis,
                diameter=diameter,
                facets=facets,
                search_seconds=time.perf_counter() - weighted_started,
                search_metadata=weighted.as_json(),
            )
            payload["results"].append(weighted_record)
            save()
            print(
                f"  power={power:g} killed={weighted_record['killed']} "
                f"loss={weighted_record['weighted_loss']:.9g} "
                f"min-ratio={weighted_record['minimum_conflict_ratio']}",
                flush=True,
            )
            if weighted_record.get(
                "complete_separation", {}
            ).get("valid"):
                payload["valid_candidate"] = weighted_record
                save()
                print("*** VALID BLOCK KERNEL FOUND ***", flush=True)
                return 0

            if args.weighted_pair_top and len(moduli) >= 2:
                pair_started = time.perf_counter()
                pair_loss, pair_rows = search.pair_polish_weighted(
                    weighted.rows,
                    weights,
                    first_top=args.weighted_pair_top,
                )
                weighted_pair_record = candidate_record(
                    label=f"deficit-power-{power:g}-pair-polish",
                    beta=float(power),
                    rows=pair_rows,
                    moduli=moduli,
                    forbidden=forbidden,
                    ratios=ratios,
                    weights=weights,
                    basis=basis,
                    diameter=diameter,
                    facets=facets,
                    search_seconds=time.perf_counter() - pair_started,
                    search_metadata={
                        "weighted_pair_top": args.weighted_pair_top,
                        "starting_loss": weighted.weighted_loss,
                        "final_loss": pair_loss,
                    },
                )
                payload["results"].append(weighted_pair_record)
                save()
                print(
                    f"  weighted pair power={power:g} "
                    f"killed={weighted_pair_record['killed']} "
                    f"loss={weighted_pair_record['weighted_loss']:.9g} "
                    "min-ratio="
                    f"{weighted_pair_record['minimum_conflict_ratio']}",
                    flush=True,
                )
                if weighted_pair_record.get(
                    "complete_separation", {}
                ).get("valid"):
                    payload["valid_candidate"] = weighted_pair_record
                    save()
                    print("*** VALID BLOCK KERNEL FOUND ***", flush=True)
                    return 0

    save()
    best = min(
        payload["results"],
        key=lambda record: (
            record["killed"],
            -float(record["minimum_conflict_ratio"] or -1.0),
        ),
    )
    print(
        f"\nFINAL best structure={best['moduli']} "
        f"killed={best['killed']} "
        f"min-ratio={best['minimum_conflict_ratio']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
