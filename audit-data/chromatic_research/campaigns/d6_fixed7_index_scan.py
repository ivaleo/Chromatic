"""Nonmonotonic E6* index scan retaining one source modulo-7 character.

After the structured target 336=7*48, scan arithmetically different lower
residual orders rather than assuming that the best separation is monotone in
the number of colors.  The default targets are

    329=7*47, 322=7*23*2, 315=7*9*5,
    308=7*11*4, 301=7*43, 280=7*8*5.

Small projective pools use complete primary-block best responses.  Large prime
rows use exact all-values coefficient responses, avoiding materialization of
hundreds of millions of projective points.  Every incumbent is checked on the
full forbidden set, for its exact image size, and by the complete geometric
separation oracle.
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
from chromatic_research.campaigns.d6_fixed7_campaign import (
    CoordinatePrimarySearch,
    DEFAULT_SOURCE,
    full_rows,
    load_e6_source_rows,
    parse_powers,
    residual_mask,
)
from chromatic_research.core.prime_radon import (
    PrimarySearch,
    _prime_power,
    load_forbidden,
)
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


HERE = Path(__file__).resolve().parent
DEFAULT_TARGETS = [[47], [23, 2], [9, 5], [11, 4], [43], [8, 5]]


def parse_targets(text: str) -> list[list[int]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("targets must be a non-empty list")
    targets: list[list[int]] = []
    for structure_raw in raw:
        if not isinstance(structure_raw, list) or not structure_raw:
            raise argparse.ArgumentTypeError(
                "every target must be a non-empty modulus list"
            )
        structure = [int(value) for value in structure_raw]
        try:
            [_prime_power(value) for value in structure]
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        if math.prod(structure) >= 49:
            raise argparse.ArgumentTypeError(
                f"residual order {math.prod(structure)} must be below 49"
            )
        targets.append(structure)
    return targets


def projective_pool_size(n: int, modulus: int) -> int:
    prime, _ = _prime_power(modulus)
    smaller = modulus // prime
    return sum(
        smaller**pivot * modulus ** (n - pivot - 1)
        for pivot in range(n)
    )


def coordinate_search_needed(
    n: int, structure: Sequence[int], limit: int = 250_000
) -> bool:
    return any(
        projective_pool_size(n, modulus) > limit
        for modulus in structure
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--targets", type=parse_targets, default=DEFAULT_TARGETS
    )
    parser.add_argument("--restarts", type=int, default=80)
    parser.add_argument("--coordinate-restarts", type=int, default=800)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument(
        "--weight-powers",
        type=parse_powers,
        default=[4.0, 8.0, 16.0],
    )
    parser.add_argument("--seed", type=int, default=6337501)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.restarts < 1 or args.coordinate_restarts < 1:
        parser.error("restart budgets must be positive")
    if args.sweeps < 1 or args.top < 1:
        parser.error("sweeps and top must be positive")

    basis, _, diameter = load_forbidden("E6*")
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    deficits = np.maximum(0.0, 1.0 - ratios)
    source_rows = load_e6_source_rows(args.source)
    facets = combigeo.relevant_facets(basis.tolist())
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "one-fixed-F7 nonmonotonic residual-index scan with complete "
            "small-block or exact coefficient best responses"
        ),
        "lattice": "E6*",
        "n": 6,
        "source_certificate": str(args.source),
        "source_rows": [row.astype(int).tolist() for row in source_rows],
        "targets": args.targets,
        "target_indices": [
            7 * math.prod(structure) for structure in args.targets
        ],
        "budget": {
            "restarts": args.restarts,
            "coordinate_restarts": args.coordinate_restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "weight_powers": args.weight_powers,
            "seed": args.seed,
        },
        "results": [],
        "valid_candidate": None,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for target_number, structure in enumerate(args.targets):
        target = 7 * math.prod(structure)
        use_coordinate = coordinate_search_needed(6, structure)
        if use_coordinate and any(
            math.gcd(structure[left], structure[right]) != 1
            for left in range(len(structure))
            for right in range(left)
        ):
            raise ValueError(
                f"large-pool structure {structure} is not pairwise coprime"
            )
        search_type = (
            CoordinatePrimarySearch if use_coordinate else PrimarySearch
        )
        restarts = (
            args.coordinate_restarts if use_coordinate else args.restarts
        )
        for fixed_index, fixed_row in enumerate(source_rows):
            mask = residual_mask(forbidden, fixed_row)
            residual = forbidden[mask]
            local_seed = (
                args.seed + 10_007 * target_number + 101 * fixed_index
            )
            print(
                f"\ntarget={target} residual={structure} "
                f"fixed={fixed_index} constraints={int(mask.sum())} "
                f"oracle={'coordinate' if use_coordinate else 'complete'}",
                flush=True,
            )
            search = search_type(residual, structure, seed=local_seed)
            count_started = time.perf_counter()
            count = search.run(
                restarts=restarts,
                max_sweeps=args.sweeps,
                top=args.top,
                progress_every=max(1, restarts // 4),
            )

            def add(
                label: str,
                beta: float,
                rows: Sequence[np.ndarray],
                weights: np.ndarray,
                seconds: float,
                metadata: dict,
            ) -> dict:
                record = candidate_record(
                    label=label,
                    beta=beta,
                    rows=full_rows(fixed_row, rows),
                    moduli=[7, *structure],
                    forbidden=forbidden,
                    ratios=ratios,
                    weights=weights,
                    basis=basis,
                    diameter=diameter,
                    facets=facets,
                    search_seconds=seconds,
                    search_metadata={
                        "fixed_row_index": fixed_index,
                        "residual_structure": structure,
                        "residual_constraints": int(mask.sum()),
                        "oracle": (
                            "coordinate"
                            if use_coordinate
                            else "complete-projective"
                        ),
                        **metadata,
                    },
                )
                if record["image_index"] != target:
                    raise AssertionError(
                        f"candidate image {record['image_index']} != {target}"
                    )
                record["fixed_row_index"] = fixed_index
                record["target_index"] = target
                payload["results"].append(record)
                if record.get("complete_separation", {}).get("valid"):
                    payload["valid_candidate"] = record
                save()
                return record

            count_record = add(
                f"index-{target}-fixed-{fixed_index}-count",
                0.0,
                count.rows,
                np.ones(len(forbidden), dtype=np.float64),
                time.perf_counter() - count_started,
                count.as_json(),
            )
            print(
                f"  count killed={count_record['killed']} "
                f"min={count_record['minimum_conflict_ratio']}",
                flush=True,
            )
            if payload["valid_candidate"] is not None:
                print("*** VALID LOWER-INDEX KERNEL FOUND ***", flush=True)
                return 0

            for power_index, power in enumerate(args.weight_powers):
                weights = np.power(deficits, power)
                weighted_search = search_type(
                    residual,
                    structure,
                    seed=local_seed + 104_729 * (power_index + 1),
                )
                weighted_started = time.perf_counter()
                weighted = weighted_search.run_weighted(
                    weights[mask],
                    restarts=restarts,
                    max_sweeps=args.sweeps,
                    top=args.top,
                    progress_every=max(1, restarts // 4),
                    initial_rows=count.rows,
                )
                record = add(
                    f"index-{target}-fixed-{fixed_index}-deficit-{power:g}",
                    float(power),
                    weighted.rows,
                    weights,
                    time.perf_counter() - weighted_started,
                    weighted.as_json(),
                )
                print(
                    f"  power={power:g} killed={record['killed']} "
                    f"min={record['minimum_conflict_ratio']}",
                    flush=True,
                )
                if payload["valid_candidate"] is not None:
                    print(
                        "*** VALID LOWER-INDEX KERNEL FOUND ***",
                        flush=True,
                    )
                    return 0

    save()
    best = max(
        payload["results"],
        key=lambda record: float(
            record["complete_separation"]["minimum_distance_ratio"]
        ),
    )
    print(
        f"\nFINAL target={best['target_index']} "
        f"moduli={best['moduli']} killed={best['killed']} "
        "min="
        f"{best['complete_separation']['minimum_distance_ratio']:.12g} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
