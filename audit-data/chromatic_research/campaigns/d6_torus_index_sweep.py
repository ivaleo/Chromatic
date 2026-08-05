"""Search small valid periods whose compatible pairs beat 343 colors.

For a period of order ``N`` with ``343 < N <= 684``, a 342-coloring needs only
``N-342`` disjoint compatible pairs; every remaining quotient cell may keep a
singleton color.  This is much weaker than finding a lower-index color
sublattice and is therefore a genuinely non-coset construction.

The default campaign scans every 7-smooth index in that interval whose prime
rank is at most six.  At each index, prime-primary coordinate descent samples
valid periods.  Exact forbidden quotient images define the complement graph,
and a HiGHS maximum-matching MIP either constructs the merged coloring or
proves that the sampled period does not contain a large enough matching.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.active_metric_refine import _load_problem
from chromatic_research.campaigns.d6_torus_period_portfolio import (
    quotient_independent_set_target,
    quotient_matching_coloring,
    signed_connection_images,
)
from chromatic_research.core.determinant_repair import exact_det, load_preset
from chromatic_research.core.prime_radon import (
    PrimarySearch,
    hnf_columns,
    image_size,
    kernel_basis,
    killed_mask,
    smith_diagonal,
)
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


def smooth_prime_structures(
    lower: int,
    upper: int,
    primes: Sequence[int] = (2, 3, 5, 7),
    max_prime_rank: int = 6,
) -> list[tuple[int, list[int]]]:
    """Return repeated-prime quotient structures in an index interval."""
    structures: list[tuple[int, list[int]]] = []
    for value in range(lower, upper + 1):
        residual = value
        moduli: list[int] = []
        valid = True
        for prime in primes:
            exponent = 0
            while residual % prime == 0:
                residual //= prime
                exponent += 1
            if exponent > max_prime_rank:
                valid = False
                break
            moduli.extend([prime] * exponent)
        if valid and residual == 1 and moduli:
            structures.append((value, moduli))
    return structures


def parse_structures(text: str) -> list[tuple[int, list[int]]]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError(
            "expected [[index, [moduli...]], ...]"
        )
    result: list[tuple[int, list[int]]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise argparse.ArgumentTypeError("invalid structure record")
        index = int(item[0])
        moduli = [int(value) for value in item[1]]
        if math.prod(moduli) != index:
            raise argparse.ArgumentTypeError(
                f"moduli {moduli} do not multiply to {index}"
            )
        result.append((index, moduli))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower", type=int, default=344)
    parser.add_argument("--upper", type=int, default=684)
    parser.add_argument("--structures", type=parse_structures)
    parser.add_argument("--samples-per-index", type=int, default=120)
    parser.add_argument("--sweeps", type=int, default=20)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--keep-per-index", type=int, default=8)
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument(
        "--metric",
        type=Path,
        help="optional deformed-metric JSON; defaults to exact E6*",
    )
    parser.add_argument("--temperature", type=float, default=20000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--matching-time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.lower < 2 or args.upper < args.lower:
        parser.error("invalid index interval")
    if (
        args.samples_per_index < 1
        or args.sweeps < 1
        or args.top < 1
        or args.keep_per_index < 1
    ):
        parser.error("search budgets must be positive")
    if args.target_colors < 1 or args.matching_time_limit <= 0:
        parser.error("invalid target or matching time limit")

    structures = (
        args.structures
        if args.structures is not None
        else smooth_prime_structures(args.lower, args.upper)
    )
    metric_source: str | None = None
    if args.metric is None:
        lattice, basis, diameter, _, _ = load_preset("d6")
    else:
        (
            metric_payload,
            _,
            _,
            _,
            _,
            evaluator,
        ) = _load_problem(
            args.metric, args.temperature, args.max_h_norm
        )
        parameters = np.asarray(
            metric_payload["best"]["parameters"], dtype=np.float64
        )
        evaluation = evaluator.evaluate(parameters, with_witnesses=True)
        basis = np.asarray(evaluation.basis, dtype=np.float64)
        diameter = float(evaluation.diameter)
        lattice = "deformed d6 metric"
        metric_source = str(args.metric)
    forbidden, _, _ = _forbidden_with_weights(basis, diameter)
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "valid-period primary search followed by exact compatible-pair "
            "maximum matching for non-coset coloring"
        ),
        "lattice": lattice,
        "metric_source": metric_source,
        "parent_basis": basis.tolist(),
        "parent_diameter": diameter,
        "dimension": basis.shape[0],
        "target_colors": args.target_colors,
        "index_interval": [args.lower, args.upper],
        "structures": [
            {"index": index, "moduli": moduli}
            for index, moduli in structures
        ],
        "forbidden_projective_pairs": len(forbidden),
        "settings": {
            "samples_per_index": args.samples_per_index,
            "sweeps": args.sweeps,
            "top": args.top,
            "keep_per_index": args.keep_per_index,
            "matching_time_limit": args.matching_time_limit,
            "seed": args.seed,
        },
        "index_results": [],
        "coloring": None,
        "valid_combinatorial_witness": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"indices={len(structures)} target={args.target_colors} "
        f"samples/index={args.samples_per_index}",
        flush=True,
    )
    for case_number, (index, moduli) in enumerate(structures):
        necessary_alpha = math.ceil(index / args.target_colors)
        search = PrimarySearch(
            forbidden, moduli, seed=args.seed + 1009 * case_number
        )
        archive: dict[tuple[int, ...], dict] = {}
        valid_samples = 0
        invalid_samples = 0
        statuses: dict[str, int] = {}
        case_record: dict = {
            "index": index,
            "moduli": moduli,
            "necessary_independence_number": necessary_alpha,
            "valid_samples": 0,
            "invalid_samples": 0,
            "unique_valid_kernels": 0,
            "target_status_counts": {},
            "best_connection_count": None,
            "records": [],
        }
        payload["index_results"].append(case_record)
        print(
            f"[{case_number+1}/{len(structures)}] index={index} "
            f"moduli={moduli} need-alpha={necessary_alpha}",
            flush=True,
        )
        for sample in range(args.samples_per_index):
            score, rows, sweeps = search.descend(
                max_sweeps=args.sweeps,
                top=args.top,
                kick_probability=0.15,
                temperature=0.35,
            )
            exact_killed = int(
                killed_mask(forbidden, rows, moduli).sum()
            )
            if exact_killed != int(round(score)):
                raise AssertionError("search score mismatch")
            if exact_killed or image_size(
                rows, moduli, basis.shape[0]
            ) != index:
                invalid_samples += 1
                continue
            valid_samples += 1
            kernel = hnf_columns(
                kernel_basis(rows, moduli, basis.shape[0])
            )
            if abs(exact_det(kernel)) != index:
                raise AssertionError("kernel determinant mismatch")
            key = tuple(int(value) for value in kernel.flat)
            if key in archive:
                continue
            connections = signed_connection_images(
                forbidden, rows, moduli
            )
            decision = quotient_independent_set_target(
                connections,
                moduli,
                necessary_alpha,
                time_limit=args.matching_time_limit,
            )
            statuses[decision["status"]] = (
                statuses.get(decision["status"], 0) + 1
            )
            record: dict = {
                "sample": sample,
                "sweeps": sweeps,
                "rows": [
                    np.asarray(row, dtype=np.int64).astype(int).tolist()
                    for row in rows
                ],
                "kernel_basis_columns": kernel.astype(int).tolist(),
                "kernel_smith": smith_diagonal(kernel),
                "connection_keys": len(connections),
                "missing_nonzero_quotient_classes": (
                    index - 1 - len(connections)
                ),
                "target_independent_set": decision,
            }
            if decision["feasible"]:
                matching = quotient_matching_coloring(
                    connections,
                    moduli,
                    args.target_colors,
                    time_limit=args.matching_time_limit,
                )
                record["matching_coloring"] = matching
                if matching["success"]:
                    payload["coloring"] = {
                        "index": index,
                        "moduli": moduli,
                        "rows": record["rows"],
                        "kernel_basis_columns": record[
                            "kernel_basis_columns"
                        ],
                        "kernel_smith": record["kernel_smith"],
                        "connection_keys": record["connection_keys"],
                        "matching_coloring": matching,
                    }
                    payload["valid_combinatorial_witness"] = True
                    archive[key] = record
                    print(
                        f"FOUND index={index} colors="
                        f"{matching['color_count']} sample={sample} "
                        f"connections={len(connections)}",
                        flush=True,
                    )
                    break
            archive[key] = record

        ranked = sorted(
            archive.values(),
            key=lambda record: (
                record["connection_keys"],
                0
                if record["target_independent_set"]["feasible"]
                else 1,
                record["sample"],
            ),
        )[: args.keep_per_index]
        case_record.update(
            {
                "valid_samples": valid_samples,
                "invalid_samples": invalid_samples,
                "unique_valid_kernels": len(archive),
                "target_status_counts": statuses,
                "best_connection_count": (
                    min(
                        (
                            record["connection_keys"]
                            for record in archive.values()
                        ),
                        default=None,
                    )
                ),
                "records": ranked,
            }
        )
        save()
        print(
            f"  valid={valid_samples} unique={len(archive)} "
            f"best-connections={case_record['best_connection_count']} "
            f"statuses={statuses}",
            flush=True,
        )
        if payload["valid_combinatorial_witness"]:
            break
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
