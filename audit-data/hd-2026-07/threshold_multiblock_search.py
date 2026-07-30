"""Exact fixed-metric threshold search across complete block products.

Each option of a quotient block is converted to a bitmask of below-threshold
vectors that it kills.  A coloring kernel meets the threshold exactly when
the intersection of one mask from every block is empty.  The companion
``threshold_mask_enum.cpp`` exhausts that finite Cartesian product.

Repeated equal prime blocks are quotiented by their full row-space symmetry:
for example ``[3,3,3,5]`` uses the 1,210 rank-three subspaces of F_3^5 rather
than 121^3 ordered triples.  This is an exact symmetry reduction.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import combigeo
from block_row_metric_opt import candidate_record
from prime_radon import (
    gaussian_binomial,
    projective_forms,
    rank_mod,
    rref_subspaces,
)
from prime_row_opt import _forbidden_with_weights, _source_lattice


@dataclass
class BlockGroup:
    label: str
    moduli: list[int]
    # Shape: option x row x coordinate.
    options: np.ndarray


def build_groups(n: int, structure: Sequence[int]) -> list[BlockGroup]:
    """Build canonical option pools, grouping repeated equal primes."""
    structure = [int(value) for value in structure]
    groups: list[BlockGroup] = []
    used = [False] * len(structure)
    for index, modulus in enumerate(structure):
        if used[index]:
            continue
        same = [
            position
            for position in range(index, len(structure))
            if not used[position] and structure[position] == modulus
        ]
        multiplicity = len(same)
        is_prime = modulus >= 2 and all(
            modulus % divisor
            for divisor in range(2, int(math.isqrt(modulus)) + 1)
        )
        if multiplicity > 1:
            if not is_prime:
                raise ValueError(
                    "repeated blocks are symmetry-reduced only for primes"
                )
            if multiplicity > n:
                raise ValueError("block rank cannot exceed dimension")
            options = np.asarray(
                list(rref_subspaces(n, modulus, multiplicity)),
                dtype=np.int64,
            )
            expected = gaussian_binomial(n, multiplicity, modulus)
            if len(options) != expected:
                raise AssertionError(
                    f"subspace pool has {len(options)} entries, expected {expected}"
                )
            groups.append(
                BlockGroup(
                    label=f"F_{modulus}-rank-{multiplicity}",
                    moduli=[modulus] * multiplicity,
                    options=options,
                )
            )
            for position in same:
                used[position] = True
        else:
            rows = projective_forms(n, modulus)
            groups.append(
                BlockGroup(
                    label=f"row-mod-{modulus}",
                    moduli=[modulus],
                    options=rows[:, None, :],
                )
            )
            used[index] = True
    return groups


def killed_masks(
    forbidden: np.ndarray,
    group: BlockGroup,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Return option x uint64-word masks of constraints killed by a group."""
    constraint_count = len(forbidden)
    word_count = max(1, (constraint_count + 63) // 64)
    masks = np.zeros((len(group.options), word_count), dtype=np.uint64)
    byte_count = word_count * 8
    if group.options.shape[1] == 1:
        modulus = group.moduli[0]
        rows = group.options[:, 0, :]
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            killed = (rows[start:stop] @ forbidden.T) % modulus == 0
            packed = np.packbits(killed, axis=1, bitorder="little")
            padded = np.zeros((stop - start, byte_count), dtype=np.uint8)
            padded[:, : packed.shape[1]] = packed
            masks[start:stop] = padded.view(np.uint64)
        return masks

    for option_index, rows in enumerate(group.options):
        killed = np.ones(constraint_count, dtype=bool)
        for row, modulus in zip(rows, group.moduli):
            killed &= (forbidden @ row) % modulus == 0
        packed = np.packbits(killed, bitorder="little")
        padded = np.zeros(byte_count, dtype=np.uint8)
        padded[: len(packed)] = packed
        masks[option_index] = padded.view(np.uint64)
    return masks


def run_enumerator(
    masks: Sequence[np.ndarray],
    executable: Path,
    workers: int,
) -> dict:
    word_count = int(masks[0].shape[1])
    process = subprocess.Popen(
        [str(executable), str(workers), "binary"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to open enumerator pipes")
    header = np.asarray([len(masks), word_count], dtype=np.uint64)
    process.stdin.write(header.tobytes())
    for block in masks:
        if block.shape[1] != word_count:
            process.kill()
            raise ValueError("all mask blocks must have the same width")
        process.stdin.write(np.asarray([len(block)], dtype=np.uint64).tobytes())
        process.stdin.write(
            np.ascontiguousarray(block, dtype=np.uint64).tobytes()
        )
    process.stdin.close()
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            [str(executable), str(workers), "binary"],
            output=stdout,
            stderr=stderr,
        )
    result = json.loads(stdout.decode())
    result["stderr"] = stderr.decode()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", type=Path)
    parser.add_argument("--structure", type=json.loads, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--enumerator", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if (
        not isinstance(args.structure, list)
        or not args.structure
        or any(int(value) < 2 for value in args.structure)
    ):
        parser.error("--structure must be a nonempty JSON list of moduli")
    structure = [int(value) for value in args.structure]
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    metric = json.loads(args.metric.read_text())
    lattice = _source_lattice(args.metric, metric)
    basis = np.asarray(metric["best"]["basis"], dtype=np.float64)
    diameter = float(metric["best"]["diameter"])
    facets = combigeo.relevant_facets(basis.tolist())
    forbidden, ratios, weights = _forbidden_with_weights(
        basis, diameter, max(1.0, args.threshold)
    )
    bad = forbidden[ratios < args.threshold]
    groups = build_groups(len(basis), structure)
    started = time.perf_counter()
    masks = []
    group_metadata = []
    for group in groups:
        group_masks = killed_masks(bad, group)
        masks.append(group_masks)
        group_metadata.append(
            {
                "label": group.label,
                "moduli": group.moduli,
                "options": int(len(group.options)),
                "mask_words": int(group_masks.shape[1]),
            }
        )
        print(
            f"{group.label}: options={len(group.options)} "
            f"bad={len(bad)}",
            flush=True,
        )

    enumeration = run_enumerator(masks, args.enumerator, args.workers)
    payload: dict = {
        "method": (
            "complete bitmask-product threshold enumeration with exact "
            "row-space symmetry reduction"
        ),
        "source_metric": str(args.metric),
        "lattice": lattice,
        "dimension": len(basis),
        "structure": structure,
        "target_index": int(math.prod(structure)),
        "threshold": args.threshold,
        "forbidden_projective_pairs": int(len(forbidden)),
        "below_threshold_pairs": int(len(bad)),
        "groups": group_metadata,
        "enumeration": enumeration,
        "candidate": None,
        "valid_candidate": None,
        "elapsed_seconds": time.perf_counter() - started,
    }

    if enumeration.get("status") == "FEASIBLE":
        choices = [int(value) for value in enumeration["choices"]]
        rows: list[np.ndarray] = []
        moduli: list[int] = []
        for group, choice in zip(groups, choices):
            option = group.options[choice]
            if len(group.moduli) > 1 and rank_mod(
                option, group.moduli[0]
            ) != len(group.moduli):
                raise AssertionError("enumerated field subspace lost rank")
            rows.extend(np.asarray(row, dtype=np.int64) for row in option)
            moduli.extend(group.moduli)
        record = candidate_record(
            label=f"threshold-{args.threshold:.12g}",
            beta=float(args.threshold),
            rows=rows,
            moduli=moduli,
            forbidden=forbidden,
            ratios=ratios,
            weights=weights,
            basis=basis,
            diameter=diameter,
            facets=facets,
            search_seconds=time.perf_counter() - started,
            search_metadata={
                "choices": choices,
                "enumeration": enumeration,
            },
        )
        if record["image_index"] != math.prod(structure):
            raise AssertionError("candidate does not have the requested index")
        payload["candidate"] = record
        if record.get("complete_separation", {}).get("valid"):
            payload["valid_candidate"] = record
        print(
            f"FEASIBLE killed={record['killed']} "
            f"min-ratio={record['minimum_conflict_ratio']}",
            flush=True,
        )
    elif enumeration.get("status") == "INFEASIBLE":
        expected = math.prod(len(group.options) for group in groups)
        if int(enumeration["tested"]) != expected:
            raise AssertionError("enumerator returned incomplete UNSAT")
        print(
            f"UNSAT: exhausted {expected} canonical block tuples",
            flush=True,
        )
    else:
        raise RuntimeError(f"unexpected enumerator result: {enumeration}")

    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
