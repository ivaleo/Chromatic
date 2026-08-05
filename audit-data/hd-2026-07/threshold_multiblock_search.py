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
import itertools
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
    _prime_power,
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


def free_submodules_prime_power(
    n: int,
    modulus: int,
    rank: int,
):
    """Canonical free rank-``rank`` direct summands of (Z/q Z)^n.

    Let ``q=p^e``.  Reduction modulo ``p`` gives a unique field row space,
    represented by its RREF matrix.  For its pivot columns fixed to the
    identity, every nonpivot entry has exactly ``q/p`` lifts.  This systematic
    form is unique up to GL(rank, Z/q Z), so it removes the full row-basis
    symmetry for repeated prime-power blocks such as ``[4,4]``.
    """
    prime, _ = _prime_power(modulus)
    if rank < 1 or rank > n:
        raise ValueError("submodule rank must lie in [1,n]")
    lift_count = modulus // prime
    for reduced in rref_subspaces(n, prime, rank):
        reduced = np.asarray(reduced, dtype=np.int64)
        pivots = []
        for row in reduced:
            nonzero = np.flatnonzero(row)
            if not len(nonzero):
                raise AssertionError("RREF generator returned a zero row")
            pivots.append(int(nonzero[0]))
        nonpivots = [
            coordinate
            for coordinate in range(n)
            if coordinate not in set(pivots)
        ]
        free_entries = rank * len(nonpivots)
        for encoded in range(lift_count**free_entries):
            option = reduced.copy()
            value = encoded
            for row_index in range(rank):
                for coordinate in nonpivots:
                    digit = value % lift_count
                    value //= lift_count
                    option[row_index, coordinate] = (
                        option[row_index, coordinate] + prime * digit
                    ) % modulus
            yield option


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
        if multiplicity > 1:
            if multiplicity > n:
                raise ValueError("block rank cannot exceed dimension")
            prime, exponent = _prime_power(modulus)
            if exponent == 1:
                options = np.asarray(
                    list(rref_subspaces(n, modulus, multiplicity)),
                    dtype=np.int64,
                )
                expected = gaussian_binomial(
                    n, multiplicity, modulus
                )
                label = f"F_{modulus}-rank-{multiplicity}"
            else:
                options = np.asarray(
                    list(
                        free_submodules_prime_power(
                            n, modulus, multiplicity
                        )
                    ),
                    dtype=np.int64,
                )
                expected = (
                    gaussian_binomial(n, multiplicity, prime)
                    * (modulus // prime)
                    ** (multiplicity * (n - multiplicity))
                )
                label = (
                    f"Z/{modulus}-free-rank-{multiplicity}"
                )
            if len(options) != expected:
                raise AssertionError(
                    f"subspace pool has {len(options)} entries, expected {expected}"
                )
            groups.append(
                BlockGroup(
                    label=label,
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

    for start in range(0, len(group.options), batch_size):
        stop = min(len(group.options), start + batch_size)
        options = group.options[start:stop]
        killed = np.ones(
            (stop - start, constraint_count), dtype=bool
        )
        for row_index, modulus in enumerate(group.moduli):
            killed &= (
                options[:, row_index, :] @ forbidden.T
            ) % modulus == 0
        packed = np.packbits(killed, axis=1, bitorder="little")
        padded = np.zeros(
            (stop - start, byte_count), dtype=np.uint8
        )
        padded[:, : packed.shape[1]] = packed
        masks[start:stop] = padded.view(np.uint64)
    return masks


def row_module_key(
    rows: Sequence[Sequence[int]] | np.ndarray,
    modulus: int,
) -> tuple[tuple[int, ...], ...]:
    """Basis-independent key for the finite row submodule modulo ``modulus``."""
    matrix = np.asarray(rows, dtype=np.int64) % modulus
    if matrix.ndim != 2:
        raise ValueError("rows must be a matrix")
    combinations = itertools.product(
        range(modulus), repeat=len(matrix)
    )
    vectors = {
        tuple(
            int(value)
            for value in (
                np.asarray(coefficients, dtype=np.int64) @ matrix
            )
            % modulus
        )
        for coefficients in combinations
    }
    return tuple(sorted(vectors))


def source_option_index(
    group: BlockGroup,
    source_rows: Sequence[Sequence[int]] | np.ndarray,
) -> int | None:
    """Locate source rows in a symmetry-reduced option pool."""
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if source_rows.shape != group.options.shape[1:]:
        return None
    if len(set(group.moduli)) != 1:
        return None
    modulus = group.moduli[0]
    target = row_module_key(source_rows, modulus)
    for index, option in enumerate(group.options):
        if row_module_key(option, modulus) == target:
            return index
    return None


def source_option_distances(
    group: BlockGroup,
    source_rows: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray | None:
    """Basis/projective distance from every option to a source block."""
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if source_rows.shape != group.options.shape[1:]:
        return None
    if len(set(group.moduli)) != 1:
        return None
    modulus = group.moduli[0]
    if len(group.moduli) == 1:
        units = [
            value
            for value in range(1, modulus)
            if math.gcd(value, modulus) == 1
        ]
        orbit = np.asarray(
            [
                (unit * source_rows[0]) % modulus
                for unit in units
            ],
            dtype=np.int64,
        )
        return np.asarray(
            [
                int(
                    np.count_nonzero(
                        orbit != option[0][None, :], axis=1
                    ).min()
                )
                for option in group.options
            ],
            dtype=np.int64,
        )

    target = set(row_module_key(source_rows, modulus))
    distances = np.empty(len(group.options), dtype=np.int64)
    for index, option in enumerate(group.options):
        option_module = set(row_module_key(option, modulus))
        distances[index] = len(target) - len(
            target.intersection(option_module)
        )
    return distances


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
    option_orders = []
    group_metadata = []
    source_record = metric.get("source_record", {})
    source_moduli = source_record.get("moduli", [])
    source_rows_raw = source_record.get("rows", [])
    source_by_modulus: dict[int, list[np.ndarray]] = {}
    if (
        isinstance(source_moduli, list)
        and isinstance(source_rows_raw, list)
        and len(source_moduli) == len(source_rows_raw)
    ):
        for modulus, row in zip(source_moduli, source_rows_raw):
            source_by_modulus.setdefault(int(modulus), []).append(
                np.asarray(row, dtype=np.int64)
            )
    for group in groups:
        group_masks = killed_masks(bad, group)
        popcounts = np.bitwise_count(group_masks).sum(
            axis=1, dtype=np.int64
        )
        # Low-popcount masks are more likely to make an empty intersection.
        # Sorting changes only the early-feasible search order; the exhaustive
        # Cartesian product and the exact UNSAT conclusion are unchanged.
        option_order = np.argsort(popcounts, kind="stable")
        source_option = None
        source_distances = None
        if len(set(group.moduli)) == 1:
            candidates = source_by_modulus.get(group.moduli[0], [])
            if len(candidates) == len(group.moduli):
                source_distances = source_option_distances(
                    group, candidates
                )
        if source_distances is not None:
            source_option = int(np.argmin(source_distances))
            if source_distances[source_option] != 0:
                raise AssertionError(
                    f"could not locate source option for {group.label}"
                )
            option_order = np.lexsort(
                (popcounts, source_distances)
            )
        masks.append(group_masks[option_order])
        option_orders.append(option_order)
        group_metadata.append(
            {
                "label": group.label,
                "moduli": group.moduli,
                "options": int(len(group.options)),
                "mask_words": int(group_masks.shape[1]),
                "killed_popcount": {
                    "minimum": int(popcounts.min()),
                    "median": float(np.median(popcounts)),
                    "maximum": int(popcounts.max()),
                },
                "enumeration_order": "ascending killed popcount",
                "source_option": (
                    int(source_option)
                    if source_option is not None
                    else None
                ),
                "source_option_first": source_option is not None,
                "source_distance": (
                    {
                        "minimum": int(source_distances.min()),
                        "median": float(np.median(source_distances)),
                        "maximum": int(source_distances.max()),
                    }
                    if source_distances is not None
                    else None
                ),
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
        sorted_choices = [
            int(value) for value in enumeration["choices"]
        ]
        choices = [
            int(order[choice])
            for order, choice in zip(option_orders, sorted_choices)
        ]
        enumeration["sorted_choices"] = sorted_choices
        enumeration["choices"] = choices
        rows: list[np.ndarray] = []
        moduli: list[int] = []
        for group, choice in zip(groups, choices):
            option = group.options[choice]
            group_prime, _ = _prime_power(group.moduli[0])
            if len(group.moduli) > 1 and rank_mod(
                option, group_prime
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
