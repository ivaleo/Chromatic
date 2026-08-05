"""Exact bitset screen of every prime cover of the d6 index-343 kernel.

The generic torus oracle builds an HNF quotient and a CP-SAT model for every
character.  That is unnecessarily expensive for a prime refinement

    P = K ker(a mod p).

Every quotient vertex can instead be written as ``(g,t)``, where ``g`` is a
source class in ``Z^n/K`` and ``t`` is the extra prime residue.  A precomputed
integer cocycle gives subtraction in these coordinates.  For one character,
the forbidden catalogue becomes one small residue mask per source class.

After translation fixes ``(0,0)``, an independent set is a clique in the
compatibility graph of its non-neighbours.  The target is only ``p+1`` for a
``p``-cover, so a Tomita-style bitset clique decision with greedy coloring is
substantially faster than materializing hundreds of thousands of CP-SAT edge
constraints.  The calculation uses exact modular arithmetic throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.campaigns.d6_periodic_lift_highs import (
    quotient_key,
    quotient_map,
    quotient_representatives,
)
from chromatic_research.campaigns.d6_torus_column_generation import (
    _json_profile,
    character_extension_profiles,
    source_extension_coordinates,
)
from chromatic_research.core.determinant_repair import exact_det, load_preset
from chromatic_research.core.prime_radon import projective_forms
from chromatic_research.core.prime_row_opt import _forbidden_with_weights


@dataclass(frozen=True)
class PrimeExtensionModel:
    """Exact source quotient and its representative subtraction cocycle."""

    source_kernel: np.ndarray
    representatives: np.ndarray
    zero_class: int
    difference_classes: np.ndarray
    correction_coordinates: np.ndarray

    @property
    def source_order(self) -> int:
        return int(len(self.representatives))


def build_prime_extension_model(
    source_kernel: np.ndarray,
) -> PrimeExtensionModel:
    """Precompute ``r_h-r_g=r_{h-g}+K*c(g,h)`` exactly."""
    source_kernel = np.asarray(source_kernel, dtype=np.int64)
    determinant, adjugate_object = quotient_map(source_kernel)
    representatives_list, index_by_key = quotient_representatives(
        source_kernel
    )
    representatives = np.asarray(representatives_list, dtype=np.int64)
    adjugate = np.asarray(adjugate_object, dtype=np.int64)
    keys = np.asarray(
        [
            quotient_key(representative, adjugate_object, determinant)
            for representative in representatives
        ],
        dtype=np.int64,
    )
    difference_classes = np.empty(
        (determinant, determinant),
        dtype=np.int32,
    )
    for left in range(determinant):
        difference_keys = np.remainder(
            keys - keys[left],
            determinant,
        )
        difference_classes[left] = [
            index_by_key[tuple(int(value) for value in key)]
            for key in difference_keys
        ]

    representative_differences = (
        representatives[None, :, :]
        - representatives[:, None, :]
        - representatives[difference_classes]
    )
    signed_determinant = int(exact_det(source_kernel))
    numerators = np.einsum(
        "ij,abj->abi",
        adjugate,
        representative_differences,
        optimize=True,
    )
    if np.any(np.remainder(numerators, signed_determinant) != 0):
        raise AssertionError("source representative cocycle is not integral")
    corrections = numerators // signed_determinant
    zero_key = tuple(0 for _ in range(source_kernel.shape[0]))
    zero_class = int(index_by_key[zero_key])
    return PrimeExtensionModel(
        source_kernel=source_kernel.copy(),
        representatives=representatives,
        zero_class=zero_class,
        difference_classes=difference_classes,
        correction_coordinates=corrections.astype(np.int32),
    )


def prime_connection_masks(
    coordinates: np.ndarray,
    class_ids: np.ndarray,
    source_order: int,
    character: Sequence[int],
    prime: int,
) -> np.ndarray:
    """Return the exact forbidden extra-residue mask in every source class."""
    coordinates = np.asarray(coordinates, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    character = np.asarray(character, dtype=np.int64)
    if coordinates.shape[1] != len(character):
        raise ValueError("character and extension coordinates are incompatible")
    if len(class_ids) != len(coordinates):
        raise ValueError("one source class is required per coordinate")
    if prime < 2 or prime > 15:
        raise ValueError("bit masks currently require 2 <= prime <= 15")
    residues = np.remainder(coordinates @ character, prime)
    masks = np.zeros(source_order, dtype=np.uint16)
    bits = np.left_shift(
        np.uint16(1),
        residues.astype(np.uint16),
    )
    np.bitwise_or.at(masks, class_ids, bits)
    return masks


def _boolean_row_to_int(values: np.ndarray) -> int:
    packed = np.packbits(
        np.asarray(values, dtype=np.uint8),
        bitorder="little",
    )
    return int.from_bytes(packed.tobytes(), byteorder="little")


def prime_extension_compatibility_graph(
    model: PrimeExtensionModel,
    masks: np.ndarray,
    character: Sequence[int],
    prime: int,
) -> dict:
    """Build bitset adjacency on all non-neighbours of the identity."""
    masks = np.asarray(masks, dtype=np.uint16)
    character = np.asarray(character, dtype=np.int64)
    if masks.shape != (model.source_order,):
        raise ValueError("one residue mask is required per source class")
    if len(character) != model.source_kernel.shape[0]:
        raise ValueError("character dimension differs from source dimension")
    correction_residues = np.remainder(
        np.einsum(
            "abi,i->ab",
            model.correction_coordinates,
            character,
            optimize=True,
        ),
        prime,
    ).astype(np.int16)

    all_source_classes = np.repeat(
        np.arange(model.source_order, dtype=np.int32),
        prime,
    )
    all_residues = np.tile(
        np.arange(prime, dtype=np.int16),
        model.source_order,
    )
    forbidden_from_identity = np.right_shift(
        masks[all_source_classes],
        all_residues.astype(np.uint16),
    ) & np.uint16(1)
    candidate_mask = (
        (all_source_classes == model.zero_class)
        | (forbidden_from_identity == 0)
    )
    source_classes = all_source_classes[candidate_mask]
    residues = all_residues[candidate_mask]
    identity_positions = np.flatnonzero(
        (source_classes == model.zero_class) & (residues == 0)
    )
    if len(identity_positions) != 1:
        raise AssertionError("the identity candidate is not unique")
    identity = int(identity_positions[0])

    adjacency: list[int] = []
    for left in range(len(source_classes)):
        difference_classes = model.difference_classes[
            source_classes[left],
            source_classes,
        ]
        difference_residues = np.remainder(
            residues
            - residues[left]
            + correction_residues[
                source_classes[left],
                source_classes,
            ],
            prime,
        )
        conflicts = (
            np.right_shift(
                masks[difference_classes],
                difference_residues.astype(np.uint16),
            )
            & np.uint16(1)
        ).astype(bool)
        compatible = ~conflicts
        compatible[left] = False
        adjacency.append(_boolean_row_to_int(compatible))

    full_mask = (1 << len(adjacency)) - 1
    expected_identity_neighbours = full_mask ^ (1 << identity)
    if adjacency[identity] != expected_identity_neighbours:
        raise AssertionError(
            "candidate reduction retained an identity conflict"
        )
    for left, neighbours in enumerate(adjacency):
        for right in range(left + 1, len(adjacency)):
            if bool(neighbours & (1 << right)) != bool(
                adjacency[right] & (1 << left)
            ):
                raise AssertionError("compatibility graph is not symmetric")
    return {
        "source_classes": source_classes,
        "residues": residues,
        "identity": identity,
        "adjacency": adjacency,
    }


def bitset_clique_target(
    adjacency: Sequence[int],
    target_size: int,
    *,
    candidate_mask: int | None = None,
    time_limit: float = 10.0,
) -> dict:
    """Exact fixed-cardinality clique decision with greedy coloring bounds."""
    adjacency = [int(value) for value in adjacency]
    vertex_count = len(adjacency)
    if target_size < 0 or target_size > vertex_count:
        raise ValueError("target clique size lies outside the graph")
    if time_limit <= 0:
        raise ValueError("time limit must be positive")
    if candidate_mask is None:
        candidate_mask = (1 << vertex_count) - 1
    else:
        candidate_mask = int(candidate_mask)
    if candidate_mask >> vertex_count:
        raise ValueError("candidate mask contains an unknown vertex")
    if target_size == 0:
        return {
            "status": "OPTIMAL",
            "feasible": True,
            "proven_infeasible": False,
            "vertices": [],
            "nodes": 0,
            "elapsed_seconds": 0.0,
            "solver": "closed form",
        }
    if candidate_mask.bit_count() < target_size:
        return {
            "status": "INFEASIBLE",
            "feasible": False,
            "proven_infeasible": True,
            "vertices": [],
            "nodes": 0,
            "elapsed_seconds": 0.0,
            "solver": "cardinality precheck",
        }

    started = time.perf_counter()
    deadline = started + time_limit
    nodes = 0
    timed_out = False
    witness: list[int] | None = None

    def color_order(vertices: int) -> tuple[list[int], list[int]]:
        order: list[int] = []
        bounds: list[int] = []
        remaining = vertices
        color = 0
        while remaining:
            color += 1
            available = remaining
            while available:
                bit = available & -available
                vertex = bit.bit_length() - 1
                remaining &= ~bit
                available &= ~bit
                available &= ~adjacency[vertex]
                order.append(vertex)
                bounds.append(color)
        return order, bounds

    def expand(vertices: int, chosen: list[int]) -> bool:
        nonlocal nodes, timed_out, witness
        nodes += 1
        if nodes & 1023 == 0 and time.perf_counter() > deadline:
            timed_out = True
            return False
        if len(chosen) >= target_size:
            witness = chosen.copy()
            return True
        if len(chosen) + vertices.bit_count() < target_size:
            return False
        order, color_bounds = color_order(vertices)
        for position in range(len(order) - 1, -1, -1):
            if len(chosen) + color_bounds[position] < target_size:
                return False
            vertex = order[position]
            bit = 1 << vertex
            if not vertices & bit:
                continue
            chosen.append(vertex)
            if expand(vertices & adjacency[vertex], chosen):
                return True
            chosen.pop()
            if timed_out:
                return False
            vertices &= ~bit
        return False

    feasible = expand(candidate_mask, [])
    elapsed = time.perf_counter() - started
    return {
        "status": (
            "FEASIBLE"
            if feasible
            else ("UNKNOWN" if timed_out else "INFEASIBLE")
        ),
        "feasible": feasible,
        "proven_infeasible": not feasible and not timed_out,
        "vertices": witness or [],
        "nodes": nodes,
        "elapsed_seconds": elapsed,
        "solver": "exact bitset clique decision with greedy coloring bound",
    }


def prime_extension_target(
    model: PrimeExtensionModel,
    coordinates: np.ndarray,
    class_ids: np.ndarray,
    character: Sequence[int],
    prime: int,
    target_size: int,
    *,
    time_limit: float = 10.0,
) -> dict:
    """Decide a target independent-set size in prime-extension coordinates."""
    started = time.perf_counter()
    masks = prime_connection_masks(
        coordinates,
        class_ids,
        model.source_order,
        character,
        prime,
    )
    compatible = prime_extension_compatibility_graph(
        model,
        masks,
        character,
        prime,
    )
    identity = int(compatible["identity"])
    target_companions = target_size - 1
    decision = bitset_clique_target(
        compatible["adjacency"],
        target_companions,
        candidate_mask=compatible["adjacency"][identity],
        time_limit=time_limit,
    )
    witness = [identity, *decision["vertices"]] if decision["feasible"] else []
    return {
        **decision,
        "vertices": witness,
        "vertex_labels": (
            [
                [
                    int(compatible["source_classes"][vertex]),
                    int(compatible["residues"][vertex]),
                ]
                for vertex in witness
            ]
            if witness
            else []
        ),
        "candidate_vertices": len(compatible["adjacency"]),
        "connection_count": int(
            sum(int(mask).bit_count() for mask in masks)
        ),
        "minimum_source_coverage": int(
            min(
                int(mask).bit_count()
                for index, mask in enumerate(masks)
                if index != model.zero_class
            )
        ),
        "total_elapsed_seconds": time.perf_counter() - started,
    }


def _parse_primes(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one prime is required")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primes", type=_parse_primes, default=[3, 5, 7])
    parser.add_argument("--max-exceptions", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--target-colors", type=int, default=342)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.max_exceptions < 0
        or args.time_limit <= 0
        or args.progress_every < 1
        or args.target_colors < 1
    ):
        parser.error("invalid budget, time limit, progress, or target")

    lattice, basis, diameter, _, source_kernel = load_preset("d6")
    forbidden, _, _ = _forbidden_with_weights(basis, diameter)
    coordinates, class_ids, source_order = source_extension_coordinates(
        source_kernel,
        forbidden,
    )
    model = build_prime_extension_model(source_kernel)
    if model.source_order != source_order:
        raise AssertionError("source quotient orders differ")
    started = time.perf_counter()
    payload: dict = {
        "method": (
            "complete prime-cover residue enumeration with an exact "
            "source-cocycle bitset clique decision"
        ),
        "lattice": lattice,
        "dimension": len(source_kernel),
        "source_index": source_order,
        "source_kernel_basis_columns": source_kernel.astype(int).tolist(),
        "source_kernel_determinant": abs(exact_det(source_kernel)),
        "forbidden_projective_pairs": len(forbidden),
        "target_colors": args.target_colors,
        "settings": {
            "primes": args.primes,
            "max_exceptions": args.max_exceptions,
            "time_limit": args.time_limit,
            "progress_every": args.progress_every,
        },
        "prime_screens": [],
        "records": [],
        "candidate_periods": [],
        "complete_all_requested_characters": False,
    }

    def save() -> None:
        payload["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    for prime in args.primes:
        characters = projective_forms(len(source_kernel), prime)
        profile = character_extension_profiles(
            coordinates,
            class_ids,
            source_order,
            characters,
            prime,
        )
        public_profile = _json_profile(profile)
        ranked = sorted(
            profile["incomplete_character_indices"],
            key=lambda index: (
                int(profile["_connection_counts"][index]),
                int(profile["_minimum_coverages"][index]),
                int(index),
            ),
        )
        if args.max_exceptions:
            ranked = ranked[: args.max_exceptions]
        public_profile["exact_exception_selected"] = len(ranked)
        public_profile["exact_exception_unscreened"] = (
            len(profile["incomplete_character_indices"]) - len(ranked)
        )
        payload["prime_screens"].append(public_profile)
        save()

        prime_started = time.perf_counter()
        for local_index, character_index in enumerate(ranked, start=1):
            necessary_alpha = int(
                math.ceil(source_order * prime / args.target_colors)
            )
            decision = prime_extension_target(
                model,
                coordinates,
                class_ids,
                characters[character_index],
                prime,
                necessary_alpha,
                time_limit=args.time_limit,
            )
            record = {
                "prime": prime,
                "character_index": int(character_index),
                "character": characters[character_index].astype(int).tolist(),
                "necessary_independence_number_for_target": necessary_alpha,
                **decision,
            }
            if decision["proven_infeasible"]:
                record["independence_number"] = prime
                record["independence_number_proof"] = (
                    "one source fiber gives alpha >= p and the exact bitset "
                    "decision proves alpha < p+1"
                )
            else:
                payload["candidate_periods"].append(record)
            payload["records"].append(record)
            if (
                local_index % args.progress_every == 0
                or local_index == len(ranked)
            ):
                save()
                rate = local_index / max(
                    time.perf_counter() - prime_started,
                    1e-9,
                )
                print(
                    f"p={prime} {local_index}/{len(ranked)} "
                    f"rate={rate:.2f}/s "
                    f"candidates={len(payload['candidate_periods'])}",
                    flush=True,
                )

    payload["complete_all_requested_characters"] = bool(
        args.max_exceptions == 0
        and all(
            screen["exact_exception_unscreened"] == 0
            for screen in payload["prime_screens"]
        )
    )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
