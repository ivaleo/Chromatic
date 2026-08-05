"""Metric deformation for a periodic coloring with affine color differences.

A paired cyclic coloring of ``Z^n/P`` is not described by one color
sublattice.  If the quotient row is ``a mod N`` and every color contains a
pair whose quotient difference is ``y``, then all same-color displacement
vectors lie in

    A = {z in Z^n : <a,z> in {0,+y,-y} mod N}.

This module generalizes the exhaustive metric oracle from one kernel lattice
to this finite union of affine kernel cosets.  For every deformation it
enumerates all vectors in every coset with norm below ``2*diam(V)``; the
standard inequality

    dist(V,z+V) >= ||z|| - diam(V)

makes this enumeration sufficient for deciding whether the normalized
separation is at least one.  CMA-ES then optimizes the same hard/soft minimum
used by ``metric_deform.py``.

All modular and coset arithmetic is exact.  Voronoi construction, distances,
and optimization are numerical; crossing one must still be rationalized and
independently verified before changing a Euclidean upper bound.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from sympy import Matrix

import combigeo
from d6_cyclic_hole_search import primitive_cyclic_row
from determinant_repair import exact_det, load_preset
from metric_deform import (
    MetricEvaluation,
    exhaustive_covering_radius,
    trace_free_matrix,
)
from prime_radon import hnf_columns, kernel_basis, smith_diagonal


_MPL_CACHE = Path(tempfile.gettempdir()) / "chromatic-affine-metric-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
import cma  # noqa: E402


def bezout_coefficients(values: Sequence[int]) -> tuple[int, list[int]]:
    """Return ``gcd(values)`` and integer coefficients for that gcd."""
    coefficients: list[int] = []
    current_gcd = 0
    for value in values:
        value = int(value)
        old_gcd = current_gcd
        new_gcd = math.gcd(old_gcd, value)
        if old_gcd == 0:
            coefficients = [1 if value >= 0 else -1]
            current_gcd = abs(value)
            continue
        # Extended Euclid for old_gcd*s + value*t = new_gcd.
        a, b = old_gcd, abs(value)
        old_r, r = a, b
        old_s, s = 1, 0
        old_t, t = 0, 1
        while r:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t
        sign = 1 if value >= 0 else -1
        coefficients = [coefficient * old_s for coefficient in coefficients]
        coefficients.append(old_t * sign)
        current_gcd = new_gcd
    return current_gcd, coefficients


def cyclic_residue_representative(
    row: Sequence[int],
    modulus: int,
    residue: int,
) -> np.ndarray:
    """Integer ``z`` satisfying ``row . z == residue (mod modulus)``."""
    row = [int(value) for value in row]
    gcd_value, coefficients = bezout_coefficients([*row, int(modulus)])
    if gcd_value != 1:
        raise ValueError("cyclic row is not primitive")
    representative = np.asarray(coefficients[:-1], dtype=np.int64)
    representative *= int(residue)
    if int(np.dot(np.asarray(row, dtype=np.int64), representative)) % modulus != (
        int(residue) % modulus
    ):
        raise AssertionError("Bezout representative has the wrong residue")
    return representative


def affine_coset_representatives(
    row: Sequence[int],
    modulus: int,
    difference: int,
) -> np.ndarray:
    """Represent the distinct cosets with residues ``0,+y,-y``."""
    return cyclic_residue_representatives(
        row,
        modulus,
        [0, int(difference), -int(difference)],
    )


def cyclic_residue_representatives(
    row: Sequence[int],
    modulus: int,
    residues: Sequence[int],
) -> np.ndarray:
    """Represent an arbitrary finite set of cyclic residue cosets."""
    canonical_residues = sorted(
        {int(residue) % modulus for residue in residues}
    )
    return np.asarray(
        [
            cyclic_residue_representative(row, modulus, residue)
            for residue in canonical_residues
        ],
        dtype=np.int64,
    )


def checkpoint_affine_cosets(
    payload: dict,
    row: Sequence[int],
    modulus: int,
) -> tuple[np.ndarray, int | None, int | None, np.ndarray | None]:
    """Recover affine residue cosets from old and transversal checkpoints."""
    raw_difference = payload.get("target_difference")
    difference = (
        int(raw_difference) if raw_difference is not None else None
    )
    raw_block_size = payload.get("block_size")
    block_size = (
        int(raw_block_size) if raw_block_size is not None else None
    )
    raw_residues = payload.get("difference_residues")
    difference_residues = None
    if raw_residues is not None:
        difference_residues = np.unique(
            np.remainder(
                np.asarray(raw_residues, dtype=np.int64),
                modulus,
            )
        )
        if not len(difference_residues):
            raise ValueError("difference residue set is empty")
        if not np.any(difference_residues == 0):
            raise ValueError("difference residue set must contain zero")
        return (
            cyclic_residue_representatives(
                row,
                modulus,
                difference_residues,
            ),
            difference,
            block_size,
            difference_residues,
        )
    if block_size is not None:
        if block_size < 1:
            raise ValueError("block size must be positive")
        return (
            cyclic_residue_representatives(
                row,
                modulus,
                range(-(block_size - 1), block_size),
            ),
            difference,
            block_size,
            None,
        )
    if difference is not None:
        return (
            affine_coset_representatives(
                row,
                modulus,
                difference,
            ),
            difference,
            None,
            None,
        )
    raise ValueError("checkpoint has no affine difference data")


def canonical_projective_coordinates(
    coordinates: Sequence[Sequence[int]],
) -> np.ndarray:
    """Deduplicate nonzero coordinates modulo central symmetry."""
    unique: dict[tuple[int, ...], tuple[int, ...]] = {}
    for raw in coordinates:
        positive = tuple(int(value) for value in raw)
        if not any(positive):
            continue
        negative = tuple(-value for value in positive)
        key = min(positive, negative)
        unique.setdefault(key, key)
    if not unique:
        width = len(coordinates[0]) if len(coordinates) else 0
        return np.empty((0, width), dtype=np.int64)
    return np.asarray(sorted(unique), dtype=np.int64)


def affine_coordinates_within(
    basis: np.ndarray,
    period_columns: np.ndarray,
    coset_representatives: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Exhaustively enumerate the affine union inside a physical radius."""
    basis = np.asarray(basis, dtype=np.float64)
    period_columns = np.asarray(period_columns, dtype=np.int64)
    representatives = np.asarray(coset_representatives, dtype=np.int64)
    n = len(basis)
    if (
        basis.shape != (n, n)
        or period_columns.shape != (n, n)
        or representatives.ndim != 2
        or representatives.shape[1] != n
    ):
        raise ValueError("basis, period, and cosets have incompatible shapes")
    period_rows = np.asarray(
        Matrix(period_columns.T.tolist()).lll().tolist(),
        dtype=np.int64,
    )
    sub_basis = period_rows @ basis
    inverse = np.linalg.inv(basis)
    coordinates: list[np.ndarray] = []
    for representative in representatives:
        representative_physical = representative @ basis
        lattice_vectors = combigeo._vectors_near(
            sub_basis.tolist(),
            (-representative_physical).tolist(),
            float(radius),
        )
        for raw_lattice_vector in lattice_vectors:
            vector = (
                np.asarray(raw_lattice_vector, dtype=np.float64)
                + representative_physical
            )
            raw_coordinate = vector @ inverse
            coordinate = np.rint(raw_coordinate).astype(np.int64)
            if float(np.max(np.abs(raw_coordinate - coordinate))) > 2e-6:
                raise RuntimeError("could not recover an affine coordinate")
            coordinates.append(coordinate)
    return canonical_projective_coordinates(coordinates)


class AffineMetricEvaluator:
    """Complete Voronoi/separation oracle for a finite affine coset union."""

    def __init__(
        self,
        basis0: np.ndarray,
        period_columns: np.ndarray,
        coset_representatives: np.ndarray,
        *,
        softmin_temperature: float = 60.0,
        max_h_norm: float = 0.8,
    ) -> None:
        self.basis0 = np.asarray(basis0, dtype=np.float64)
        self.period = np.asarray(period_columns, dtype=np.int64)
        self.cosets = np.asarray(coset_representatives, dtype=np.int64)
        self.n = len(self.basis0)
        self.temperature = float(softmin_temperature)
        self.max_h_norm = float(max_h_norm)

    def evaluate(
        self,
        parameters: Sequence[float],
        *,
        with_witnesses: bool = False,
        witness_window: float = 1e-7,
    ) -> MetricEvaluation:
        parameters_array = np.asarray(parameters, dtype=np.float64)
        deformation = trace_free_matrix(parameters_array, self.n)
        h_norm = float(np.linalg.norm(deformation))
        if not np.isfinite(h_norm) or h_norm > self.max_h_norm:
            excess = max(0.0, h_norm - self.max_h_norm)
            objective = 10.0 + excess * excess
            return MetricEvaluation(
                objective=objective,
                soft_min=-objective,
                min_ratio=-objective,
                min_distance=0.0,
                diameter=float("inf"),
                facet_count=0,
                vertex_count=0,
                subvector_count=0,
                h_norm=h_norm,
                parameters=parameters_array,
                basis=np.full_like(self.basis0, np.nan),
                witnesses=[],
            )

        basis = self.basis0 @ expm(deformation)
        facets = combigeo.relevant_facets(basis.tolist())
        radius, vertex_count = exhaustive_covering_radius(facets)
        diameter = 2.0 * radius
        coordinates = affine_coordinates_within(
            basis,
            self.period,
            self.cosets,
            2.0 * diameter + 1e-8,
        )
        values: list[tuple[float, float, np.ndarray]] = []
        for coordinate in coordinates:
            vector = coordinate @ basis
            distance = 2.0 * combigeo.dist_to_halfspaces(
                (0.5 * vector).tolist(),
                facets,
            )
            values.append((distance / diameter, distance, coordinate))
        if values:
            ratios = np.asarray([value[0] for value in values])
            min_ratio = float(ratios.min())
            min_distance = float(min(value[1] for value in values))
            shifted = np.exp(
                -self.temperature * (ratios - min_ratio)
            ).sum()
            soft_min = (
                min_ratio
                - math.log(float(shifted)) / self.temperature
            )
        else:
            min_ratio = 1.0
            min_distance = diameter
            soft_min = 1.0
        witnesses: list[dict] = []
        if with_witnesses:
            cutoff = min_ratio + witness_window
            for ratio, distance, coordinate in sorted(
                values,
                key=lambda item: item[0],
            ):
                if ratio > cutoff:
                    break
                vector = coordinate @ basis
                witnesses.append(
                    {
                        "coordinate": coordinate.astype(int).tolist(),
                        "distance": float(distance),
                        "distance_ratio": float(ratio),
                        "norm_squared": float(vector @ vector),
                    }
                )
        return MetricEvaluation(
            objective=-soft_min,
            soft_min=soft_min,
            min_ratio=min_ratio,
            min_distance=min_distance,
            diameter=diameter,
            facet_count=len(facets),
            vertex_count=vertex_count,
            subvector_count=len(coordinates),
            h_norm=h_norm,
            parameters=parameters_array,
            basis=basis,
            witnesses=witnesses,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.015)
    parser.add_argument("--temperature", type=float, default=120.0)
    parser.add_argument("--max-h-norm", type=float, default=0.8)
    parser.add_argument("--target-margin", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "affine-metric checkpoint whose best parameters seed CMA-ES; "
            "the cyclic row, modulus, and target difference are verified"
        ),
    )
    parser.add_argument(
        "--initial-metric",
        type=Path,
        help=(
            "seed from best deformation parameters of another coloring "
            "checkpoint with the same d6 base metric"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.record_index < 0
        or args.generations < 1
        or args.population < 2
        or args.sigma <= 0
        or args.temperature <= 0
        or args.max_h_norm <= 0
        or (args.resume is not None and args.initial_metric is not None)
    ):
        parser.error("invalid record, budget, or optimizer parameter")

    source = json.loads(args.campaign.read_text())
    if args.record_index >= len(source.get("records", [])):
        parser.error("record index lies outside the campaign")
    record = source["records"][args.record_index]
    row_data = record.get("search", {}).get("row")
    selected_candidate = None
    for candidate_key in (
        "heuristic",
        "soft_cpsat",
        "full_cpsat",
        "coloring",
    ):
        candidate = record.get(candidate_key)
        if (
            row_data is None
            and isinstance(candidate, dict)
            and candidate.get("row") is not None
        ):
            row_data = candidate["row"]
            selected_candidate = candidate
            break
    if row_data is None:
        parser.error("selected record has no cyclic row")
    row = np.asarray(row_data, dtype=np.int64)
    modulus = int(record.get("period_index", record.get("modulus", 0)))
    if modulus < 2:
        parser.error("selected record has no valid cyclic modulus")
    block_size = (
        int(record["block_size"])
        if record.get("block_size") is not None
        else None
    )
    phases = (
        np.asarray(selected_candidate["phases"], dtype=np.int64)
        if selected_candidate is not None
        and selected_candidate.get("phases") is not None
        else None
    )
    difference_residues = None
    if selected_candidate is not None:
        verification = selected_candidate.get("verification") or {}
        raw_residues = verification.get("difference_residues")
        if raw_residues is not None:
            difference_residues = np.unique(
                np.remainder(
                    np.asarray(raw_residues, dtype=np.int64),
                    modulus,
                )
            )
    difference = (
        int(record["target_difference"])
        if block_size is None
        and record.get("target_difference") is not None
        else None
    )
    if block_size is None and difference is None:
        parser.error("selected record has no block or target difference")
    if not primitive_cyclic_row(row, modulus):
        parser.error("selected cyclic row is not primitive")

    lattice, basis0, _, _, _ = load_preset("d6")
    period = hnf_columns(
        kernel_basis([row], [modulus], len(row))
    )
    if abs(exact_det(period)) != modulus:
        raise AssertionError("period determinant differs from cyclic modulus")
    if difference_residues is not None:
        cosets = cyclic_residue_representatives(
            row,
            modulus,
            difference_residues,
        )
    elif block_size is not None:
        cosets = cyclic_residue_representatives(
            row,
            modulus,
            range(-(block_size - 1), block_size),
        )
    else:
        cosets = affine_coset_representatives(
            row,
            modulus,
            difference,
        )
    evaluator = AffineMetricEvaluator(
        basis0,
        period,
        cosets,
        softmin_temperature=args.temperature,
        max_h_norm=args.max_h_norm,
    )
    parameter_count = len(row) * (len(row) + 1) // 2 - 1
    initial = np.zeros(parameter_count, dtype=np.float64)
    baseline = evaluator.evaluate(initial, with_witnesses=True)
    seeded = baseline
    if args.resume is not None:
        try:
            resume_payload = json.loads(args.resume.read_text())
            resume_row = np.asarray(
                resume_payload["cyclic_row"],
                dtype=np.int64,
            )
            resume_modulus = int(resume_payload["period_index"])
            resume_difference = resume_payload.get("target_difference")
            resume_block_size = resume_payload.get("block_size")
            resume_difference_residues = resume_payload.get(
                "difference_residues"
            )
            if resume_difference is not None:
                resume_difference = int(resume_difference)
            if resume_block_size is not None:
                resume_block_size = int(resume_block_size)
            if resume_difference_residues is not None:
                resume_difference_residues = np.unique(
                    np.remainder(
                        np.asarray(
                            resume_difference_residues,
                            dtype=np.int64,
                        ),
                        modulus,
                    )
                )
            initial = np.asarray(
                resume_payload["best"]["parameters"],
                dtype=np.float64,
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid --resume checkpoint: {error}")
        if (
            resume_modulus != modulus
            or resume_block_size != block_size
            or (
                difference_residues is not None
                and (
                    resume_difference_residues is None
                    or not np.array_equal(
                        resume_difference_residues,
                        difference_residues,
                    )
                )
            )
            or (
                difference_residues is None
                and block_size is None
                and (
                    resume_difference is None
                    or resume_difference % modulus
                    != difference % modulus
                )
            )
            or resume_row.shape != row.shape
            or not np.array_equal(resume_row % modulus, row % modulus)
        ):
            parser.error(
                "--resume checkpoint belongs to a different affine coloring"
            )
        if initial.shape != (parameter_count,):
            parser.error(
                "resume parameters have shape "
                f"{initial.shape}, expected {(parameter_count,)}"
            )
        seeded = evaluator.evaluate(initial, with_witnesses=True)
        recorded_ratio = float(resume_payload["best"]["min_ratio"])
        tolerance = max(5e-8, 5e-7 * abs(recorded_ratio))
        if abs(seeded.min_ratio - recorded_ratio) > tolerance:
            parser.error(
                "--resume metric mismatch: recomputed "
                f"{seeded.min_ratio:.12g}, recorded "
                f"{recorded_ratio:.12g}"
            )
    elif args.initial_metric is not None:
        try:
            metric_payload = json.loads(args.initial_metric.read_text())
            initial = np.asarray(
                metric_payload["best"]["parameters"],
                dtype=np.float64,
            )
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid --initial-metric checkpoint: {error}")
        if initial.shape != (parameter_count,):
            parser.error(
                "initial metric parameters have shape "
                f"{initial.shape}, expected {(parameter_count,)}"
            )
        seeded = evaluator.evaluate(initial, with_witnesses=True)
    print(
        f"N={modulus} y={difference} block={block_size} "
        f"differences="
        f"{len(difference_residues) if difference_residues is not None else 'standard'} "
        f"cosets={len(cosets)} "
        f"baseline={baseline.min_ratio:.12f} "
        f"start={seeded.min_ratio:.12f} "
        f"vectors={baseline.subvector_count}",
        flush=True,
    )

    strategy = cma.CMAEvolutionStrategy(
        initial.tolist(),
        args.sigma,
        {
            "popsize": args.population,
            "maxiter": args.generations,
            "seed": args.seed,
            "verbose": -9,
            "tolfun": 1e-11,
            "tolx": 1e-9,
        },
    )
    best = max(
        (baseline, seeded),
        key=lambda item: item.min_ratio,
    )
    best_soft = max(
        (baseline, seeded),
        key=lambda item: item.soft_min,
    )
    evaluations = 0
    started = time.perf_counter()

    def payload(generation: int) -> dict:
        return {
            "method": (
                "exhaustive affine-coset Voronoi metric deformation"
            ),
            "lattice": lattice,
            "source_campaign": str(args.campaign),
            "source_record_index": args.record_index,
            "source_record": record,
            "resume": str(args.resume) if args.resume is not None else None,
            "initial_metric": (
                str(args.initial_metric)
                if args.initial_metric is not None
                else None
            ),
            "period_index": modulus,
            "target_colors": int(
                source.get(
                    "target_colors",
                    record.get("colors"),
                )
            ),
            "target_difference": difference,
            "block_size": block_size,
            "phases": (
                phases.astype(int).tolist() if phases is not None else None
            ),
            "difference_residues": (
                difference_residues.astype(int).tolist()
                if difference_residues is not None
                else None
            ),
            "cyclic_row": row.astype(int).tolist(),
            "period_basis_columns": period.astype(int).tolist(),
            "period_smith": smith_diagonal(period),
            "affine_coset_representatives": cosets.astype(int).tolist(),
            "generation": generation,
            "evaluations": evaluations,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer": {
                "generations": args.generations,
                "population": args.population,
                "sigma": args.sigma,
                "temperature": args.temperature,
                "max_h_norm": args.max_h_norm,
                "seed": args.seed,
            },
            "best": best.as_json(),
            "best_soft_min_seen": best_soft.soft_min,
            "valid_numerical_witness": best.min_ratio >= 1.0,
        }

    generation = 0
    while not strategy.stop() and generation < args.generations:
        generation += 1
        population = strategy.ask()
        summaries = [
            evaluator.evaluate(parameters)
            for parameters in population
        ]
        objectives = [summary.objective for summary in summaries]
        strategy.tell(population, objectives)
        evaluations += len(population)
        soft_candidate = max(summaries, key=lambda item: item.soft_min)
        hard_candidate = max(summaries, key=lambda item: item.min_ratio)
        if soft_candidate.soft_min > best_soft.soft_min + 1e-12:
            best_soft = evaluator.evaluate(
                soft_candidate.parameters,
                with_witnesses=True,
            )
        if hard_candidate.min_ratio > best.min_ratio + 1e-12:
            best = evaluator.evaluate(
                hard_candidate.parameters,
                with_witnesses=True,
            )
            args.output.write_text(
                json.dumps(payload(generation), indent=2) + "\n"
            )
            print(
                f"  new best gen={generation} "
                f"min={best.min_ratio:.12f} "
                f"soft={best.soft_min:.12f} "
                f"|H|={best.h_norm:.5f}",
                flush=True,
            )
        print(
            f"gen {generation:3d}/{args.generations} "
            f"generation={hard_candidate.min_ratio:.9f} "
            f"best={best.min_ratio:.9f} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
        if best.min_ratio >= 1.0 + args.target_margin:
            print("*** numerical affine target reached ***", flush=True)
            break

    best = evaluator.evaluate(best.parameters, with_witnesses=True)
    args.output.write_text(
        json.dumps(payload(generation), indent=2) + "\n"
    )
    print(
        f"FINAL min={best.min_ratio:.12f} "
        f"D={best.min_distance:.12f} "
        f"diam={best.diameter:.12f} "
        f"valid={best.min_ratio >= 1.0}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
