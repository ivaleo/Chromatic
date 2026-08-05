"""Continuous metric deformation for a fixed prime/Radon coloring kernel.

The modular search chooses a fixed sublattice C Z^n of the parent lattice.
This module then changes the parent Gram form while keeping C fixed.  Unlike
the older integer ``best_killed`` Gram objective, the objective here is the
continuous normalized separation

    min_{0 != z in C Z^n} dist(V, z + V) / diam(V).

The covering radius is not sampled: every Voronoi vertex is enumerated by the
dual Qhull halfspace intersection in each evaluation.  Potentially dangerous
sublattice vectors are also exhaustive, using

    dist(V, z + V) >= |z| - diam(V).

Thus only vectors with |z| < 2 diam(V) need to be inspected.

Volume-preserving deformations are parameterized as

    B = B0 exp(H),       H = H^T, trace(H) = 0.

CMA-ES maximizes a soft minimum of all exact D/diam values.  Final records
contain the hard minimum, all witnesses, the deformed basis and Gram matrix,
and the integer coloring kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import expm
from scipy.spatial import HalfspaceIntersection

import combigeo
from e7_abpr import M_E7
from prime_radon import hnf_columns, kernel_basis, load_forbidden, smith_diagonal


_MPL_CACHE = Path(tempfile.gettempdir()) / "chromatic-metric-mpl"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
import cma  # noqa: E402  (MPLCONFIGDIR must be set first)


def parse_moduli(text: str) -> list[int]:
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise argparse.ArgumentTypeError("moduli must be a non-empty JSON list")
    return [int(value) for value in values]


def resolve_saved_path(anchor: Path, saved: str | Path) -> Path:
    """Resolve a path stored in a checkpoint, including after it was moved."""
    path = Path(saved)
    if path.is_absolute() or path.exists():
        return path
    relative = anchor.resolve().parent / path
    if relative.exists():
        return relative
    sibling = anchor.resolve().parent / path.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"cannot resolve saved path {str(saved)!r}")


def checkpoint_base_metric(payload: dict) -> str | None:
    """Return the base-metric reference used by a metric checkpoint."""
    direct = payload.get("base_metric")
    if direct:
        return str(direct)
    optimizer = payload.get("optimizer")
    if isinstance(optimizer, dict) and optimizer.get("base_metric"):
        return str(optimizer["base_metric"])
    return None


def trace_free_matrix(parameters: Sequence[float], n: int) -> np.ndarray:
    """Symmetric trace-zero matrix from n(n+1)/2-1 free parameters."""
    parameters = np.asarray(parameters, dtype=np.float64)
    expected = n * (n + 1) // 2 - 1
    if parameters.shape != (expected,):
        raise ValueError(f"expected {expected} parameters, got {parameters.shape}")
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[np.arange(n - 1), np.arange(n - 1)] = parameters[: n - 1]
    matrix[n - 1, n - 1] = -float(parameters[: n - 1].sum())
    cursor = n - 1
    for row in range(n):
        for col in range(row):
            matrix[row, col] = matrix[col, row] = parameters[cursor]
            cursor += 1
    return matrix


def matrix_parameters(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("matrix must be square")
    centered = 0.5 * (matrix + matrix.T)
    centered -= np.trace(centered) / n * np.eye(n)
    result = [float(centered[index, index]) for index in range(n - 1)]
    result.extend(
        float(centered[row, col])
        for row in range(n)
        for col in range(row)
    )
    return np.asarray(result, dtype=np.float64)


def exhaustive_covering_radius(
    facets: Sequence[tuple[Sequence[float], float]],
) -> tuple[float, int]:
    """Maximum norm over every vertex of a full-dimensional Voronoi cell."""
    normals = np.asarray([facet[0] for facet in facets], dtype=np.float64)
    norms = np.linalg.norm(normals, axis=1)
    unit = normals / norms[:, None]
    offsets = np.asarray([facet[1] for facet in facets], dtype=np.float64)
    halfspaces = np.column_stack((unit, -offsets))
    n = normals.shape[1]
    try:
        intersection = HalfspaceIntersection(
            halfspaces, np.zeros(n), qhull_options="Qx"
        )
    except Exception:
        # QJ is a numerical fallback for a degenerate but valid Voronoi form.
        intersection = HalfspaceIntersection(
            halfspaces, np.zeros(n), qhull_options="QJ Qx"
        )
    vertices = np.asarray(intersection.intersections, dtype=np.float64)
    if not len(vertices) or not np.all(np.isfinite(vertices)):
        raise RuntimeError("Qhull returned no finite Voronoi vertices")
    return float(np.linalg.norm(vertices, axis=1).max()), int(len(vertices))


@dataclass
class MetricEvaluation:
    objective: float
    soft_min: float
    min_ratio: float
    min_distance: float
    diameter: float
    facet_count: int
    vertex_count: int
    subvector_count: int
    h_norm: float
    parameters: np.ndarray
    basis: np.ndarray
    witnesses: list[dict]

    def as_json(self, gram: bool = True) -> dict:
        payload = {
            "objective": self.objective,
            "soft_min": self.soft_min,
            "min_ratio": self.min_ratio,
            "min_distance": self.min_distance,
            "diameter": self.diameter,
            "facet_count": self.facet_count,
            "vertex_count": self.vertex_count,
            "subvector_count": self.subvector_count,
            "deformation_frobenius_norm": self.h_norm,
            "parameters": self.parameters.tolist(),
            "basis": self.basis.tolist(),
            "witnesses": self.witnesses,
        }
        if gram:
            payload["gram"] = (self.basis @ self.basis.T).tolist()
        return payload


class MetricEvaluator:
    def __init__(
        self,
        basis0: np.ndarray,
        kernel_columns: np.ndarray,
        *,
        softmin_temperature: float = 60.0,
        max_h_norm: float = 0.8,
    ) -> None:
        self.basis0 = np.asarray(basis0, dtype=np.float64)
        self.kernel = np.asarray(kernel_columns, dtype=np.int64)
        self.n = self.basis0.shape[0]
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
        sub_basis = self.kernel.T @ basis
        vectors = combigeo._vectors_near(
            sub_basis.tolist(),
            [0.0] * self.n,
            2.0 * diameter + 1e-8,
        )
        inverse = np.linalg.inv(basis)
        values: list[tuple[float, float, np.ndarray]] = []
        for raw_vector in vectors:
            vector = np.asarray(raw_vector, dtype=np.float64)
            if np.linalg.norm(vector) < 1e-10:
                continue
            distance = 2.0 * combigeo.dist_to_halfspaces(
                (0.5 * vector).tolist(), facets
            )
            coordinate = np.rint(vector @ inverse).astype(np.int64)
            values.append((distance / diameter, distance, coordinate))
        if values:
            ratios = np.asarray([value[0] for value in values], dtype=np.float64)
            min_ratio = float(ratios.min())
            min_distance = float(min(value[1] for value in values))
            shifted = np.exp(
                -self.temperature * (ratios - min_ratio)
            ).sum()
            soft_min = min_ratio - math.log(float(shifted)) / self.temperature
        else:
            # The length bound proves every omitted vector has ratio >= 1.
            min_ratio = 1.0
            min_distance = diameter
            soft_min = 1.0
        objective = -soft_min

        witnesses: list[dict] = []
        if with_witnesses:
            cutoff = min_ratio + float(witness_window)
            for ratio, distance, coordinate in sorted(values, key=lambda item: item[0]):
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
            objective=objective,
            soft_min=soft_min,
            min_ratio=min_ratio,
            min_distance=min_distance,
            diameter=diameter,
            facet_count=len(facets),
            vertex_count=vertex_count,
            subvector_count=max(0, len(vectors) - 1),
            h_norm=h_norm,
            parameters=parameters_array,
            basis=basis,
            witnesses=witnesses,
        )


_WORKER_EVALUATOR: MetricEvaluator | None = None


def _worker_init(
    basis0: np.ndarray,
    kernel_columns: np.ndarray,
    temperature: float,
    max_h_norm: float,
) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = MetricEvaluator(
        basis0,
        kernel_columns,
        softmin_temperature=temperature,
        max_h_norm=max_h_norm,
    )


def _worker_evaluate(parameters: Sequence[float]) -> dict:
    assert _WORKER_EVALUATOR is not None
    try:
        result = _WORKER_EVALUATOR.evaluate(parameters)
        return {
            "objective": result.objective,
            "soft_min": result.soft_min,
            "min_ratio": result.min_ratio,
            "diameter": result.diameter,
            "parameters": result.parameters.tolist(),
        }
    except Exception as error:
        return {
            "objective": 1e6,
            "soft_min": -1e6,
            "min_ratio": -1e6,
            "diameter": float("inf"),
            "parameters": list(parameters),
            "error": f"{type(error).__name__}: {error}",
        }


def campaign_records(payload: dict) -> list[dict]:
    """Return deduplicated modular candidates from every campaign format.

    The original Radon and block searches store candidates in ``results``.
    Exact threshold-frontier searches instead keep one candidate under each
    ``small_rows`` entry.  Treating both as first-class campaign sources lets
    metric optimization start from a geometrically diverse exact frontier
    without copying records into ad-hoc JSON files.
    """
    raw: list[dict] = []
    raw.extend(
        record
        for record in payload.get("results", [])
        if isinstance(record, dict)
    )
    raw.extend(
        record
        for record in payload.get("candidates", [])
        if isinstance(record, dict)
    )
    raw.extend(
        entry["candidate"]
        for entry in payload.get("small_rows", [])
        if isinstance(entry, dict) and isinstance(entry.get("candidate"), dict)
    )
    for key in (
        "candidate",
        "best_candidate",
        "best_by_minimum_ratio",
        "valid_candidate",
    ):
        record = payload.get(key)
        if isinstance(record, dict):
            raw.append(record)

    records: list[dict] = []
    seen: set[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = set()
    for record in raw:
        if record.get("killed", 0) <= 0:
            continue
        moduli = tuple(int(value) for value in record.get("moduli", []))
        rows = tuple(
            tuple(int(value) for value in row)
            for row in record.get("rows", [])
        )
        if not moduli or not rows:
            continue
        key = (moduli, rows)
        if key not in seen:
            seen.add(key)
            records.append(record)
    return records


def select_record(
    payload: dict,
    moduli: Sequence[int] | None,
    beta: float | None,
    *,
    rank: int = 0,
    rows: Sequence[Sequence[int]] | None = None,
) -> dict:
    candidates = campaign_records(payload)
    if moduli is not None:
        candidates = [
            record for record in candidates if record["moduli"] == list(moduli)
        ]
    if beta is not None:
        candidates = [
            record
            for record in candidates
            if math.isclose(float(record["beta"]), beta, abs_tol=1e-12)
        ]
    if rows is not None:
        expected_rows = [
            [int(value) for value in row]
            for row in rows
        ]
        candidates = [
            record for record in candidates
            if record.get("rows") == expected_rows
        ]
    if not candidates:
        raise ValueError("no matching non-valid campaign record")
    candidates.sort(
        key=lambda record: (
            float(record.get("minimum_conflict_ratio") or -1.0),
            -int(record.get("killed", 0)),
        ),
        reverse=True,
    )
    if not 0 <= rank < len(candidates):
        raise ValueError(
            f"record rank {rank} outside matching candidates "
            f"[0:{len(candidates)}]"
        )
    return candidates[rank]


def span_stretch_seed(
    basis0: np.ndarray, record: dict, amount: float
) -> np.ndarray:
    conflict_records = record.get("conflicts", [])
    # A weighted modular screen can leave several geometrically distinct
    # shells.  Stretching the span of *all* of them is often a no-op because
    # their union has full rank.  The max-min objective is controlled locally
    # by the deepest active shell, so use only conflicts tied with its minimum.
    # The full exhaustive evaluator below still guards against a different
    # shell becoming active after the step.
    finite_ratios = [
        float(item["distance_ratio"])
        for item in conflict_records
        if item.get("distance_ratio") is not None
        and math.isfinite(float(item["distance_ratio"]))
    ]
    if finite_ratios:
        minimum_ratio = min(finite_ratios)
        active_records = [
            item
            for item in conflict_records
            if item.get("distance_ratio") is not None
            and float(item["distance_ratio"]) <= minimum_ratio + 1e-7
        ]
    else:
        active_records = conflict_records
    conflicts = np.asarray(
        [item["coordinate"] for item in active_records],
        dtype=np.float64,
    )
    if not len(conflicts) or amount == 0:
        n = basis0.shape[0]
        return np.zeros(n * (n + 1) // 2 - 1, dtype=np.float64)
    physical = conflicts @ basis0
    _, _, right = np.linalg.svd(physical, full_matrices=True)
    rank = int(np.linalg.matrix_rank(physical))
    if rank == 0 or rank == basis0.shape[0]:
        return np.zeros(
            basis0.shape[0] * (basis0.shape[0] + 1) // 2 - 1,
            dtype=np.float64,
        )
    projector = right[:rank].T @ right[:rank]
    complement = np.eye(basis0.shape[0]) - projector
    generator = (
        amount / rank * projector
        - amount / (basis0.shape[0] - rank) * complement
    )
    return matrix_parameters(generator)


def checkpoint_payload(
    source: Path,
    record: dict,
    kernel: np.ndarray,
    evaluation: MetricEvaluation,
    *,
    generation: int,
    evaluations: int,
    elapsed: float,
    optimizer: dict,
) -> dict:
    return {
        "method": "fixed-kernel exhaustive-Voronoi metric deformation",
        "source_campaign": str(source),
        # Keep this at top level as well as in ``optimizer`` so that every
        # downstream verifier can reconstruct the parameterization without
        # knowing which optimizer produced the checkpoint.
        "base_metric": optimizer.get("base_metric"),
        "source_record": {
            "moduli": record["moduli"],
            "rows": record["rows"],
            "image_index": record["image_index"],
            "beta": record.get("beta"),
            "source_minimum_conflict_ratio": record.get(
                "minimum_conflict_ratio"
            ),
        },
        "kernel_basis_columns": kernel.astype(int).tolist(),
        "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
        "kernel_smith": smith_diagonal(kernel),
        "generation": generation,
        "evaluations": evaluations,
        "elapsed_seconds": round(elapsed, 6),
        "optimizer": optimizer,
        "best": evaluation.as_json(),
        "valid_numerical_witness": evaluation.min_ratio >= 1.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--moduli", type=parse_moduli)
    parser.add_argument("--beta", type=float)
    parser.add_argument(
        "--record-rank",
        type=int,
        default=0,
        help=(
            "rank by decreasing fixed-metric minimum ratio among matching "
            "modular/threshold campaign records"
        ),
    )
    parser.add_argument(
        "--candidate-rank",
        type=int,
        default=0,
        help=(
            "rank in the best[] list of a determinant_repair.py campaign; "
            "ignored for modular campaign JSON"
        ),
    )
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=18)
    parser.add_argument("--sigma", type=float, default=0.035)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--initial-stretch", type=float, default=0.14)
    parser.add_argument("--temperature", type=float, default=60.0)
    parser.add_argument("--max-h-norm", type=float, default=0.8)
    parser.add_argument("--target-margin", type=float, default=2e-4)
    parser.add_argument(
        "--base-metric",
        type=Path,
        help=(
            "optional metric JSON whose best.basis is used as the undeformed "
            "parent form; useful when a modular campaign was run on an "
            "already-deformed lattice"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "optional earlier metric_deform JSON; restart CMA-ES from its "
            "best deformation parameters"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "metric_deform_best.json",
    )
    args = parser.parse_args(argv)
    if args.record_rank < 0:
        parser.error("--record-rank must be nonnegative")

    resume_payload: dict | None = None
    if args.resume is not None:
        try:
            resume_payload = json.loads(args.resume.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"invalid --resume JSON: {error}")
        inherited_base = checkpoint_base_metric(resume_payload)
        if inherited_base is not None:
            try:
                inherited_path = resolve_saved_path(args.resume, inherited_base)
            except FileNotFoundError as error:
                parser.error(str(error))
            if args.base_metric is None:
                args.base_metric = inherited_path

    source_payload = json.loads(args.campaign.read_text())
    lattice_name = source_payload.get("lattice")
    if not lattice_name:
        parser.error("campaign JSON has no lattice name")
    repair_candidates = source_payload.get("best")
    if (
        isinstance(repair_candidates, list)
        and source_payload.get("target_index") is not None
    ):
        if not 0 <= args.candidate_rank < len(repair_candidates):
            parser.error(
                f"candidate rank {args.candidate_rank} outside "
                f"best[0:{len(repair_candidates)}]"
            )
        candidate = repair_candidates[args.candidate_rank]
        kernel = np.asarray(
            candidate["kernel_basis_columns"], dtype=np.int64
        )
        n = len(kernel)
        record = {
            "moduli": [
                int(value)
                for value in candidate.get("smith", [])
                if int(value) > 1
            ],
            "rows": [],
            "image_index": int(source_payload["target_index"]),
            "killed": int(candidate.get("conflict_count_with_sign", 1)),
            "beta": float(args.candidate_rank),
            "minimum_conflict_ratio": candidate.get("distance_ratio"),
            "conflicts": candidate.get("conflicts", []),
        }
    else:
        record = select_record(
            source_payload,
            args.moduli,
            args.beta,
            rank=args.record_rank,
        )
        rows = [np.asarray(row, dtype=np.int64) for row in record["rows"]]
        n = int(source_payload.get("n", len(rows[0])))
        kernel = hnf_columns(kernel_basis(rows, record["moduli"], n))
    if args.base_metric is not None:
        base_payload = json.loads(args.base_metric.read_text())
        try:
            basis0 = np.asarray(
                base_payload["best"]["basis"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid --base-metric JSON: {error}")
        if basis0.shape != (n, n) or not np.all(np.isfinite(basis0)):
            parser.error(
                f"--base-metric basis has shape {basis0.shape}, "
                f"expected {(n, n)} with finite entries"
            )
    elif lattice_name == "E7*-ABPR":
        basis0 = np.linalg.cholesky(M_E7().T @ M_E7())
    else:
        basis0, _, _ = load_forbidden(lattice_name)
    evaluator = MetricEvaluator(
        basis0,
        kernel,
        softmin_temperature=args.temperature,
        max_h_norm=args.max_h_norm,
    )
    zero = np.zeros(n * (n + 1) // 2 - 1, dtype=np.float64)
    baseline = evaluator.evaluate(zero, with_witnesses=True)
    if resume_payload is not None:
        resume_kernel = np.asarray(
            resume_payload.get("kernel_basis_columns"), dtype=np.int64
        )
        if resume_kernel.shape != kernel.shape or not np.array_equal(
            resume_kernel, kernel
        ):
            parser.error(
                "--resume checkpoint belongs to a different coloring kernel"
            )
        initial = np.asarray(
            resume_payload["best"]["parameters"], dtype=np.float64
        )
        expected = n * (n + 1) // 2 - 1
        if initial.shape != (expected,):
            parser.error(
                f"resume parameters have shape {initial.shape}, expected {(expected,)}"
            )
    else:
        initial = span_stretch_seed(basis0, record, args.initial_stretch)
    seeded = evaluator.evaluate(initial, with_witnesses=True)
    if resume_payload is not None:
        try:
            recorded_resume_ratio = float(
                resume_payload["best"]["min_ratio"]
            )
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid --resume best.min_ratio: {error}")
        consistency_tolerance = max(
            5e-8, 5e-7 * abs(recorded_resume_ratio)
        )
        if (
            abs(seeded.min_ratio - recorded_resume_ratio)
            > consistency_tolerance
        ):
            parser.error(
                "--resume metric parameterization mismatch: recomputed "
                f"{seeded.min_ratio:.12g} != recorded "
                f"{recorded_resume_ratio:.12g}; check the base-metric chain"
            )
    print(
        f"record: moduli={record['moduli']} index={record['image_index']} "
        f"killed={record['killed']} source-ratio="
        f"{record.get('minimum_conflict_ratio')}",
        flush=True,
    )
    print(
        f"baseline: min={baseline.min_ratio:.9f} soft={baseline.soft_min:.9f} "
        f"diam={baseline.diameter:.9f} facets={baseline.facet_count} "
        f"vertices={baseline.vertex_count}",
        flush=True,
    )
    seed_label = (
        f"resume={args.resume}"
        if args.resume is not None
        else f"stretch={args.initial_stretch:g}"
    )
    print(
        f"seed {seed_label}: min={seeded.min_ratio:.9f} "
        f"soft={seeded.soft_min:.9f} diam={seeded.diameter:.9f} "
        f"|H|={seeded.h_norm:.5f}",
        flush=True,
    )

    options = {
        "popsize": args.population,
        "maxiter": args.generations,
        "seed": args.seed,
        "verbose": -9,
        "tolfun": 1e-11,
        "tolx": 1e-9,
    }
    strategy = cma.CMAEvolutionStrategy(
        initial.tolist(), args.sigma, options
    )
    best = seeded if seeded.min_ratio >= baseline.min_ratio else baseline
    best_soft = seeded if seeded.soft_min >= baseline.soft_min else baseline
    total_evaluations = 0
    start = time.perf_counter()
    optimizer_info = {
        "generations_budget": args.generations,
        "population": args.population,
        "sigma": args.sigma,
        "workers": args.workers,
        "seed": args.seed,
        "initial_stretch": args.initial_stretch,
        "record_rank": args.record_rank,
        "resume": str(args.resume) if args.resume is not None else None,
        "base_metric": (
            str(args.base_metric) if args.base_metric is not None else None
        ),
        "temperature": args.temperature,
        "max_h_norm": args.max_h_norm,
    }

    import multiprocessing as mp

    with mp.Pool(
        max(1, args.workers),
        initializer=_worker_init,
        initargs=(
            basis0,
            kernel,
            args.temperature,
            args.max_h_norm,
        ),
    ) as pool:
        generation = 0
        while not strategy.stop() and generation < args.generations:
            generation += 1
            population = strategy.ask()
            summaries = pool.map(_worker_evaluate, population)
            objectives = [summary["objective"] for summary in summaries]
            strategy.tell(population, objectives)
            total_evaluations += len(population)

            soft_index = int(np.argmin(objectives))
            if summaries[soft_index]["soft_min"] > best_soft.soft_min + 1e-12:
                best_soft = evaluator.evaluate(
                    summaries[soft_index]["parameters"], with_witnesses=True
                )
            ratio_index = int(
                np.argmax([summary["min_ratio"] for summary in summaries])
            )
            if summaries[ratio_index]["min_ratio"] > best.min_ratio + 1e-12:
                best = evaluator.evaluate(
                    summaries[ratio_index]["parameters"], with_witnesses=True
                )
                payload = checkpoint_payload(
                    args.campaign,
                    record,
                    kernel,
                    best,
                    generation=generation,
                    evaluations=total_evaluations,
                    elapsed=time.perf_counter() - start,
                    optimizer=optimizer_info,
                )
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    f"  new hard best gen={generation}: "
                    f"min={best.min_ratio:.9f} soft={best.soft_min:.9f} "
                    f"diam={best.diameter:.9f} |H|={best.h_norm:.5f}",
                    flush=True,
                )
            print(
                f"gen {generation:3d}/{args.generations}: "
                f"generation-min="
                f"{max(summary['min_ratio'] for summary in summaries):.9f} "
                f"best-min={best.min_ratio:.9f} "
                f"best-soft={best_soft.soft_min:.9f} "
                f"elapsed={time.perf_counter()-start:.1f}s",
                flush=True,
            )
            if best.min_ratio >= 1.0 + args.target_margin:
                print("*** numerical separation target reached ***", flush=True)
                break

    # Always leave a complete final checkpoint, even if the initial point won.
    final = evaluator.evaluate(best.parameters, with_witnesses=True)
    payload = checkpoint_payload(
        args.campaign,
        record,
        kernel,
        final,
        generation=generation,
        evaluations=total_evaluations,
        elapsed=time.perf_counter() - start,
        optimizer=optimizer_info,
    )
    payload["best_soft_min_seen"] = best_soft.soft_min
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL min={final.min_ratio:.12f} D={final.min_distance:.12f} "
        f"diam={final.diameter:.12f} valid={final.min_ratio >= 1.0} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
