"""Order-statistic metric optimization for a portfolio of coloring kernels.

A usual fixed-kernel deformation maximizes one function ``m_C(Q)`` and can
become trapped far from every discrete transition.  Given several exact
integer kernels, this script instead maximizes the r-th largest value among
them.  In particular, rank two maximizes the *second-best* kernel and seeks a
bridge metric on which at least two discrete basins are simultaneously good.

The complete Voronoi and sublattice-vector oracle from ``metric_deform.py`` is
used for every kernel and every trial form.  Results remain numerical until a
single winning kernel is rationalized by ``verify_metric_candidate.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from chromatic_research.core.metric_deform import (
    MetricEvaluation,
    MetricEvaluator,
    checkpoint_base_metric,
    resolve_saved_path,
)
from chromatic_research.core.prime_radon import load_forbidden, smith_diagonal
from chromatic_research.core.prime_row_opt import _source_lattice

import cma


_WORKER_EVALUATORS: list[MetricEvaluator] = []
_WORKER_RANK = 1


def _worker_init(
    basis0: np.ndarray,
    kernels: Sequence[np.ndarray],
    temperature: float,
    max_h_norm: float,
    rank: int,
) -> None:
    global _WORKER_EVALUATORS, _WORKER_RANK
    _WORKER_EVALUATORS = [
        MetricEvaluator(
            basis0,
            kernel,
            softmin_temperature=temperature,
            max_h_norm=max_h_norm,
        )
        for kernel in kernels
    ]
    _WORKER_RANK = int(rank)


def _ranked(values: Sequence[float], rank: int) -> float:
    if not 1 <= rank <= len(values):
        raise ValueError(f"rank {rank} outside [1,{len(values)}]")
    return float(sorted((float(value) for value in values), reverse=True)[rank - 1])


def _worker_evaluate(parameters: Sequence[float]) -> dict:
    try:
        evaluations = [
            evaluator.evaluate(parameters)
            for evaluator in _WORKER_EVALUATORS
        ]
        soft = [evaluation.soft_min for evaluation in evaluations]
        hard = [evaluation.min_ratio for evaluation in evaluations]
        return {
            "objective": -_ranked(soft, _WORKER_RANK),
            "soft_rank": _ranked(soft, _WORKER_RANK),
            "hard_rank": _ranked(hard, _WORKER_RANK),
            "ratios": hard,
            "parameters": list(parameters),
        }
    except Exception as error:
        return {
            "objective": 1e6,
            "soft_rank": -1e6,
            "hard_rank": -1e6,
            "ratios": [],
            "parameters": list(parameters),
            "error": f"{type(error).__name__}: {error}",
        }


def _full_evaluations(
    evaluators: Sequence[MetricEvaluator], parameters: Sequence[float]
) -> list[MetricEvaluation]:
    return [
        evaluator.evaluate(parameters, with_witnesses=True)
        for evaluator in evaluators
    ]


def _optimization_frame(
    *,
    lattice: str,
    n: int,
    source_payload: dict,
    base_metric: Path | None,
    resume: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a basis and compatible deformation coordinates.

    Historical portfolio campaigns used the undeformed parent lattice and
    inherited the parameter vector from the first source metric.  That is
    correct only when the source metric is itself parameterized over the
    undeformed parent.  Alternating discrete/continuous searches instead
    create chains of local deformations.  ``base_metric`` starts a fresh,
    zero-centered chart at any saved basis; a resumed checkpoint must declare
    the same chart explicitly.
    """

    expected = n * (n + 1) // 2 - 1
    if base_metric is None:
        basis0, _, _ = load_forbidden(lattice)
        payload = json.loads(resume.read_text()) if resume else source_payload
        initial = np.asarray(payload["best"]["parameters"], dtype=np.float64)
    else:
        base_payload = json.loads(base_metric.read_text())
        basis0 = np.asarray(base_payload["best"]["basis"], dtype=np.float64)
        if resume is None:
            initial = np.zeros(expected, dtype=np.float64)
        else:
            payload = json.loads(resume.read_text())
            saved_base = checkpoint_base_metric(payload)
            if saved_base is None:
                raise ValueError(
                    "a locally parameterized resume checkpoint has no "
                    "base_metric declaration"
                )
            resolved = resolve_saved_path(resume, saved_base)
            if resolved.resolve() != base_metric.resolve():
                raise ValueError(
                    "resume checkpoint uses a different local base metric: "
                    f"{resolved} != {base_metric}"
                )
            initial = np.asarray(payload["best"]["parameters"], dtype=np.float64)
    if basis0.shape != (n, n) or not np.all(np.isfinite(basis0)):
        raise ValueError(
            f"optimization basis has shape {basis0.shape}, expected {(n, n)}"
        )
    if initial.shape != (expected,) or not np.all(np.isfinite(initial)):
        raise ValueError(
            f"initial parameters have shape {initial.shape}, "
            f"expected {(expected,)}"
        )
    return basis0, initial


def _checkpoint(
    *,
    lattice: str,
    metric_paths: Sequence[Path],
    base_metric: Path | None,
    kernels: Sequence[np.ndarray],
    rank: int,
    evaluations: Sequence[MetricEvaluation],
    generation: int,
    evaluation_count: int,
    elapsed: float,
    settings: dict,
) -> dict:
    hard = [evaluation.min_ratio for evaluation in evaluations]
    soft = [evaluation.soft_min for evaluation in evaluations]
    parent = evaluations[0].as_json()
    parent["portfolio_ratios"] = hard
    parent["portfolio_soft_minima"] = soft
    parent["portfolio_hard_rank"] = _ranked(hard, rank)
    parent["portfolio_soft_rank"] = _ranked(soft, rank)
    return {
        "method": "CMA-ES order-statistic bridge optimization over exact kernels",
        "lattice": lattice,
        "source_metrics": [str(path) for path in metric_paths],
        "base_metric": str(base_metric) if base_metric is not None else None,
        "rank": rank,
        "kernel_basis_columns": [
            kernel.astype(int).tolist() for kernel in kernels
        ],
        "kernel_determinants": [
            abs(int(round(np.linalg.det(kernel)))) for kernel in kernels
        ],
        "kernel_smith": [smith_diagonal(kernel) for kernel in kernels],
        "generation": generation,
        "evaluations": evaluation_count,
        "elapsed_seconds": elapsed,
        "optimizer": settings,
        "best": parent,
        "per_kernel": [
            evaluation.as_json() for evaluation in evaluations
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help=(
            "descending order statistic to maximize; zero means the worst "
            "kernel, i.e. rank equal to the portfolio size"
        ),
    )
    parser.add_argument(
        "--base-metric",
        type=Path,
        help=(
            "start a zero-centered local deformation chart at this saved "
            "metric; required for chained alternating-continuation metrics"
        ),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--sigma", type=float, default=0.04)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=30000.0)
    parser.add_argument("--max-h-norm", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--target", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if len(args.metrics) < 2:
        parser.error("at least two metric files are required")
    if args.generations < 1 or args.population < 2 or args.workers < 1:
        parser.error("generation/population/worker budgets must be positive")
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be finite and positive")

    payloads = [json.loads(path.read_text()) for path in args.metrics]
    lattice = _source_lattice(args.metrics[0], payloads[0])
    for path, payload in zip(args.metrics[1:], payloads[1:]):
        if _source_lattice(path, payload) != lattice:
            parser.error("all metrics must use the same parent lattice")
    kernels = [
        np.asarray(payload["kernel_basis_columns"], dtype=np.int64)
        for payload in payloads
    ]
    n = len(kernels[0])
    if any(kernel.shape != (n, n) for kernel in kernels):
        parser.error("all kernels must be square and have the same dimension")
    rank = len(kernels) if args.rank == 0 else args.rank
    if not 1 <= rank <= len(kernels):
        parser.error("--rank must lie between one and the portfolio size")

    try:
        basis0, initial = _optimization_frame(
            lattice=lattice,
            n=n,
            source_payload=payloads[0],
            base_metric=args.base_metric,
            resume=args.resume,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    evaluators = [
        MetricEvaluator(
            basis0,
            kernel,
            softmin_temperature=args.temperature,
            max_h_norm=args.max_h_norm,
        )
        for kernel in kernels
    ]
    expected = n * (n + 1) // 2 - 1

    settings = {
        "generations_budget": args.generations,
        "population": args.population,
        "sigma": args.sigma,
        "workers": args.workers,
        "temperature": args.temperature,
        "max_h_norm": args.max_h_norm,
        "seed": args.seed,
        "base_metric": (
            str(args.base_metric) if args.base_metric is not None else None
        ),
        "resume": str(args.resume) if args.resume else None,
        "target": args.target,
    }
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
    initial_full = _full_evaluations(evaluators, initial)
    best_parameters = initial.copy()
    best_hard = _ranked(
        [evaluation.min_ratio for evaluation in initial_full], rank
    )
    best_soft = _ranked(
        [evaluation.soft_min for evaluation in initial_full], rank
    )
    started = time.perf_counter()
    evaluation_count = 0
    generation = 0
    print(
        f"start rank={rank}/{len(kernels)} hard={best_hard:.12f} "
        f"soft={best_soft:.12f} ratios="
        f"{[round(e.min_ratio, 9) for e in initial_full]}",
        flush=True,
    )

    with mp.Pool(
        max(1, args.workers),
        initializer=_worker_init,
        initargs=(basis0, kernels, args.temperature, args.max_h_norm, rank),
    ) as pool:
        while not strategy.stop() and generation < args.generations:
            generation += 1
            population = strategy.ask()
            summaries = pool.map(_worker_evaluate, population)
            objectives = [summary["objective"] for summary in summaries]
            strategy.tell(population, objectives)
            evaluation_count += len(population)
            hard_index = int(
                np.argmax([summary["hard_rank"] for summary in summaries])
            )
            candidate = summaries[hard_index]
            if candidate["hard_rank"] > best_hard + 1e-12:
                best_hard = float(candidate["hard_rank"])
                best_parameters = np.asarray(
                    candidate["parameters"], dtype=np.float64
                )
                full = _full_evaluations(evaluators, best_parameters)
                best_soft = _ranked(
                    [evaluation.soft_min for evaluation in full], rank
                )
                args.output.write_text(
                    json.dumps(
                        _checkpoint(
                            lattice=lattice,
                            metric_paths=args.metrics,
                            base_metric=args.base_metric,
                            kernels=kernels,
                            rank=rank,
                            evaluations=full,
                            generation=generation,
                            evaluation_count=evaluation_count,
                            elapsed=time.perf_counter() - started,
                            settings=settings,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    f"  new bridge gen={generation}: hard={best_hard:.9f} "
                    f"soft={best_soft:.9f} ratios="
                    f"{[round(e.min_ratio, 9) for e in full]}",
                    flush=True,
                )
            if generation == 1 or generation % 10 == 0:
                print(
                    f"gen {generation}/{args.generations}: "
                    f"best={best_hard:.9f} elapsed="
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if best_hard >= args.target:
                print("*** portfolio target reached ***", flush=True)
                break

    final = _full_evaluations(evaluators, best_parameters)
    payload = _checkpoint(
        lattice=lattice,
        metric_paths=args.metrics,
        base_metric=args.base_metric,
        kernels=kernels,
        rank=rank,
        evaluations=final,
        generation=generation,
        evaluation_count=evaluation_count,
        elapsed=time.perf_counter() - started,
        settings=settings,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL bridge={payload['best']['portfolio_hard_rank']:.12f} "
        f"ratios={payload['best']['portfolio_ratios']} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
