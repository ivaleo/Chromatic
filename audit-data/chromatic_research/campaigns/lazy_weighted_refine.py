"""Geometry-dual refinement of a lazy prime/Radon campaign.

Binary cutting planes make the discrete kernel jump to the first solution of
the current finite core.  This script instead assigns each discovered
forbidden vector a geometric loss

    exp(beta * (1 - dist(V, v+V)/diam))

and applies a multiplicative dual penalty whenever the current kernel keeps
annihilating that vector.  A complete kernel separation oracle supplies new
columns lazily.  Thus persistent/deep conflicts dominate, while cheap
near-boundary conflicts may be left for a later metric deformation.
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
from chromatic_research.core.lazy_prime_campaign import (
    canonical_rows,
    initial_short_core,
    merge_core,
    parent_geometry,
    separate_kernel,
)
from chromatic_research.core.prime_radon import (
    PrimarySearch,
    hnf_columns,
    kernel_basis,
    killed_mask,
    smith_diagonal,
)


def parse_schedule(text: str) -> list[float]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not raw:
        raise argparse.ArgumentTypeError("schedule must be a nonempty list")
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise argparse.ArgumentTypeError("schedule values must be nonnegative")
    return values


def reconstruct_core(
    source: dict,
    basis: np.ndarray,
    diameter: float,
) -> np.ndarray:
    core = initial_short_core(basis, diameter)
    histories = source["results"][0].get("history", [])
    additions: list[np.ndarray] = []
    for record in histories:
        for conflict in record.get("separation", {}).get("conflicts", []):
            additions.append(
                np.asarray(conflict["coordinate"], dtype=np.int64)
            )
    if additions:
        core = merge_core(core, np.asarray(additions, dtype=np.int64))
    return core


def geometry_ratios(
    core: np.ndarray,
    basis: np.ndarray,
    diameter: float,
    facets: Sequence[tuple[Sequence[float], float]],
) -> np.ndarray:
    ratios = np.empty(len(core), dtype=np.float64)
    for index, coordinate in enumerate(core):
        vector = coordinate @ basis
        ratios[index] = (
            2.0
            * combigeo.dist_to_halfspaces(
                (0.5 * vector).tolist(), facets
            )
            / diameter
        )
    if np.any(ratios >= 1.0 + 1e-8):
        raise AssertionError("lazy core contains a non-forbidden vector")
    return ratios


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--schedule",
        type=parse_schedule,
        default=[4.0, 8.0, 12.0, 16.0],
    )
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--sweeps", type=int, default=14)
    parser.add_argument("--top", type=int, default=14)
    parser.add_argument("--dual-rate", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=9916875)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text())
    lattice = source["lattice"]
    if len(source.get("results", [])) != 1:
        parser.error("source must contain exactly one lazy structure")
    source_record = source["results"][0]
    moduli = [int(value) for value in source_record["moduli"]]
    basis, diameter, facets = parent_geometry(lattice)
    core = reconstruct_core(source, basis, diameter)
    ratios = geometry_ratios(core, basis, diameter, facets)
    dual = np.ones(len(core), dtype=np.float64)
    initial = source_record.get("best", {})
    rows = (
        [
            np.asarray(row, dtype=np.int64)
            for row in initial.get("search", {}).get("rows", [])
        ]
        or None
    )
    print(
        f"{lattice}: reconstructed-core={len(core)} "
        f"ratio-range=[{ratios.min():.6f},{ratios.max():.6f}] "
        f"moduli={moduli}",
        flush=True,
    )

    payload: dict = {
        "method": (
            "lazy geometry-weighted prime/Radon refinement with "
            "multiplicative dual conflict penalties"
        ),
        "source": str(args.source),
        "lattice": lattice,
        "n": len(basis),
        "moduli": moduli,
        "target_product": math.prod(moduli),
        "initial_core_size": len(core),
        "schedule": args.schedule,
        "dual_rate": args.dual_rate,
        "budget": {
            "restarts_per_epoch": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "seed": args.seed,
        },
        "epochs": [],
        "best": None,
    }
    best_ratio = -1.0
    start = time.perf_counter()

    for epoch, beta in enumerate(args.schedule, start=1):
        epoch_start = time.perf_counter()
        base = np.exp(beta * (1.0 - ratios))
        weights = base * dual
        weights /= max(1.0, float(np.median(weights)))
        search = PrimarySearch(
            core,
            moduli,
            seed=args.seed + 104729 * epoch,
        )
        result = search.run_weighted(
            weights,
            restarts=args.restarts,
            max_sweeps=args.sweeps,
            top=args.top,
            progress_every=max(1, args.restarts // 2),
            initial_rows=rows,
        )
        rows = [row.copy() for row in result.rows]
        kernel = hnf_columns(kernel_basis(rows, moduli, len(basis)))
        separation = separate_kernel(
            basis, diameter, facets, kernel
        )
        true_ratio = float(separation["minimum_distance_ratio"])
        core_killed = killed_mask(core, rows, moduli)
        record = {
            "epoch": epoch,
            "beta": beta,
            "core_size": len(core),
            "core_weighted_loss": result.weighted_loss,
            "core_killed": int(core_killed.sum()),
            "rows": [row.astype(int).tolist() for row in rows],
            "image_index": result.image_index,
            "kernel_basis_columns": kernel.astype(int).tolist(),
            "kernel_determinant": abs(int(round(np.linalg.det(kernel)))),
            "kernel_smith": smith_diagonal(kernel),
            "separation": {
                key: value
                for key, value in separation.items()
                if key != "conflict_coordinates"
            },
            "seconds": time.perf_counter() - epoch_start,
        }
        payload["epochs"].append(record)
        if true_ratio > best_ratio:
            best_ratio = true_ratio
            payload["best"] = record
        print(
            f"epoch {epoch}/{len(args.schedule)} beta={beta:g}: "
            f"core-killed={record['core_killed']} "
            f"oracle-conflicts={separation['conflict_count_with_sign']} "
            f"true-min={true_ratio:.12f} "
            f"time={record['seconds']:.1f}s",
            flush=True,
        )
        if separation["valid"]:
            payload["status"] = "certified_valid"
            break

        # Repeatedly annihilated vectors receive increasing dual pressure.
        dual[core_killed] *= np.exp(
            args.dual_rate * (1.0 - ratios[core_killed])
        )
        dual = np.minimum(dual, 1e12)

        additions = separation["conflict_coordinates"]
        previous = {tuple(row.tolist()) for row in core}
        core = merge_core(core, additions)
        if len(core) > len(dual):
            new_rows = np.asarray(
                [row for row in core if tuple(row.tolist()) not in previous],
                dtype=np.int64,
            )
            new_ratios = geometry_ratios(
                new_rows, basis, diameter, facets
            )
            ratios = np.concatenate((ratios, new_ratios))
            dual = np.concatenate(
                (dual, np.ones(len(new_rows), dtype=np.float64))
            )
            # merge_core preserves old insertion order; assert alignment.
            if len(ratios) != len(core):
                raise AssertionError("core/weight alignment was lost")
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        payload["status"] = "budget_exhausted"

    payload["final_core_size"] = len(core)
    payload["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"FINAL status={payload['status']} best={best_ratio:.12f} "
        f"saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
