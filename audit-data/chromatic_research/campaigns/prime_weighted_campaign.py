"""Geometry-weighted prime/Radon campaigns.

The ordinary objective counts every forbidden vector equally.  That is a poor
surrogate when the best algebraic coloring will subsequently be paired with a
small deformation of the parent lattice: a vector whose translated Voronoi
cells almost satisfy the diameter constraint is much easier to remove than a
Voronoi-relevant vector at distance zero.

For every forbidden vector v this campaign computes

    rho(v) = dist(V, v + V) / diam(V)

and uses the smooth loss

    weight_beta(v) = exp(beta * (1 - rho(v))).

beta=0 recovers the ordinary killed-vector count.  Larger beta makes severe
geometric conflicts dominate while retaining a positive cost for every
conflict.  All final kernels, image indices, and killed sets are checked with
exact integer modular arithmetic.
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
from chromatic_research.campaigns.prime_campaign import parse_structures
from chromatic_research.core.prime_radon import PrimarySearch, killed_mask, load_forbidden
from chromatic_research.paths import results_path


HERE = Path(__file__).resolve().parent


def parse_betas(text: str) -> list[float]:
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise argparse.ArgumentTypeError("betas must be a non-empty JSON list")
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise argparse.ArgumentTypeError("betas must be finite and nonnegative")
    return result


def geometry(name: str) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    basis, forbidden, diameter = load_forbidden(name)
    facets = combigeo.relevant_facets(basis.tolist())
    ratios = np.empty(len(forbidden), dtype=np.float64)
    start = time.perf_counter()
    for index, coordinate in enumerate(forbidden):
        vector = coordinate @ basis
        distance = 2.0 * combigeo.dist_to_halfspaces(
            (0.5 * vector).tolist(), facets
        )
        ratios[index] = distance / diameter
    if np.any(ratios >= 1.0 + 1e-8):
        raise AssertionError("forbidden set contains a non-forbidden vector")
    print(
        f"geometry: {name} n={basis.shape[0]} |F|={len(forbidden)} "
        f"diam={diameter:.12g} score_time={time.perf_counter()-start:.2f}s",
        flush=True,
    )
    return basis, forbidden, diameter, ratios


def screen_seeds(path: Path | None) -> dict[tuple[int, ...], list[list[int]]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        tuple(int(value) for value in record["moduli"]): record["rows"]
        for record in payload.get("results", [])
    }


def conflict_details(
    basis: np.ndarray,
    forbidden: np.ndarray,
    ratios: np.ndarray,
    rows: Sequence[np.ndarray],
    moduli: Sequence[int],
) -> list[dict]:
    mask = killed_mask(forbidden, rows, moduli)
    details = []
    for coordinate, ratio in zip(forbidden[mask], ratios[mask]):
        vector = coordinate @ basis
        details.append(
            {
                "coordinate": coordinate.astype(int).tolist(),
                "distance_ratio": round(float(ratio), 12),
                "norm_squared": round(float(vector @ vector), 12),
            }
        )
    details.sort(key=lambda item: item["distance_ratio"])
    return details


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice")
    parser.add_argument("structures", type=parse_structures)
    parser.add_argument("--betas", type=parse_betas, default=[0.0, 2.0, 4.0, 6.0])
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--sweeps", type=int, default=24)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--seed-screen",
        type=Path,
        default=results_path("prime_screen_e7.json"),
        help="optional ordinary-search JSON whose rows seed matching structures",
    )
    args = parser.parse_args(argv)

    basis, forbidden, diameter, ratios = geometry(args.lattice)
    seeds = screen_seeds(args.seed_screen)
    payload = {
        "method": "geometry-weighted prime/Radon block descent",
        "lattice": args.lattice,
        "n": int(basis.shape[0]),
        "n_forbidden": int(len(forbidden)),
        "diameter": float(diameter),
        "structures": args.structures,
        "betas": args.betas,
        "budget": {
            "restarts": args.restarts,
            "sweeps": args.sweeps,
            "top": args.top,
            "seed": args.seed,
        },
        "results": [],
    }
    campaign_start = time.perf_counter()
    task_number = 0
    task_total = len(args.structures) * len(args.betas)
    for structure_number, moduli in enumerate(args.structures):
        initial = seeds.get(tuple(moduli))
        for beta_number, beta in enumerate(args.betas):
            task_number += 1
            weights = np.exp(beta * (1.0 - ratios))
            print(
                f"\n[{task_number}/{task_total}] moduli={moduli} "
                f"product={math.prod(moduli)} beta={beta:g}",
                flush=True,
            )
            search = PrimarySearch(
                forbidden,
                moduli,
                seed=args.seed + 1009 * structure_number + 104729 * beta_number,
            )
            result = search.run_weighted(
                weights,
                restarts=args.restarts,
                max_sweeps=args.sweeps,
                top=args.top,
                progress_every=max(1, args.restarts // 4),
                initial_rows=initial,
            )
            details = conflict_details(
                basis, forbidden, ratios, result.rows, result.moduli
            )
            record = {
                "beta": beta,
                "target_product": math.prod(moduli),
                **result.as_json(),
                "minimum_conflict_ratio": (
                    min(item["distance_ratio"] for item in details)
                    if details
                    else None
                ),
                "conflicts": details,
            }
            payload["results"].append(record)
            if args.output:
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                f"[{task_number}/{task_total}] loss={result.weighted_loss:.9g} "
                f"killed={result.killed} image={result.image_index} "
                f"min-ratio={record['minimum_conflict_ratio']}",
                flush=True,
            )
            if result.found and result.image_index < 1372:
                print(
                    f"*** FIXED-LATTICE IMPROVEMENT: index={result.image_index} ***",
                    flush=True,
                )
    payload["campaign_seconds"] = round(
        time.perf_counter() - campaign_start, 6
    )
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
