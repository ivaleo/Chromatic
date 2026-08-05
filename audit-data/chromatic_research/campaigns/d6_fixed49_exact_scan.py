"""Exhaust the source-preserving index-294 continuation family in E6*.

The nonmonotonic alternative to the index-336 search is

    294 = 7^2 * 2 * 3.

It keeps two of the three exact modulo-7 source characters instead of one.
For each of the three rank-two source subspaces, every nonzero modulo-2 row
(63 projective choices) and every nonzero modulo-3 row (364 projective
choices) is enumerated.  This is a complete scan of the family with those two
source characters fixed, not a heuristic restart campaign.

Two exact objectives are retained:

* minimum number of fixed-metric conflicts;
* maximum worst conflict distance, with conflict count and weighted loss as
  tie breakers.

Every selected kernel is checked for exact image size 294 and by the complete
geometric separation oracle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import combigeo
from chromatic_research.core.block_row_metric_opt import candidate_record
from chromatic_research.campaigns.d6_fixed7_campaign import DEFAULT_SOURCE, load_e6_source_rows
from chromatic_research.core.prime_radon import load_forbidden, projective_forms
from chromatic_research.core.prime_row_opt import _forbidden_with_weights
from chromatic_research.paths import results_path


HERE = Path(__file__).resolve().parent


def best_rows_for_objectives(
    forbidden: np.ndarray,
    ratios: np.ndarray,
    weights: np.ndarray,
) -> dict[str, dict]:
    """Enumerate every modulo-2/modulo-3 row pair exactly."""
    forbidden = np.asarray(forbidden, dtype=np.int64)
    ratios = np.asarray(ratios, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if ratios.shape != (len(forbidden),) or weights.shape != (
        len(forbidden),
    ):
        raise ValueError("ratio/weight arrays have incompatible shape")
    forms2 = projective_forms(forbidden.shape[1], 2)
    forms3 = projective_forms(forbidden.shape[1], 3)
    best_count_key: tuple | None = None
    best_count_rows: tuple[np.ndarray, np.ndarray] | None = None
    best_ratio_key: tuple | None = None
    best_ratio_rows: tuple[np.ndarray, np.ndarray] | None = None

    for row2 in forms2:
        active = (forbidden @ row2) % 2 == 0
        active_forbidden = forbidden[active]
        active_ratios = ratios[active]
        active_weights = weights[active]
        zero = (active_forbidden @ forms3.T) % 3 == 0
        counts = zero.sum(axis=0).astype(np.int64)
        losses = active_weights @ zero
        minima = np.full(len(forms3), np.inf, dtype=np.float64)
        for row3_index in np.flatnonzero(counts):
            minima[row3_index] = float(
                active_ratios[zero[:, row3_index]].min()
            )
        for row3_index, row3 in enumerate(forms3):
            count = int(counts[row3_index])
            minimum = float(minima[row3_index])
            loss = float(losses[row3_index])
            # Smaller count/loss and larger minimum ratio are preferable.
            count_key = (count, -minimum, loss)
            ratio_key = (-minimum, count, loss)
            if best_count_key is None or count_key < best_count_key:
                best_count_key = count_key
                best_count_rows = (row2.copy(), row3.copy())
            if best_ratio_key is None or ratio_key < best_ratio_key:
                best_ratio_key = ratio_key
                best_ratio_rows = (row2.copy(), row3.copy())

    assert best_count_key is not None and best_count_rows is not None
    assert best_ratio_key is not None and best_ratio_rows is not None
    return {
        "count": {
            "rows": [row.copy() for row in best_count_rows],
            "killed": int(best_count_key[0]),
            "minimum_ratio": float(-best_count_key[1]),
            "weighted_loss": float(best_count_key[2]),
        },
        "max_min_ratio": {
            "rows": [row.copy() for row in best_ratio_rows],
            "killed": int(best_ratio_key[1]),
            "minimum_ratio": float(-best_ratio_key[0]),
            "weighted_loss": float(best_ratio_key[2]),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--weight-power", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=results_path("fixed49_d6_294_exact.json"),
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.weight_power) or args.weight_power <= 0:
        parser.error("--weight-power must be finite and positive")

    basis, _, diameter = load_forbidden("E6*")
    forbidden, ratios, _ = _forbidden_with_weights(basis, diameter)
    weights = np.power(
        np.maximum(0.0, 1.0 - ratios), args.weight_power
    )
    source_rows = load_e6_source_rows(args.source)
    facets = combigeo.relevant_facets(basis.tolist())
    started = time.perf_counter()
    payload = {
        "method": (
            "complete enumeration of every mod-2/mod-3 continuation after "
            "fixing two source mod-7 characters"
        ),
        "lattice": "E6*",
        "n": 6,
        "source_certificate": str(args.source),
        "source_rows": [row.astype(int).tolist() for row in source_rows],
        "target_index": 294,
        "moduli": [7, 7, 2, 3],
        "weight_power": args.weight_power,
        "projective_pool_sizes": {"mod2": 63, "mod3": 364},
        "fixed_pairs": [],
        "results": [],
        "valid_candidate": None,
    }

    for pair in itertools.combinations(range(3), 2):
        residual = np.ones(len(forbidden), dtype=bool)
        for row_index in pair:
            residual &= (
                forbidden @ source_rows[row_index]
            ) % 7 == 0
        objectives = best_rows_for_objectives(
            forbidden[residual],
            ratios[residual],
            weights[residual],
        )
        pair_record = {
            "fixed_row_indices": list(pair),
            "residual_projective_pairs": int(residual.sum()),
            "enumerated_row_pairs": 63 * 364,
            "objectives": {},
        }
        for label, result in objectives.items():
            full_rows = [
                source_rows[pair[0]],
                source_rows[pair[1]],
                *result["rows"],
            ]
            record = candidate_record(
                label=f"fixed-{pair[0]}-{pair[1]}-{label}",
                beta=(0.0 if label == "count" else args.weight_power),
                rows=full_rows,
                moduli=[7, 7, 2, 3],
                forbidden=forbidden,
                ratios=ratios,
                weights=weights,
                basis=basis,
                diameter=diameter,
                facets=facets,
                search_seconds=time.perf_counter() - started,
                search_metadata={
                    "complete_family": True,
                    "fixed_row_indices": list(pair),
                    "residual_projective_pairs": int(residual.sum()),
                    "enumerated_row_pairs": 63 * 364,
                    "objective": label,
                },
            )
            if record["image_index"] != 294:
                raise AssertionError("enumerated rows lost image size 294")
            pair_record["objectives"][label] = {
                "killed": record["killed"],
                "minimum_conflict_ratio": record[
                    "minimum_conflict_ratio"
                ],
                "rows": record["rows"],
            }
            payload["results"].append(record)
            if record.get("complete_separation", {}).get("valid"):
                payload["valid_candidate"] = record
        payload["fixed_pairs"].append(pair_record)
        print(
            f"pair={pair} residual={int(residual.sum())} "
            f"count={objectives['count']['killed']} "
            "best-min="
            f"{objectives['max_min_ratio']['minimum_ratio']:.12g}",
            flush=True,
        )

    payload["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    best = max(
        payload["results"],
        key=lambda record: float(
            record["minimum_conflict_ratio"]
            if record["minimum_conflict_ratio"] is not None
            else math.inf
        ),
    )
    print(
        f"FINAL best-min={best['minimum_conflict_ratio']} "
        f"killed={best['killed']} valid="
        f"{payload['valid_candidate'] is not None} saved={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
