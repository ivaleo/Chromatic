"""Push the rigorous width of ``chi(R^7) <= 1029`` with the certificate in the loop.

The piecewise certificate (:mod:`chromatic_research.core.piecewise_covrad`)
computes a *certified* diameter of a lamination in ~0.3 s, so it can serve as
the CMA objective directly: maximise

    min( separation_min(c), sqrt(7) ) / certified_diam(c)

over the 6-dimensional layer offset ``c`` (and a small grid of heights).  The
``sqrt(7)`` ceiling is the horizontal floor (proposition ``prop:planar``); the
separation is the exhaustive Fincke--Pohst minimum over the kernel.  Unlike the
earlier measured-mode search, every reported ratio ``>= 1`` here is already a
certificate-grade claim -- nothing rests on a from-below LP estimate.

Usage::

    python -m chromatic_research.campaigns.dim7_certified_push \
        --heights 1.00 1.03 1.06 --budget-each 900
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from chromatic_research.core.lamination import Lamination, kernel_rows, min_separation, unit_facets
from chromatic_research.core.piecewise_covrad import certify_two_layer
from chromatic_research.campaigns.dim7_laminate_e6s import e6star_geometry, kernels_for
from chromatic_research.paths import results_path

SQRT7 = math.sqrt(7.0)
R_BASE = math.sqrt(1.5)


def push(height: float, seed: int, budget: float, start_offset, output: Path,
         *, sigma: float = 0.15, popsize: int = 10) -> dict:
    import cma

    basis, radius, rows = e6star_geometry()
    kernels = kernels_for(rows, 3, seed, pool_size=60)

    def evaluate(offset):
        offset = np.asarray(offset, dtype=float)
        cert = certify_two_layer(basis, offset, height,
                                 covering_radius=R_BASE, verbose=False)
        diam = cert["certified_diam_upper"]
        lam = Lamination(basis, radius, offset, height)
        facets = unit_facets(lam.basis)
        best = (-1.0, None)
        for glue, kernel in kernels:
            sep = min_separation(kernel @ lam.basis, diam, facets)
            ratio = min(sep, SQRT7) / diam
            if ratio > best[0]:
                best = (ratio, glue)
        return best[0], best[1], diam

    start = time.time()
    value, glue, diam = evaluate(start_offset)
    print(f"t={height}: start certified ratio {value:.7f} (diam {diam:.6f})", flush=True)
    best_record = {"index": 1029, "modulus": 3, "height": height, "seed": seed,
                   "ratio_certified": value, "glue": list(glue) if glue else None,
                   "offset": [float(x) for x in start_offset],
                   "certified_diam": diam, "evaluations": 0}
    output.write_text(json.dumps(best_record, indent=1) + "\n")
    strategy = cma.CMAEvolutionStrategy(list(map(float, start_offset)), sigma,
                                        {"popsize": popsize, "verbose": -9, "seed": seed})
    best, evaluations, last_gain = value, 0, time.time()
    while not strategy.stop():
        if time.time() - start > budget or time.time() - last_gain > 240:
            break
        candidates = strategy.ask()
        losses = []
        for candidate in candidates:
            v, g, d = evaluate(candidate)
            evaluations += 1
            losses.append(-v)
            if v > best:
                best, last_gain = v, time.time()
                best_record = {"index": 1029, "modulus": 3, "height": height,
                               "seed": seed, "ratio_certified": v, "glue": list(g),
                               "offset": np.asarray(candidate, float).tolist(),
                               "certified_diam": d, "evaluations": evaluations}
                output.write_text(json.dumps(best_record, indent=1) + "\n")
                print(f"  eval {evaluations}: certified ratio {v:.7f} "
                      f"diam {d:.6f} [{time.time() - start:.0f}s]", flush=True)
        strategy.tell(candidates, losses)
    print(f"t={height}: final certified ratio {best:.7f} after {evaluations} evals",
          flush=True)
    return best_record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heights", type=float, nargs="+", default=[1.00, 1.03, 1.06])
    parser.add_argument("--budget-each", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-config", type=Path,
                        default=None)
    args = parser.parse_args(argv)

    start_cfg = args.start_config or results_path("dim7_laminate_m3_t1.00_verified.json")
    start_offset = json.loads(Path(start_cfg).read_text())["offset"]

    records = []
    for height in args.heights:
        out = results_path(f"dim7_certified_t{height:.2f}.json")
        records.append(push(height, args.seed, args.budget_each, start_offset, out))
    best = max(records, key=lambda r: r["ratio_certified"])
    summary = results_path("dim7_certified_push.json")
    summary.write_text(json.dumps({"runs": records, "best": best}, indent=1) + "\n")
    print(json.dumps(best, indent=1))
    print(f"saved {summary}", flush=True)
    return 0 if best["ratio_certified"] >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
