"""Certify a laminated colouring end-to-end: box-B&B diameter vs. separation.

A lamination config (offset, height, glue, modulus) is a valid colouring iff
``diam(Lambda) <= separation_min``, where ``separation_min`` comes from the
exhaustive Fincke--Pohst check of the verify pass (exact up to QP projection,
double-checked by two independent routines).  The missing rigorous piece has
always been the *diameter upper bound*; the (P1) bound ``sqrt(diam_base^2 +
t^2)`` is often too weak.  :mod:`chromatic_research.core.layered_covrad`
closes the gap: it certifies ``R^2 <= target`` by branch-and-bound over boxes,
using only exact base-lattice distances.

This campaign glues the two: read a verified config, take its
``separation_min``, and try to certify ``diam <= separation_min / margin`` for
a ladder of margins.  Any success turns the config into a rigorous bound

    chi(R^n, [1, l]) <= index,   l = separation_min / certified_diam >= 1.

Supported bases: ``e8`` (dimension-9 laminations) and ``e6s`` (dimension-7,
index ``1029 = 3 * 343``).

Usage::

    python -m chromatic_research.campaigns.certify_lamination \
        --base e6s --config results/dim7_laminate_m3_t1.00_verified.json \
        --budget 2400
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from chromatic_research.core.layered_covrad import LayeredCertifier
from chromatic_research.paths import results_path


def base_basis(name: str) -> np.ndarray:
    if name == "e8":
        from chromatic_research.campaigns.e8_neighbor_search import e8_geometry
        basis, _, _ = e8_geometry()
        return basis
    if name == "e6s":
        from chromatic_research.campaigns.dim7_laminate_e6s import e6star_geometry
        basis, _, _ = e6star_geometry()
        return basis
    raise SystemExit(f"unknown base {name!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=("e8", "e6s"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=2400.0)
    parser.add_argument("--margins", type=float, nargs="+",
                        default=[1.0, 1.005, 1.01, 1.02],
                        help="certify diam <= separation/margin, tightest first that works")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    separation = float(config["separation_min"])
    basis = base_basis(args.base)
    certifier = LayeredCertifier(base=basis,
                                 offset=np.asarray(config["offset"], float),
                                 height=float(config["height"]), base_r2=1.5)
    print(f"index {config.get('exact_index', config.get('index'))}: "
          f"separation={separation:.9f}, measured diam="
          f"{config.get('diameter_measured', float('nan')):.9f}, "
          f"safe (P1) diam={config.get('diameter_safe', float('nan')):.9f}",
          flush=True)

    runs, best = [], None
    for margin in args.margins:
        diam_target = separation / margin
        result = certifier.certify((diam_target / 2.0) ** 2,
                                   initial_radius=math.sqrt(6.0) / 2.0 + 1e-6,
                                   budget=args.budget)
        result["diam_target"] = diam_target
        result["margin"] = margin
        result["width_if_certified"] = separation / diam_target
        runs.append(result)
        print(f"  diam <= {diam_target:.6f} (width {separation / diam_target:.6f}): "
              f"certified={result['certified']} ({result.get('reason', 'ok')}, "
              f"{result['processed']} boxes, {result['seconds']}s)", flush=True)
        if result["certified"]:
            best = result
            break                          # margins ascend: first success is tightest width=margin... keep going for wider margin? margins given ascending: first that certifies gives the smallest margin (largest target) -- stop
    report = {"config_file": str(args.config), "base": args.base,
              "index": config.get("exact_index", config.get("index")),
              "separation_min": separation, "runs": runs, "certified": best}
    out = args.output or results_path(
        f"certify_{args.base}_{config.get('exact_index', config.get('index'))}.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    if best:
        print(f"RIGOROUS: chi(R^n,[1,{best['width_if_certified']:.6f}]) <= "
              f"{report['index']}  (diam <= {best['diam_target']:.6f} certified)",
              flush=True)
    print(f"saved {out}", flush=True)
    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())
