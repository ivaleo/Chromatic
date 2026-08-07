"""Certified diameter bounds for the dimension-9 laminations (box B&B certifier).

Targets, in order:

1. **Calibration** on index 12005 (``t = 0.69``, deep-hole shift): the true
   radius is exactly ``R^2 = R8^2 + t^2/4`` (a doubly-deep hole exists), so the
   certifier must accept a target just above it and reject one just below.
2. **Index 9604** (``t = 0.93``, off-hole shift): every certified diameter
   below the (P1) bound ``sqrt(6 + t^2) = 2.62`` widens the RIGOROUS interval
   of ``chi(R^9, [1, l]) <= 9604`` beyond ``l = 1.0098``; the measured diameter
   is ``2.5004`` (would give ``l = 1.058``).
3. **Index 7203** (``t = 1.15``): certify ``diam <= sqrt(7) = 2.6458`` and the
   candidate becomes ``chi(R^9) <= 7203``; the measured diameter is ``2.6026``
   and the previous best rigorous bound was ``2.7060``.

Usage::

    python -m chromatic_research.campaigns.dim9_certify_radius calibrate
    python -m chromatic_research.campaigns.dim9_certify_radius push9604
    python -m chromatic_research.campaigns.dim9_certify_radius decide7203
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from chromatic_research.campaigns.e8_neighbor_search import e8_geometry
from chromatic_research.core.layered_covrad import LayeredCertifier
from chromatic_research.paths import results_path


def load_config(name: str) -> dict:
    return json.loads((results_path(name)).read_text())


E8_COVERING_RADIUS = math.sqrt(6.0) / 2.0     # lambda1^2 = 3, 2R/lambda1 = sqrt(2)


def certifier_for(config: dict, **kwargs) -> LayeredCertifier:
    basis, _, _ = e8_geometry()
    return LayeredCertifier(base=basis, offset=np.asarray(config["offset"], float),
                            height=float(config["height"]), base_r2=1.5, **kwargs)


def calibrate(budget: float) -> dict:
    config = load_config("dim9_laminate_m5_verified.json")
    cert = certifier_for(config)
    exact_r2 = 1.5 + float(config["height"]) ** 2 / 4.0
    print(f"calibration on 12005: t={config['height']}, exact R^2 = {exact_r2:.6f}",
          flush=True)
    above = cert.certify(exact_r2 * 1.002, initial_radius=E8_COVERING_RADIUS + 1e-6, budget=budget)
    print(f"  target 1.002x exact: certified={above['certified']} "
          f"({above.get('reason', 'ok')})", flush=True)
    below = cert.certify(exact_r2 * 0.998, initial_radius=E8_COVERING_RADIUS + 1e-6, budget=budget / 4)
    print(f"  target 0.998x exact: certified={below['certified']} "
          f"({below.get('reason', 'ok')}) -- must NOT certify", flush=True)
    report = {"exact_r2": exact_r2, "above": above, "below": below,
              "passed": bool(above["certified"] and not below["certified"])}
    out = results_path("dim9_certify_calibration.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"calibration {'PASSED' if report['passed'] else 'FAILED'}; saved {out}",
          flush=True)
    return report


def push9604(budget: float, diam_targets) -> dict:
    config = load_config("dim9_laminate_m4_strict_verified.json")
    cert = certifier_for(config)
    safe = math.sqrt(6.0 + float(config["height"]) ** 2)
    print(f"9604: t={config['height']} safe(P1) diam={safe:.6f}, "
          f"measured {config.get('diameter_measured', float('nan')):.6f}", flush=True)
    runs = []
    best = None
    for diam in diam_targets:
        result = cert.certify((diam / 2.0) ** 2, initial_radius=E8_COVERING_RADIUS + 1e-6, budget=budget)
        result["diam_target"] = diam
        result["width_if_certified"] = math.sqrt(7.0) / diam
        runs.append(result)
        print(f"  diam<={diam:.4f} (width {math.sqrt(7)/diam:.6f}): "
              f"certified={result['certified']} ({result.get('reason', 'ok')}, "
              f"{result['processed']} boxes, {result['seconds']}s)", flush=True)
        if result["certified"]:
            best = result
        else:
            break                        # targets are descending; stop at first failure
    report = {"config": config["index"], "height": config["height"],
              "safe_diameter": safe, "runs": runs, "best_certified": best}
    out = results_path("dim9_certify_9604.json")
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"saved {out}", flush=True)
    return report


def decide7203(budget: float) -> dict:
    config = load_config("dim9_laminate_m3_verified.json")
    cert = certifier_for(config)
    target_diam = math.sqrt(7.0)
    print(f"7203: t={config['height']}, target diam <= sqrt(7) = {target_diam:.9f}, "
          f"measured {config.get('diameter_measured', float('nan'))}", flush=True)
    result = cert.certify((target_diam / 2.0) ** 2, initial_radius=E8_COVERING_RADIUS + 1e-6, budget=budget)
    result["diam_target"] = target_diam
    out = results_path("dim9_certify_7203.json")
    out.write_text(json.dumps({"config": config["index"], "height": config["height"],
                               "result": result}, indent=1) + "\n")
    verdict = ("CERTIFIED: chi(R^9) <= 7203" if result["certified"]
               else f"not certified ({result.get('reason', '?')})")
    print(f"{verdict}; {result['processed']} boxes in {result['seconds']}s; saved {out}",
          flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--budget", type=float, default=1200.0)
    p = sub.add_parser("push9604")
    p.add_argument("--budget", type=float, default=1500.0)
    p.add_argument("--diam-targets", type=float, nargs="+",
                   default=[2.60, 2.58, 2.56, 2.54, 2.52])
    d = sub.add_parser("decide7203")
    d.add_argument("--budget", type=float, default=3600.0)
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        report = calibrate(args.budget)
        return 0 if report["passed"] else 2
    if args.command == "push9604":
        report = push9604(args.budget, args.diam_targets)
        return 0 if report["best_certified"] else 2
    result = decide7203(args.budget)
    return 0 if result["certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
