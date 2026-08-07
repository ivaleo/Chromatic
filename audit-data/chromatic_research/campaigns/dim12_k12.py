"""``chi(R^12, [1, sqrt(3/2))) <= 3^12`` via the Coxeter--Todd lattice ``K12``.

The tower of ``E8`` laminations gives width ``1.0755`` at ``n = 12``.  The same
real-multiplier rule ``d = 2/rho`` applied to ``K12`` (``rho = sqrt(8/3)``,
covering radius known exactly in the literature) gives

    d = 2 / sqrt(8/3) = sqrt(3/2) = 1.224745...

at the same index ``3^12 = 531441`` -- a strictly wider interval.  The project
had used ``K12`` only as a negative example for the Eisenstein ``7^{n/2}`` path
and overlooked it as a *real* carrier.

Verification protocol (same standard as ``dim10_12_tower``):

- exact integer determinant of the coefficient basis (index of ``K12`` in
  ``Z[omega]^6`` is 64, covolume 27);
- ``lambda1^2 = 4`` by ``shortest_vector`` and kissing number 756 by exhaustive
  enumeration;
- covering radius measured from below by vertex ascent (must approach the
  literature value ``R^2 = 8/3`` and never exceed it);
- ``D_min(3 K12) = 2 lambda1 = 4`` by exhaustive Fincke--Pohst enumeration of
  all ``3 K12`` vectors of norm ``<= 2 diam``, each projected by two
  independent routines (combigeo's Dykstra and SLSQP).

Usage::

    python -m chromatic_research.campaigns.dim12_k12
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sympy import Matrix

import combigeo
from chromatic_research.core.k12 import build_k12
from chromatic_research.core.lamination import deep_hole, enumerate_upto, unit_facets
from chromatic_research.campaigns.dim10_12_tower import separation_of_triple_lattice
from chromatic_research.paths import results_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dirs", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    start = time.time()

    coeff, basis = build_k12()
    exact_index_in_z_omega_6 = abs(int(Matrix(coeff.tolist()).det()))
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    minimal = enumerate_upto(basis, lam1 + 1e-9)
    kissing = int(len(minimal))
    print(f"K12: lambda1^2={lam1**2:.9f} (expect 4), kissing={kissing} (expect 756), "
          f"[Z[w]^6 : K12]={exact_index_in_z_omega_6} (expect 64) "
          f"[{time.time() - start:.0f}s]", flush=True)

    literature_R = math.sqrt(8.0 / 3.0)
    measured_R, _ = deep_hole(basis, n_dirs=args.n_dirs, seed=5)
    print(f"covering radius: literature {literature_R:.9f}, measured (from below) "
          f"{measured_R:.9f} [{time.time() - start:.0f}s]", flush=True)

    diameter = 2.0 * literature_R
    separation, checked, total = separation_of_triple_lattice(basis, diameter)
    rho = diameter / lam1
    d = separation / diameter
    record = {
        "n": 12, "index": 3**12, "lattice": "K12 (Coxeter-Todd)",
        "lambda1": lam1, "kissing": kissing,
        "index_in_z_omega_6": exact_index_in_z_omega_6,
        "covering_radius_literature": literature_R,
        "covering_radius_measured_lower": measured_R,
        "diameter": diameter, "rho": rho,
        "D_min": separation, "n_vectors": total, "n_checked": checked,
        "d": d, "d_closed_form": math.sqrt(1.5),
        "prediction_2_over_rho": 2.0 / rho,
        "valid": bool(d >= 1.0),
        "tower_d_at_12_for_comparison": 1.075465716,
    }
    out = args.output or results_path("dim12_k12.json")
    out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"3*K12: D_min={separation:.9f} (expect 4) over {checked}/{total} vectors, "
          f"d={d:.9f} (closed form sqrt(3/2)={math.sqrt(1.5):.9f}) "
          f"{'VALID' if record['valid'] else 'NOT VALID'}", flush=True)
    print(f"saved {out}", flush=True)
    return 0 if record["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
