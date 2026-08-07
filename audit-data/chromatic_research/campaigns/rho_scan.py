"""``rho = diam / lambda1`` scan over the lattice zoo in dimensions 9--12.

For the colouring ``Gamma = 3 Lambda`` the width is ``d = 2 / rho``
(real-multiplier rule), so every lattice with a smaller ``rho`` than the
current carrier improves the interval width at the same index ``3^n``.
Carriers so far: the ``E8`` lamination tower (rigorous widths
``1.1795 / 1.1166 / 1.0755`` at ``n = 10, 11, 12``) and now ``K12``
(``sqrt(3/2) = 1.2247`` at ``n = 12``).

The scan measures ``lambda1`` exactly (shortest_vector) and ``R`` from below
(vertex ascent), so the reported ``rho`` is a *lower* estimate and the implied
width ``2/rho`` an upper one -- candidates must then be certified separately.
Sections of ``K12`` (real hyperplane for ``n = 11``, complex hyperplane for
``n = 10``) are included: they inherit good packing from ``K12`` and are the
natural candidates for beating the tower's widths.

Usage::

    python -m chromatic_research.campaigns.rho_scan
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import combigeo
from chromatic_research.core.k12 import (
    build_k12, complex_section, real_section, real_embedding, rotate_to_dimension,
)
from chromatic_research.core.lattices import Astar, D, Dstar, E8
from chromatic_research.core.lamination import deep_hole, enumerate_upto
from chromatic_research.paths import results_path


def tower_bases(nmax: int = 12) -> dict[str, np.ndarray]:
    """Laminated lattices Lambda_9..Lambda_nmax exactly as in dim10_12_tower."""
    basis = np.asarray(E8(), dtype=float)
    basis = basis / float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    radius_lower = math.sqrt(0.5)
    out = {}
    for n in range(9, nmax + 1):
        height = math.sqrt(max(1.0 - radius_lower**2, 1e-12))
        _, hole = deep_hole(basis, n_dirs=450, seed=3)
        lifted = np.zeros((n, n))
        lifted[:n - 1, :n - 1] = basis
        lifted[n - 1, :n - 1] = hole
        lifted[n - 1, n - 1] = height
        basis = lifted
        radius_lower, _ = deep_hole(basis, n_dirs=900, seed=5)
        out[f"Lam{n}"] = basis.copy()
    return out


def k12_sections() -> dict[str, np.ndarray]:
    coeff, _ = build_k12()
    T = real_embedding()
    minimal = enumerate_upto(coeff @ T, 2.0 + 1e-9)
    coords = np.rint(np.linalg.solve((coeff @ T).T, minimal.T).T).astype(np.int64)
    out = {}
    for label, index in (("a", 0), ("b", 1)):
        direction = coords[index] @ coeff        # coefficient row in Z[omega]^6
        real11 = real_section(coeff, direction) @ T
        out[f"K12sec11{label}"] = rotate_to_dimension(real11)
        real10 = complex_section(coeff, direction) @ T
        out[f"K12sec10{label}"] = rotate_to_dimension(real10)
    return out


def measure(name: str, basis: np.ndarray, n_dirs: int) -> dict:
    start = time.time()
    basis = np.asarray(basis, dtype=float)
    n = basis.shape[0]
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    radius, _ = deep_hole(basis, n_dirs=n_dirs, seed=5)
    rho = 2.0 * radius / lam1
    record = {
        "name": name, "n": n, "lambda1": lam1,
        "covering_radius_measured": radius, "rho_measured": rho,
        "d_upper_estimate": 2.0 / rho, "admissible": bool(rho <= 2.0),
        "seconds": round(time.time() - start, 1),
    }
    print(f"{name:12s} n={n:2d} lam1={lam1:.6f} R>={radius:.6f} rho>={rho:.6f} "
          f"d<={2.0 / rho:.6f} [{record['seconds']}s]", flush=True)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-dirs", type=int, default=1500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    candidates: dict[str, np.ndarray] = {}
    for n in (9, 10, 11, 12):
        candidates[f"A{n}*"] = Astar(n)
        candidates[f"D{n}"] = D(n)
        candidates[f"D{n}*"] = Dstar(n)
    _, k12 = build_k12()
    candidates["K12"] = k12
    candidates.update(tower_bases())
    candidates.update(k12_sections())

    records = [measure(name, basis, args.n_dirs) for name, basis in candidates.items()]
    best: dict[int, dict] = {}
    for record in records:
        n = record["n"]
        if record["admissible"] and (n not in best
                                     or record["rho_measured"] < best[n]["rho_measured"]):
            best[n] = record
    payload = {"records": records,
               "best_per_dimension": {str(n): best[n] for n in sorted(best)}}
    out = args.output or results_path("rho_scan_9_12.json")
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("\nbest admissible per dimension:", flush=True)
    for n in sorted(best):
        b = best[n]
        print(f"  n={n}: {b['name']} rho={b['rho_measured']:.6f} "
              f"d={b['d_upper_estimate']:.6f}", flush=True)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
