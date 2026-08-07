"""Rank-5 Eisenstein lattices: minimise ``rho = diam / lambda1`` towards ``sqrt(7/3)``.

If a rank-5 ``Z[omega]``-lattice (dimension 10) with ``rho <= sqrt(7/3) = 1.5275``
exists, the ``(3+omega)`` colouring gives ``chi(R^10) <= 7^5 = 16807`` -- against
the current ``3^10 = 59049``.  The project's negative result covered only the
Construction-A family (ternary codes, closed by the Singleton bound); arbitrary
Hermitian forms are unexplored.  This campaign searches them directly:

- parameters: complex lower-triangular Cholesky factor ``L`` of the Hermitian
  Gram ``H = L L^dagger`` (5 log-diagonals + 10 complex off-diagonals = 25 real
  parameters, scale-invariant objective);
- evaluation: real 10x10 Gram ``B((alpha e_i), (beta e_j)) = Re(conj(alpha)
  beta H_ij)`` over the Z-basis ``(e_1, omega e_1, ..., e_5, omega e_5)``;
  ``lambda1`` by ``shortest_vector``, ``R`` from below by vertex ascent;
- seeds: ``Z[omega]^5``, the complex hyperplane sections of ``K12`` (the best
  packing in their neighbourhood), and random perturbations;
- the estimator approaches ``R`` from below, so the reported ``rho`` is a lower
  estimate: any candidate near the target is re-measured with a 20x budget
  before being believed.

Usage::

    python -m chromatic_research.campaigns.eisenstein_rank5 --budget 3600
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import combigeo
from chromatic_research.core.k12 import build_k12, complex_section, real_embedding
from chromatic_research.core.lamination import deep_hole, enumerate_upto
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)
TARGET = math.sqrt(7.0 / 3.0)


def hermitian_from_params(params: np.ndarray) -> np.ndarray:
    """25 real parameters -> positive-definite Hermitian 5x5 via Cholesky."""
    params = np.asarray(params, dtype=float)
    L = np.zeros((5, 5), dtype=complex)
    k = 0
    for i in range(5):
        L[i, i] = math.exp(min(max(params[k], -6.0), 6.0)); k += 1
    for i in range(1, 5):
        for j in range(i):
            L[i, j] = complex(params[k], params[k + 1]); k += 2
    return L @ L.conj().T


def params_from_hermitian(H: np.ndarray) -> np.ndarray:
    L = np.linalg.cholesky(H)
    params = []
    for i in range(5):
        params.append(math.log(max(L[i, i].real, 1e-9)))
    for i in range(1, 5):
        for j in range(i):
            params += [L[i, j].real, L[i, j].imag]
    return np.asarray(params)


def real_gram(H: np.ndarray) -> np.ndarray:
    """Z-basis (e_1, w e_1, .., e_5, w e_5); B(a e_i, b e_j) = Re(conj(a) b H_ij)."""
    units = (1.0 + 0.0j, OMEGA)
    G = np.zeros((10, 10))
    for i in range(5):
        for j in range(5):
            for a in range(2):
                for b in range(2):
                    G[2 * i + a, 2 * j + b] = (units[a].conjugate() * units[b]
                                               * H[i, j]).real
    return G


def basis_from_params(params: np.ndarray) -> np.ndarray:
    G = real_gram(hermitian_from_params(params))
    # numerical symmetrisation; Cholesky needs strict PD
    G = (G + G.T) / 2
    return np.linalg.cholesky(G + 1e-12 * np.eye(10))


def measure_rho(params: np.ndarray, *, n_dirs: int = 160, seed: int = 5) -> float:
    try:
        basis = basis_from_params(params)
    except np.linalg.LinAlgError:
        return 10.0
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    if lam1 < 1e-6:
        return 10.0
    radius, _ = deep_hole(basis, n_dirs=n_dirs, seed=seed)
    return 2.0 * radius / lam1


def k12_section_hermitian() -> np.ndarray:
    """Hermitian Gram of the complex hyperplane section of K12 (rank 5)."""
    coeff, basis = build_k12()
    T = real_embedding()
    minimal = enumerate_upto(basis, 2.0 + 1e-9)
    coords = np.rint(np.linalg.solve(basis.T, minimal.T).T).astype(np.int64)
    direction = coords[0] @ coeff
    section_coeff = complex_section(coeff, direction)      # 10 rows in Z[omega]^6 coords
    real_rows = section_coeff @ T                          # 10 x 12

    def to_complex(row):
        return np.array([complex(row[2 * j], row[2 * j + 1]) for j in range(6)])

    complex_rows = [to_complex(r) for r in real_rows]
    # greedy Z[omega]-basis: add shortest vectors that enlarge the C-span
    order = np.argsort([float(np.linalg.norm(r)) for r in real_rows])
    chosen: list[np.ndarray] = []
    for idx in order:
        candidate = complex_rows[idx]
        trial = np.array(chosen + [candidate])
        if np.linalg.matrix_rank(trial, tol=1e-9) == len(trial):
            chosen.append(candidate)
        if len(chosen) == 5:
            break
    assert len(chosen) == 5, "failed to extract a rank-5 C-basis of the section"
    V = np.array(chosen)                                    # 5 x 6 complex
    return V @ V.conj().T


def run(budget: float, seed: int, sigma: float, output: Path) -> dict:
    import cma

    start = time.time()
    seeds = [("Z[w]^5", params_from_hermitian(np.eye(5, dtype=complex)))]
    try:
        H_sec = k12_section_hermitian()
        H_sec = H_sec / (np.trace(H_sec).real / 5)
        seeds.append(("K12-complex-section", params_from_hermitian(H_sec)))
    except Exception as error:                              # noqa: BLE001
        print(f"K12 section seed failed: {error}", flush=True)

    for name, params in seeds:
        rho = measure_rho(params, n_dirs=400)
        print(f"seed {name}: rho >= {rho:.6f} (target {TARGET:.6f})", flush=True)

    best_overall = {"rho": math.inf}
    records = []
    rng = np.random.default_rng(seed)
    per_seed = budget / len(seeds)
    for name, start_params in seeds:
        strategy = cma.CMAEvolutionStrategy(
            list(map(float, start_params)), sigma,
            {"popsize": 14, "verbose": -9, "seed": seed})
        best, best_params, evaluations = math.inf, np.asarray(start_params), 0
        seed_start = time.time()
        while not strategy.stop() and time.time() - seed_start < per_seed:
            candidates = strategy.ask()
            losses = []
            for candidate in candidates:
                rho = measure_rho(np.asarray(candidate),
                                  seed=int(rng.integers(1, 10**6)))
                evaluations += 1
                losses.append(rho)
                if rho < best:
                    best, best_params = rho, np.asarray(candidate)
                    print(f"  [{name}] eval {evaluations}: rho >= {best:.6f} "
                          f"[{time.time() - start:.0f}s]"
                          f"{'  >>> BELOW TARGET (re-verify!)' if best <= TARGET else ''}",
                          flush=True)
            strategy.tell(candidates, losses)
        verified = measure_rho(best_params, n_dirs=3000, seed=7)
        record = {"seed_name": name, "evaluations": evaluations,
                  "rho_search": best, "rho_verified": verified,
                  "params": best_params.tolist(),
                  "below_target": bool(verified <= TARGET)}
        records.append(record)
        print(f"[{name}] finished: search rho {best:.6f}, verified rho {verified:.6f} "
              f"({evaluations} evals)", flush=True)
        if verified < best_overall.get("rho", math.inf):
            best_overall = {"rho": verified, **record}
        output.write_text(json.dumps(
            {"target": TARGET, "runs": records, "best": best_overall},
            indent=1) + "\n")
    return best_overall


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    out = args.output or results_path("eisenstein_rank5_cma.json")
    best = run(args.budget, args.seed, args.sigma, out)
    print(json.dumps(best, indent=1))
    print(f"saved {out}", flush=True)
    return 0 if best.get("below_target") else 2


if __name__ == "__main__":
    raise SystemExit(main())
