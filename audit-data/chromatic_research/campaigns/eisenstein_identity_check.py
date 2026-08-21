"""Numeric verification of the Eisenstein identity (10) and of its two corollaries.

The identity ``D((3+omega)Lambda)^2 = (7/3) lambda1(Lambda)^2`` used to be a
conjecture checked at ``n = 2, 4, 6, 8``.  Its ``>=`` half is the planar theorem
(``prop:planar``, see :mod:`planar_theorem_check`); the ``<=`` half is now
``prop:eisup`` -- the point ``q = (2w + omega w)/3`` built from a minimal vector
always lies in ``V0``.  Since both halves are theorems, this module is a
*control*, not evidence; it exists so the paper's claims stay falsifiable.

Three checks:

``identity``
    ``q in V0`` and ``D((3+omega)Lambda) == sqrt(7/3) lambda1`` by exhaustive
    enumeration of the sublattice in the provably complete window
    ``|v| < D_* + 2R`` (``D(v) >= |v| - 2R`` because ``V0 ⊆ B(0, R)``; ``R`` is
    bounded above by the Babai bound of an LLL-reduced basis).  Run on ``A2``,
    ``D4``, ``E6*``, ``E8``, stretched sums ``A2 + c A2`` and random *skew*
    Eisenstein lattices of complex rank 2 and 3 -- i.e. well outside the
    original ``n = 2, 4, 6, 8`` list.

``alpha_ladder``
    ``D(alpha Lambda) == 2 lambda1 dist(alpha/2, U)`` for every ``alpha`` in the
    order, for all four orders ``Z``, ``Z[i]``, ``Z[omega]``, ``H`` -- the
    equality half of ``prop:alphaladder`` (``cor:Uexact``).

``a2_triangle``
    The Eisenstein structure is not needed: any lattice carrying minimal
    vectors ``w, u`` with ``<w,u> = -lambda1^2/2`` has ``(2w+u)/3 in V0`` and
    ``D(3w+u) = sqrt(7/3) lambda1`` (``rem:a2tri``).  Run on the *non*-Eisenstein
    ``A3, A4, A5, D5, E7``.

Usage::

    python -m chromatic_research.campaigns.eisenstein_identity_check
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import combigeo
from chromatic_research.core.lamination import enumerate_upto, unit_facets
from chromatic_research.paths import results_path

OMEGA = complex(-0.5, math.sqrt(3) / 2)
THETA = OMEGA - OMEGA.conjugate()
SQRT73 = math.sqrt(7.0 / 3.0)
TOL = 1e-7

HURWITZ_UNITS = (
    [(1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
     (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1)]
    + [tuple(s / 2 for s in c) for c in itertools.product((1, -1), repeat=4)]
)
HURWITZ_BASIS = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.5, 0.5, 0.5, 0.5)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def realify(vectors) -> np.ndarray:
    return np.array([[c for z in v for c in (z.real, z.imag)] for v in vectors], float)


def eisenstein_basis(columns) -> np.ndarray:
    """Real basis of the ``Z[omega]``-module spanned by ``columns``."""
    generators = []
    for column in columns:
        generators.append(column)
        generators.append(tuple(OMEGA * z for z in column))
    return realify(generators)


def complex_basis(columns, unit) -> np.ndarray:
    generators = []
    for column in columns:
        generators.append(column)
        generators.append(tuple(unit * z for z in column))
    return realify(generators)


def cmul(vector: np.ndarray, z: complex) -> np.ndarray:
    out = np.empty_like(vector)
    for j in range(len(vector) // 2):
        c = complex(vector[2 * j], vector[2 * j + 1]) * z
        out[2 * j], out[2 * j + 1] = c.real, c.imag
    return out


def qmul(vector: np.ndarray, q) -> np.ndarray:
    a, b, c, d = q
    out = np.empty_like(vector)
    for j in range(len(vector) // 4):
        x0, x1, x2, x3 = vector[4 * j:4 * j + 4]
        out[4 * j + 0] = x0 * a - x1 * b - x2 * c - x3 * d
        out[4 * j + 1] = x0 * b + x1 * a + x2 * d - x3 * c
        out[4 * j + 2] = x0 * c - x1 * d + x2 * a + x3 * b
        out[4 * j + 3] = x0 * d + x1 * c - x2 * b + x3 * a
    return out


def square(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis, float)
    if basis.shape[0] == basis.shape[1]:
        return basis
    return np.linalg.cholesky(basis @ basis.T)


def covering_radius_upper(basis: np.ndarray) -> float:
    """Babai bound ``R <= 1/2 sqrt(sum |b_i*|^2)`` on an LLL-reduced basis."""
    reduced = np.asarray(combigeo.lll_reduce(np.asarray(basis, float).tolist()), float)
    star: list[np.ndarray] = []
    for row in reduced:
        v = row.copy()
        for s in star:
            v = v - (v @ s) / (s @ s) * s
        star.append(v)
    return 0.5 * math.sqrt(sum(float(s @ s) for s in star))


def sublattice_cell_facets(generators, lam1: float, reach: int = 4):
    """Halfspaces of the Voronoi cell of ``Z<generators>``; normals lie in its span."""
    seen = {}
    for coefficients in itertools.product(range(-reach, reach + 1), repeat=len(generators)):
        if not any(coefficients):
            continue
        v = sum(c * g for c, g in zip(coefficients, generators))
        norm = float(np.linalg.norm(v))
        if norm <= 2.6 * lam1 + 1e-9:
            seen[tuple(np.round(v, 9))] = (list(v / norm), norm / 2)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# check 1 -- the identity itself
# --------------------------------------------------------------------------- #
def check_identity(name: str, basis: np.ndarray) -> dict:
    basis = np.asarray(basis, float)
    w = np.asarray(combigeo.shortest_vector(basis.tolist()), float)
    lam1 = float(np.linalg.norm(w))
    facets = unit_facets(basis)
    omega_w = cmul(w, OMEGA)

    q = (2 * w + omega_w) / 3.0
    dist_q = float(combigeo.dist_to_halfspaces(q.tolist(), facets))

    radius = covering_radius_upper(basis)
    gamma = np.array([cmul(row, 3 + OMEGA) for row in basis])
    d_star = 2.0 * float(combigeo.dist_to_halfspaces((cmul(w, 3 + OMEGA) / 2).tolist(), facets))
    window = d_star + 2 * radius + 1e-9          # D(v) >= |v| - 2R makes this complete
    vectors = enumerate_upto(gamma, window)
    d_min = min(2.0 * float(combigeo.dist_to_halfspaces((v / 2).tolist(), facets))
                for v in vectors)

    target = SQRT73 * lam1
    record = {
        "lattice": name, "n": int(basis.shape[0]), "lambda1": lam1,
        "dist_q_to_V0": dist_q,
        "inner_product_error": abs(float(w @ omega_w) + 0.5 * lam1 ** 2),
        "covering_radius_upper": radius,
        "window": window, "vectors_enumerated": int(len(vectors)),
        "D_min": d_min, "sqrt73_lambda1": target,
        "relative_error": (d_min - target) / target,
        "holds": bool(dist_q <= 1e-9 and abs(d_min - target) <= TOL * max(1.0, target)),
    }
    print(f"identity {name:26s} D={d_min:.12f} vs {target:.12f} "
          f"(rel {record['relative_error']:+.2e}, dist(q,V0)={dist_q:.1e}, "
          f"{len(vectors)} vectors)", flush=True)
    return record


# --------------------------------------------------------------------------- #
# check 2 -- alpha-ladder equality over all four orders
# --------------------------------------------------------------------------- #
def check_alpha_ladder(name: str, basis: np.ndarray, unit_maps, alphas) -> dict:
    basis = square(basis)
    w = np.asarray(combigeo.shortest_vector(basis.tolist()), float)
    lam1 = float(np.linalg.norm(w))
    facets = unit_facets(basis)
    cell = sublattice_cell_facets([m(w) for m in unit_maps], lam1)

    rows, worst = [], 0.0
    for label, alpha_map in alphas:
        v = alpha_map(w)
        measured = 2.0 * float(combigeo.dist_to_halfspaces((v / 2).tolist(), facets))
        predicted = 2.0 * float(combigeo.dist_to_halfspaces((v / 2).tolist(), cell))
        worst = max(worst, abs(measured - predicted))
        rows.append({"alpha": label, "D": measured, "2*dist(alpha/2,U)": predicted})
    record = {"lattice": name, "n": int(basis.shape[0]), "lambda1": lam1,
              "cell_facets": len(cell), "alphas": rows,
              "worst_gap": worst, "holds": bool(worst <= TOL)}
    print(f"ladder   {name:26s} worst |D - 2 dist(alpha/2,U)| = {worst:.2e} "
          f"over {len(rows)} alphas", flush=True)
    return record


# --------------------------------------------------------------------------- #
# check 3 -- A2 triangle, no Eisenstein structure
# --------------------------------------------------------------------------- #
def check_a2_triangle(name: str, basis: np.ndarray) -> dict:
    basis = np.asarray(combigeo.lll_reduce(square(basis).tolist()), float)
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    facets = unit_facets(basis)
    minimal = [v for v in enumerate_upto(basis, lam1 * (1 + 1e-9))
               if abs(np.linalg.norm(v) - lam1) < 1e-9]
    pair = None
    for a, b in itertools.combinations(range(len(minimal)), 2):
        if abs(float(minimal[a] @ minimal[b]) + lam1 ** 2 / 2) < 1e-9:
            pair = (minimal[a], minimal[b])
            break
    if pair is None:
        return {"lattice": name, "has_a2_triangle": False}
    w, u = pair
    vertices = [s * v for v in ((2 * w + u) / 3, (w + 2 * u) / 3, (w - u) / 3) for s in (1, -1)]
    worst_vertex = max(float(combigeo.dist_to_halfspaces(v.tolist(), facets)) for v in vertices)
    v3 = 3 * w + u
    measured = 2.0 * float(combigeo.dist_to_halfspaces((v3 / 2).tolist(), facets))
    target = SQRT73 * lam1
    record = {
        "lattice": name, "n": int(basis.shape[0]), "lambda1": lam1,
        "has_a2_triangle": True,
        "worst_hexagon_vertex_dist_to_V0": worst_vertex,
        "norm_3w_plus_u_over_lambda1": float(np.linalg.norm(v3)) / lam1,
        "D_3w_plus_u": measured, "sqrt73_lambda1": target,
        "holds": bool(worst_vertex <= 1e-9 and abs(measured - target) <= TOL * max(1.0, target)),
    }
    print(f"triangle {name:26s} hexagon in V0 (max dist {worst_vertex:.1e}), "
          f"D(3w+u)={measured:.12f} vs {target:.12f}", flush=True)
    return record


# --------------------------------------------------------------------------- #
def _An(n: int) -> np.ndarray:
    basis = np.zeros((n, n + 1))
    for i in range(n):
        basis[i, i], basis[i, i + 1] = 1, -1
    return basis


def _Dn(n: int) -> np.ndarray:
    basis = np.zeros((n, n))
    basis[0, 0], basis[0, 1] = 1, 1
    for i in range(1, n):
        basis[i, i - 1], basis[i, i] = 1, -1
    return basis


def _E8() -> np.ndarray:
    basis = _Dn(8).copy()
    basis[0] = np.full(8, 0.5)
    return basis


def _E7() -> np.ndarray:
    basis = _E8()
    v = enumerate_upto(basis, math.sqrt(2) + 1e-9)[0]
    rows: list[np.ndarray] = []
    for r in enumerate_upto(basis, 3.0):
        if abs(float(r @ v)) > 1e-9:
            continue
        if np.linalg.matrix_rank(np.array(rows + [r]), tol=1e-9) == len(rows) + 1:
            rows.append(r)
        if len(rows) == 7:
            break
    return np.array(rows)


def hurwitz_module(vectors) -> np.ndarray:
    generators = []
    for x in vectors:
        flat = np.array([c for quaternion in x for c in quaternion], float)
        for e in HURWITZ_BASIS:
            generators.append(qmul(flat, e))
    return np.array(generators, float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(20260821)

    identity = [
        check_identity("A2", eisenstein_basis([(1,)])),
        check_identity("Z[omega]^2", eisenstein_basis([(1, 0), (0, 1)])),
        check_identity("D4", eisenstein_basis([(1, 1), (THETA, 0)])),
        check_identity("E6*", eisenstein_basis([(1, 1, 1), (THETA, 0, 0), (0, THETA, 0)])),
        check_identity("E8", eisenstein_basis([(1, 1, 1, 0), (0, 1, -1, 1),
                                               (THETA, 0, 0, 0), (0, THETA, 0, 0)])),
        check_identity("A2 + 0.37 A2", eisenstein_basis([(1, 0), (0, 0.37)])),
        check_identity("A2 + 4.1 A2", eisenstein_basis([(1, 0), (0, 4.1)])),
    ]
    for index in range(4):
        rank = 2 if index < 2 else 3
        columns = [tuple(complex(*rng.normal(size=2)) for _ in range(rank))
                   for _ in range(rank)]
        identity.append(check_identity(f"random skew C^{rank} #{index}",
                                       eisenstein_basis(columns)))

    ladder = []
    for name, basis in [("Z: A3", _An(3)), ("Z: D5", _Dn(5)), ("Z: E8", _E8())]:
        ladder.append(check_alpha_ladder(
            name, basis, [lambda v: v],
            [(str(k), (lambda k: (lambda v: k * v))(k)) for k in (2, 3, 4, 5)]))

    gauss = np.random.default_rng(5)
    for name, columns in [
        ("Z[i]: Z[i]^2", [(1, 0), (0, 1)]),
        ("Z[i]: random C^2", [tuple(complex(*gauss.normal(size=2)) for _ in range(2))
                              for _ in range(2)]),
        ("Z[i]: random C^3", [tuple(complex(*gauss.normal(size=2)) for _ in range(3))
                              for _ in range(3)]),
    ]:
        alphas = [("2", 2), ("2+i", 2 + 1j), ("3", 3), ("3+i", 3 + 1j),
                  ("2+2i", 2 + 2j), ("4+i", 4 + 1j)]
        ladder.append(check_alpha_ladder(
            name, complex_basis(columns, 1j),
            [lambda v: v, lambda v: cmul(v, 1j)],
            [(lab, (lambda z: (lambda v: cmul(v, z)))(z)) for lab, z in alphas]))

    eis = np.random.default_rng(3)
    for name, columns in [
        ("Z[omega]: Z[omega]^2", [(1, 0), (0, 1)]),
        ("Z[omega]: random C^3", [tuple(complex(*eis.normal(size=2)) for _ in range(3))
                                  for _ in range(3)]),
    ]:
        alphas = [("2", 2), ("2+w", 2 + OMEGA), ("3", 3), ("3+w", 3 + OMEGA),
                  ("4+w", 4 + OMEGA), ("5+2w", 5 + 2 * OMEGA)]
        ladder.append(check_alpha_ladder(
            name, eisenstein_basis(columns),
            [lambda v: v, lambda v: cmul(v, OMEGA)],
            [(lab, (lambda z: (lambda v: cmul(v, z)))(z)) for lab, z in alphas]))

    quat = np.random.default_rng(17)
    modules = [
        ("H: the order", hurwitz_module([((1, 0, 0, 0),)])),
        ("H: H^2 orthogonal", hurwitz_module([((1, 0, 0, 0), (0, 0, 0, 0)),
                                              ((0, 0, 0, 0), (1.3, 0, 0, 0))])),
        ("H: H^2 skew", hurwitz_module([((1, 0, 0, 0), (0, 0, 0, 0)),
                                        (tuple(quat.normal(size=4)),
                                         tuple(quat.normal(size=4)))])),
    ]
    quaternion_alphas = [("2", (2, 0, 0, 0)), ("2+i+j+k", (2, 1, 1, 1)),
                         ("3", (3, 0, 0, 0)), ("2+i", (2, 1, 0, 0)),
                         ("1+i+j+k", (1, 1, 1, 1)), ("2+2i", (2, 2, 0, 0))]
    for name, basis in modules:
        ladder.append(check_alpha_ladder(
            name, basis,
            [(lambda e: (lambda v: qmul(v, e)))(e) for e in HURWITZ_BASIS],
            [(lab, (lambda a: (lambda v: qmul(v, a)))(a)) for lab, a in quaternion_alphas]))

    triangle = [check_a2_triangle(name, basis) for name, basis in
                [("A3", _An(3)), ("A4", _An(4)), ("A5", _An(5)),
                 ("D5", _Dn(5)), ("E7", _E7())]]

    # the 24 vertices of the 24-cell are midpoints of orthogonal unit pairs
    units = np.array(HURWITZ_UNITS, float)
    cell24 = np.asarray(combigeo.voronoi_cell(np.array(HURWITZ_BASIS, float).tolist()).vertices,
                        float)
    midpoints = sum(
        any(abs(float(units[a] @ units[b])) < 1e-9
            and np.allclose((units[a] + units[b]) / 2, vertex, atol=1e-9)
            for a in range(len(units)) for b in range(a + 1, len(units)))
        for vertex in cell24)
    hurwitz = {"units": len(units), "cell_vertices": len(cell24),
               "vertices_as_orthogonal_unit_midpoints": int(midpoints),
               "holds": bool(midpoints == len(cell24))}
    print(f"hurwitz  24-cell: {midpoints}/{len(cell24)} vertices are (u1+u2)/2, u1 ⟂ u2",
          flush=True)

    payload = {
        "identity": identity, "alpha_ladder": ladder, "a2_triangle": triangle,
        "hurwitz_24cell_vertices": hurwitz,
        "all_hold": bool(all(r["holds"] for r in identity)
                         and all(r["holds"] for r in ladder)
                         and all(r.get("holds", False) for r in triangle)
                         and hurwitz["holds"]),
    }
    out = args.output or results_path("eisenstein_identity_checks.json")
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"saved {out}; ALL {'OK' if payload['all_hold'] else 'FAILED'}", flush=True)
    return 0 if payload["all_hold"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
