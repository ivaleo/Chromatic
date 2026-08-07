"""Exact covering-radius certificate for layered lattices via the two-tiling
arrangement (no branch-and-bound).

Setting: layers of a base lattice ``L'`` at height ``t``, offset ``c``.  With
the minimum restricted to layers ``{0, 1}`` (a conservative upper bound),

    R^2 <= max_x  U(x),   U(x) = max_{z in [0,t]} min( |x-l|^2 + z^2,
                                                       |x-c-m|^2 + (z-t)^2 ),

where ``l, m`` are the nearest base points in the two layers.  Key facts:

- on a *piece* ``P_m = V(0) \\cap (V(m) + c)`` of the common refinement of the
  two shifted Voronoi tilings both nearest points are constant (``l = 0``), so
  ``A0 = |x|^2`` and ``A1 = |x - c - m|^2`` are quadratics whose difference is
  LINEAR in ``x``;
- ``U(x) <= phi(x) := A0 + ((A1 - A0 + t^2) / (2t))^2`` for every ``x`` (when
  the parabola crossing leaves ``[0, t]`` the endpoint value is still below
  ``phi``), and ``phi`` is a CONVEX quadratic on ``P_m``;
- hence ``max_{P_m} U <= max over the VERTICES of P_m of phi`` -- an exact
  finite computation, rationalisable later.

The pieces cover a fundamental domain (``x in V(0)``), and only ``m`` with
``|c + m| <= 2R + eps`` give nonempty pieces.  Everything reduces to one
vertex enumeration per piece (qhull) plus a handful of dot products.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

from chromatic_research.core.lamination import enumerate_upto, unit_facets


def _halfspaces(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    facets = unit_facets(basis)
    normals = np.array([f[0] for f in facets], dtype=float)
    offsets = np.array([f[1] for f in facets], dtype=float)
    return normals, offsets


def _chebyshev_center(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Largest inscribed ball: max r s.t. A x + r <= b (A rows unit-norm)."""
    n = A.shape[1]
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([A, np.ones((len(A), 1))])
    res = linprog(c, A_ub=A_ub, b_ub=b, bounds=[(None, None)] * n + [(0, None)],
                  method="highs")
    if not res.success:
        return np.zeros(n), -1.0
    return res.x[:n], float(res.x[n])


def certify_two_layer(base: np.ndarray, offset: np.ndarray, height: float,
                      *, covering_radius: float, slack: float = 1e-6,
                      verbose: bool = True) -> dict:
    """Exact upper bound on R^2 of the layered lattice (two-layer restriction).

    ``covering_radius`` is the EXACT covering radius of the base lattice; it is
    used only to enumerate candidate pieces (``|c + m| <= 2R + margin``), so a
    safe upper estimate is fine.
    """
    start = time.time()
    base = np.asarray(base, dtype=float)
    offset = np.asarray(offset, dtype=float)
    t = float(height)
    normals, offsets = _halfspaces(base)

    reach = 2.0 * covering_radius + 1e-9
    candidates = [np.zeros(base.shape[1])]
    shifted = enumerate_upto(base, reach + np.linalg.norm(offset) + 1e-9)
    for point in shifted:
        if np.linalg.norm(point + offset) <= reach:
            candidates.append(point)

    worst = 0.0
    worst_piece = None
    n_pieces = 0
    n_degenerate = 0
    qhull_failures: list[dict] = []
    for m in candidates:
        shift = offset + m                     # A1 = |x - shift|^2
        A = np.vstack([normals, normals])
        b = np.concatenate([offsets, offsets + normals @ shift])
        # A1 - A0 = -2 <x, shift> + |shift|^2  is LINEAR; the position of the
        # parabola crossing z* = (A1 - A0 + t^2)/(2t) relative to [0, t] splits
        # the piece into three regions with CONVEX exact objectives:
        #   |A1-A0| <= t^2  ->  U = phi = A0 + ((A1-A0+t^2)/(2t))^2
        #    A1-A0  >= t^2  ->  U = A0 + t^2        (layer 1 dominated)
        #    A1-A0  <= -t^2 ->  U = A1 + t^2        (layer 0 dominated)
        # Each region is the piece cut by one or two extra halfspaces, so the
        # vertex maximum stays exact.
        s2 = float(shift @ shift)
        t2 = t * t
        ell_normal = -2.0 * shift              # ell(x) = ell_normal.x + s2
        norm_ell = np.linalg.norm(ell_normal) + 1e-300
        cuts = (
            ("mid", [(ell_normal / norm_ell, (t2 - s2) / norm_ell),
                     (-ell_normal / norm_ell, (t2 + s2) / norm_ell)]),
            ("hi", [(-ell_normal / norm_ell, (s2 - t2) / norm_ell)]),
            ("lo", [(ell_normal / norm_ell, (-t2 - s2) / norm_ell)]),
        )
        found_any = False
        for region, extra in cuts:
            A_r = np.vstack([A] + [row[None, :] for row, _ in extra])
            b_r = np.concatenate([b, [off for _, off in extra]])
            center, radius = _chebyshev_center(A_r, b_r)
            if radius < 1e-9:
                continue
            try:
                hs = HalfspaceIntersection(np.hstack([A_r, -b_r[:, None]]), center)
                vertices = hs.intersections
            except Exception as error:         # noqa: BLE001
                # a FULL-dimensional region that qhull cannot enumerate would be
                # a hole in the covering -- the certificate must not stay silent
                qhull_failures.append({"m": m.tolist(), "region": region,
                                       "radius": radius, "error": str(error)[:120]})
                continue
            found_any = True
            A0 = np.einsum("vd,vd->v", vertices, vertices)
            diff = vertices - shift[None, :]
            A1 = np.einsum("vd,vd->v", diff, diff)
            if region == "mid":
                values = A0 + ((A1 - A0 + t2) / (2 * t)) ** 2
            elif region == "hi":
                values = A0 + t2
            else:
                values = A1 + t2
            value = float(np.max(values))
            if value > worst:
                worst = value
                worst_piece = {"m": m.tolist(), "region": region, "value": value,
                               "vertex": vertices[int(np.argmax(values))].tolist()}
        if found_any:
            n_pieces += 1
        else:
            n_degenerate += 1
        if verbose and n_pieces % 100 == 0 and n_pieces:
            print(f"    [{n_pieces} pieces, worst {worst:.6f}, "
                  f"{time.time() - start:.0f}s]", flush=True)

    r2 = worst + slack
    return {
        "certified_r2_upper": r2,
        "certified_diam_upper": 2.0 * math.sqrt(r2),
        "pieces": n_pieces, "degenerate": n_degenerate,
        "candidates": len(candidates),
        "worst_piece": worst_piece,
        "qhull_failures": qhull_failures,
        "sound": not qhull_failures,
        "seconds": round(time.time() - start, 1),
    }
