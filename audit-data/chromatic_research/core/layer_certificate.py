"""Diameter certificate for a lamination whose layer group is two-dimensional.

:func:`chromatic_research.core.piecewise_covrad.certify_two_layer` bounds the
covering radius of a lamination by a *line* of layers.  A two-dimensional layer
lattice seems to need the common refinement of three shifted Voronoi tilings of
the base -- in dimension 8 that is a vertex enumeration over ~10^5 pieces.  The
following reduction avoids it entirely.

**Reduction.**  Let ``T`` be a Delaunay triangle of the layer lattice ``L``, with
vertices ``v_1, v_2, v_3``, circumradius ``R_L`` and side ``s``.  Every point of
the layer plane lies in such a triangle, so

    R(Lambda)^2 = max_{y,z} min_k [ f(y - c_k) + |z - v_k|^2 ],
    f(u) = dist(u, base)^2,

with ``k`` ranging over the three vertices (restricting the minimum to a subset
only raises it, so this is an upper bound).  Now fix the pair ``(k,l)`` of
vertices nearest to ``z`` and split ``z`` into its component along the line
``v_k v_l`` and the perpendicular one:

    min_k [\\cdot] \\le min(A_k, A_l) = |z_perp|^2
                    + min( f(y-c_k) + z_par^2, f(y-c_l) + (z_par - s)^2 ).

The second term is *exactly* the one-dimensional two-layer problem with offset
``c_l - c_k`` and height ``s``.  The first is bounded on the region where
``v_k, v_l`` are the two nearest vertices: that region is the triangle
``v_k v_l o`` (``o`` the circumcentre), whose greatest distance from the line
``v_k v_l`` is the inradius ``R_L/2``.  Hence

    R(Lambda)^2  <=  max_{(k,l)} TwoLayer(base, c_l - c_k, s)  +  R_L^2/4.

**Sharpness.**  For zero offsets the bound is an identity: ``TwoLayer`` returns
``R_0^2 + s^2/4`` and, for the hexagonal layer with ``s = a``, ``R_L = a/sqrt3``,
the total is ``R_0^2 + a^2/4 + a^2/12 = R_0^2 + R_L^2`` -- exactly the
orthogonal-product value.  So nothing is lost in the reduction itself; whatever
the two-layer certificate gains over ``R_0^2 + s^2/4`` is gained here too.

**Equivariance.**  For the ``Z[omega]``-equivariant construction the three
offsets of a Delaunay triangle are ``0``, ``g`` and ``(1+omega)g``, so the three
pairwise differences are ``g``, ``omega g`` and ``-omega^2 g``: all three are
unit multiples of ``g``, hence isometric copies of one another, and a single
call to the two-layer certificate covers all three edges.
"""

from __future__ import annotations

import math

import numpy as np

from chromatic_research.core.piecewise_covrad import certify_two_layer


def hexagonal_layer_geometry(scale: float) -> dict:
    """Side, circumradius and perpendicular allowance of a Delaunay triangle
    of the hexagonal layer lattice ``scale * A2``."""
    return dict(side=scale, circumradius=scale / math.sqrt(3.0),
                perp_allowance_sq=scale * scale / 12.0)


def certify_hex_layer(base: np.ndarray, offset: np.ndarray, scale: float, *,
                      base_covering_radius: float, verbose: bool = False) -> dict:
    """Rigorous upper bound on ``R^2`` of a base laminated by ``scale * A2``.

    ``offset`` is the base-space offset ``g`` of the first layer generator; the
    construction is assumed ``Z[omega]``-equivariant, so the three edges of the
    Delaunay triangle are isometric and one two-layer certificate suffices.
    """
    geometry = hexagonal_layer_geometry(scale)
    inner = certify_two_layer(base, np.asarray(offset, float), geometry["side"],
                              covering_radius=base_covering_radius,
                              verbose=verbose)
    bound = inner["certified_r2_upper"] + geometry["perp_allowance_sq"]
    return dict(
        R2_upper=bound,
        diam_upper=2.0 * math.sqrt(bound),
        two_layer_R2=inner["certified_r2_upper"],
        perp_allowance_sq=geometry["perp_allowance_sq"],
        product_R2=base_covering_radius ** 2 + geometry["circumradius"] ** 2,
        sound=bool(inner.get("sound", True)),
        pieces=inner.get("pieces"),
        **{k: v for k, v in inner.items()
           if k in ("qhull_failures", "seconds", "worst_piece")},
    )


def _coordinate_box(A: np.ndarray, b: np.ndarray):
    """Exact per-coordinate range of ``{x : A x <= b}`` by 2n linear programs."""
    from scipy.optimize import linprog
    n = A.shape[1]
    lo, hi = np.zeros(n), np.zeros(n)
    for i in range(n):
        c = np.zeros(n); c[i] = 1.0
        r1 = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        r2 = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * n, method="highs")
        if not (r1.success and r2.success):
            return None, None
        lo[i], hi[i] = r1.x[i], r2.x[i]
    return lo, hi


def certify_two_layer_window(base: np.ndarray, offset: np.ndarray, height: float,
                             half_window: float, *, covering_radius: float,
                             slack: float = 1e-6) -> dict:
    """Two-layer certificate with ``z`` restricted to ``[t/2 - w, t/2 + w]``.

    The unrestricted routine maximises over the whole interval ``[0, t]``, which
    throws away the fact that the admissible band narrows as one moves away from
    the edge of the Delaunay triangle.  Here ``z`` runs over a centred window.
    The minimum ``min(A0 + z^2, A1 + (z-t)^2)`` rises to the crossing
    ``z* = (A1 - A0 + t^2)/(2t)`` and falls after it, so its maximum over the
    window is attained at ``clamp(z*, lo, hi)``; ``z*`` is affine in ``x`` (the
    quadratic parts of ``A1 - A0`` cancel), so each of the three cases carves a
    polyhedral region out of the piece and the objective stays a convex
    quadratic -- the vertex maximum is again exact.
    """
    from scipy.spatial import HalfspaceIntersection

    from chromatic_research.core.lamination import enumerate_upto
    from chromatic_research.core.piecewise_covrad import (
        _chebyshev_center, _halfspaces,
    )

    base = np.asarray(base, float)
    offset = np.asarray(offset, float)
    t = float(height)
    lo, hi = t / 2.0 - half_window, t / 2.0 + half_window
    normals, offsets = _halfspaces(base)

    reach = 2.0 * covering_radius + 1e-9
    candidates = [np.zeros(base.shape[1])]
    for point in enumerate_upto(base, reach + float(np.linalg.norm(offset)) + 1e-9):
        if np.linalg.norm(point + offset) <= reach:
            candidates.append(point)

    worst, failures, pieces, inflated, boxed = 0.0, [], 0, [], []
    for m in candidates:
        shift = offset + m
        s2 = float(shift @ shift)
        A = np.vstack([normals, normals])
        b = np.concatenate([offsets, offsets + normals @ shift])
        # z*(x) = (ell(x) + t^2) / (2t) with ell(x) = -2<x, shift> + s2
        ell_n, ell_c = -2.0 * shift, s2
        # z* <= hi  <=>  ell(x) <= 2 t hi - t^2 ;  z* >= lo  <=>  ell(x) >= 2 t lo - t^2
        up, dn = 2 * t * hi - t * t, 2 * t * lo - t * t
        nrm = float(np.linalg.norm(ell_n)) + 1e-300
        regions = (
            ("mid", [(ell_n / nrm, (up - ell_c) / nrm),
                     (-ell_n / nrm, (ell_c - dn) / nrm)]),
            ("hi", [(-ell_n / nrm, (ell_c - up) / nrm)]),
            ("lo", [(ell_n / nrm, (dn - ell_c) / nrm)]),
        )
        touched = False
        for name, extra in regions:
            A_r = np.vstack([A] + [row[None, :] for row, _ in extra])
            b_r = np.concatenate([b, [off for _, off in extra]])
            centre, radius = _chebyshev_center(A_r, b_r)
            if radius < 1e-9:
                continue
            verts, used = None, 0.0
            for inflate in (0.0, 1e-9, 1e-7, 1e-5, 1e-3):
                # inflating the region OUTWARD can only raise the maximum, so the
                # bound stays valid; it also removes the degeneracy that makes
                # qhull report a flat initial simplex or a wide merge
                try:
                    verts = HalfspaceIntersection(
                        np.hstack([A_r, -(b_r + inflate)[:, None]]),
                        centre).intersections
                    used = inflate
                    break
                except Exception as error:          # noqa: BLE001
                    last = str(error)[:120]
            if verts is None:
                # Fallback that never fails: bound the convex quadratic over the
                # coordinate box of the region.  Each coordinate range comes from
                # an exact LP, the box contains the region, and the objective is
                # a sum of a squared norm and a squared affine form -- both are
                # bounded on a box in closed form.  Lossy but rigorous, so a
                # region qhull cannot enumerate never silently disappears.
                lo_i, hi_i = _coordinate_box(A_r, b_r)
                if lo_i is None:
                    failures.append({"m": m.tolist(), "region": name,
                                     "radius": radius, "error": last})
                    continue
                sq = float(np.sum(np.maximum(lo_i ** 2, hi_i ** 2)))
                if name == "mid":
                    lin = -2.0 * shift
                    ends = np.where(lin > 0, hi_i, lo_i) @ lin
                    other = np.where(lin > 0, lo_i, hi_i) @ lin
                    m_abs = max(abs(ends + s2 + t * t), abs(other + s2 + t * t))
                    value = sq + (m_abs / (2 * t)) ** 2
                elif name == "hi":
                    value = sq + hi * hi
                else:
                    d_lo, d_hi = lo_i - shift, hi_i - shift
                    value = float(np.sum(np.maximum(d_lo ** 2, d_hi ** 2))) \
                        + (lo - t) ** 2
                boxed.append({"m": m.tolist(), "region": name, "value": value})
                worst = max(worst, value)
                pieces += 1
                continue
            if used:
                inflated.append({"m": m.tolist(), "region": name, "delta": used})
            touched = True
            A0 = np.einsum("vd,vd->v", verts, verts)
            d = verts - shift[None, :]
            A1 = np.einsum("vd,vd->v", d, d)
            if name == "mid":
                values = A0 + ((A1 - A0 + t * t) / (2 * t)) ** 2
            elif name == "hi":                      # z* above the window
                values = A0 + hi * hi
            else:                                   # z* below the window
                values = A1 + (lo - t) ** 2
            worst = max(worst, float(np.max(values)))
        pieces += touched
    return dict(certified_r2_upper=worst + slack, pieces=pieces,
                qhull_failures=failures, inflated=inflated, boxed=boxed,
                sound=not failures)


def certify_hex_layer_strips(base: np.ndarray, offset: np.ndarray, scale: float, *,
                             base_covering_radius: float, strips: int = 6) -> dict:
    """Sharpened bound: the perpendicular offset and the along-edge window are
    coupled, not maximised independently.

    On the sub-triangle where ``v_k, v_l`` are the two nearest vertices, a point
    at perpendicular distance ``h`` from the edge has its along-edge coordinate
    confined to a window of half-width ``w(h) = (s/2)(1 - 2h/R_L)``, shrinking to
    a point at the circumcentre.  Partitioning ``h in [0, R_L/2]`` into strips
    and bounding each by ``h_max^2 + M(w(h_min))`` -- ``M`` non-decreasing in the
    window -- keeps the estimate rigorous while recovering most of the slack
    that :func:`certify_hex_layer` gives away.
    """
    geometry = hexagonal_layer_geometry(scale)
    side, r_l = geometry["side"], geometry["circumradius"]
    edges = np.linspace(0.0, r_l / 2.0, strips + 1)
    worst, sound, detail, inflations, boxed_total = 0.0, True, [], 0, 0
    for i in range(strips):
        h_lo, h_hi = float(edges[i]), float(edges[i + 1])
        window = (side / 2.0) * (1.0 - 2.0 * h_lo / r_l)
        inner = certify_two_layer_window(base, offset, side, window,
                                         covering_radius=base_covering_radius)
        value = h_hi * h_hi + inner["certified_r2_upper"]
        sound = sound and inner["sound"]
        inflations += len(inner.get("inflated", []))
        boxed_total += len(inner.get("boxed", []))
        detail.append(dict(h_lo=h_lo, h_hi=h_hi, window=window,
                           two_layer=inner["certified_r2_upper"], value=value))
        worst = max(worst, value)
    return dict(R2_upper=worst, diam_upper=2.0 * math.sqrt(worst), sound=sound,
                inflated_regions=inflations, boxed_regions=boxed_total,
                strips=detail,
                product_R2=base_covering_radius ** 2 + r_l ** 2)
