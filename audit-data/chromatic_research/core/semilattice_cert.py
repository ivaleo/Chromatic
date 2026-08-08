"""Exact rational certificate for a planar semi-lattice colouring.

Everything is done in *basis coordinates* with a rational Gram matrix ``G``, so
no square root is ever taken: a site is a point of ``Q^2``, the Laguerre
condition ``|x-p|^2 - w_p <= |x-s|^2 - w_s`` reads

    2 (s - p)^T G x  <=  s^T G s - p^T G p - w_s + w_p,

whose coefficients are rational; a cell vertex solves a rational 2x2 system;
``diam^2`` is a maximum of rationals over vertex pairs; and the distance between
two cells is a minimum of rationals over vertex/edge pairs (the projection
parameter ``t = <a-b, c-b> / <c-b, c-b>`` is rational and is clamped by rational
comparisons).  The verdict ``sep^2 / diam^2 >= l^2`` is therefore an exact
comparison of two rationals.

The colouring it certifies: pieces = the cells of ``P = U_i (Gamma + t_i)``,
colour = index of the orbit.  Two points of one colour are either in one cell
(distance <= diam) or in cells ``C, C+v`` with ``v`` a nonzero vector of
``Gamma`` (distance >= sep).  Rescaling so that the cell diameter is just under
1 gives ``chi(R^2, [1, l]) <= N`` for every ``l < sep/diam``.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations


def _ip(G, u, v):
    return (u[0]*(G[0][0]*v[0] + G[0][1]*v[1])
            + u[1]*(G[1][0]*v[0] + G[1][1]*v[1]))


def _sub(u, v):
    return (u[0]-v[0], u[1]-v[1])


def _add(u, v):
    return (u[0]+v[0], u[1]+v[1])


def gamma_coords(radius2, G):
    """All integer coordinate pairs with squared length <= radius2."""
    out = []
    lim = 0
    while _ip(G, (lim, 0), (lim, 0)) <= radius2 * 4 + 4:
        lim += 1
    for i in range(-lim, lim+1):
        for j in range(-lim, lim+1):
            if _ip(G, (i, j), (i, j)) <= radius2:
                out.append((F(i), F(j)))
    return out


def cell_vertices(G, p, wp, sites, weights, keep=22):
    """Vertices of the Laguerre cell of ``p``.

    Only the ``keep`` nearest sites are used to build the polygon (a selection
    step, done in floating point); correctness does not depend on it because
    :func:`certify` afterwards checks *exactly* that every site of the shell
    satisfies its half-plane at every vertex, and that the shell is wide enough.
    """
    rows, rhs, power = [], [], []
    for s_, w in zip(sites, weights):
        d = _sub(s_, p)
        if d == (0, 0) and w == wp:
            continue
        a0 = 2*(G[0][0]*d[0] + G[1][0]*d[1])
        a1 = 2*(G[0][1]*d[0] + G[1][1]*d[1])
        rows.append((a0, a1))
        rhs.append(_ip(G, s_, s_) - _ip(G, p, p) - w + wp)
        power.append(float(_ip(G, d, d) - w + wp))
    order = sorted(range(len(rows)), key=lambda i: power[i])
    sel = order[:keep]
    verts = []
    for ii, jj in combinations(sel, 2):
        (a, b), (c, d) = rows[ii], rows[jj]
        det = a*d - b*c
        if det == 0:
            continue
        x = (rhs[ii]*d - b*rhs[jj]) / det
        y = (a*rhs[jj] - rhs[ii]*c) / det
        if all(rows[k][0]*x + rows[k][1]*y <= rhs[k] for k in sel):
            if all(x != u or y != v for u, v in verts):
                verts.append((x, y))
    if len(verts) < 3:
        return None
    # exact global check: no other site of the shell cuts the polygon
    for k in range(len(rows)):
        for (x, y) in verts:
            if rows[k][0]*x + rows[k][1]*y > rhs[k]:
                return None
    import math
    cx = sum(v[0] for v in verts)/len(verts)
    cy = sum(v[1] for v in verts)/len(verts)
    verts.sort(key=lambda v: math.atan2(float(v[1]-cy), float(v[0]-cx)))
    return verts


def _seg_dist2(G, q, a, b):
    e = _sub(b, a)
    ee = _ip(G, e, e)
    if ee == 0:
        r = _sub(q, a)
        return _ip(G, r, r)
    t = _ip(G, _sub(q, a), e) / ee
    if t < 0:
        t = F(0)
    elif t > 1:
        t = F(1)
    r = _sub(q, (a[0] + t*e[0], a[1] + t*e[1]))
    return _ip(G, r, r)


def _meets(G, P, Q):
    """True if the convex polygons overlap (separating-axis test, exact)."""
    for R, S in ((P, Q), (Q, P)):
        m = len(R)
        for i in range(m):
            a, b = R[i], R[(i+1) % m]
            e = _sub(b, a)
            nrm = (-(G[0][1]*e[0] + G[1][1]*e[1]), G[0][0]*e[0] + G[1][0]*e[1])
            vr = [nrm[0]*v[0] + nrm[1]*v[1] for v in R]
            vs = [nrm[0]*v[0] + nrm[1]*v[1] for v in S]
            if max(vr) < min(vs) or max(vs) < min(vr):
                return False
    return True


def poly_dist2(G, P, Q):
    if _meets(G, P, Q):
        return F(0)
    best = None
    for R, S in ((P, Q), (Q, P)):
        m = len(S)
        for q in R:
            for i in range(m):
                d2 = _seg_dist2(G, q, S[i], S[(i+1) % m])
                if best is None or d2 < best:
                    best = d2
    return best


def certify(G, ts, ws, nb_shells=3):
    """Exact ``diam^2``, ``sep^2`` and the tiling check.  All inputs rational.

    ``G`` is the Gram matrix of ``Gamma`` (rows of the basis), ``ts`` the sites
    in basis coordinates, ``ws`` the Laguerre weights.
    """
    G = [[F(G[0][0]), F(G[0][1])], [F(G[1][0]), F(G[1][1])]]
    ts = [(F(t[0]), F(t[1])) for t in ts]
    ws = [F(w) for w in ws]
    N = len(ts)
    # reduce every site modulo Gamma into (a small neighbourhood of) its Voronoi
    # cell -- exact, and it keeps the shell needed for the proof small
    def reduce_mod(t):
        best = t
        for _ in range(6):
            cur = best
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)):
                cand = (cur[0]+dx, cur[1]+dy)
                if _ip(G, cand, cand) < _ip(G, best, best):
                    best = cand
            if best == cur:
                break
        return best
    ts = [reduce_mod(t) for t in ts]
    lam2 = min(_ip(G, (1, 0), (1, 0)), _ip(G, (0, 1), (0, 1)),
               _ip(G, (1, -1), (1, -1)), _ip(G, (1, 1), (1, 1)))
    shifts = gamma_coords(lam2 * nb_shells * nb_shells, G)
    shell_r2 = lam2 * nb_shells * nb_shells
    sites, weights = [], []
    for sh in shifts:
        for t, w in zip(ts, ws):
            sites.append(_add(t, sh))
            weights.append(w)
    cells = []
    for i in range(N):
        V = cell_vertices(G, ts[i], ws[i], sites, weights)
        if V is None:
            return None
        cells.append(V)
    diam2 = max(_ip(G, _sub(u, v), _sub(u, v))
                for V in cells for u, v in combinations(V, 2))
    # the shell must be wide enough: a site farther than 2*max|x-p| + slack
    # cannot cut a cell, so check that the shell covers that radius
    rad2 = max(_ip(G, _sub(u, t), _sub(u, t)) for V, t in zip(cells, ts) for u in V)
    wspread = max(ws) - min(ws)
    tmax2 = max(_ip(G, t, t) for t in ts)
    # a site t_j + shift can cut the cell of t_i only if it lies within
    # 2*rad + |t_i - t_j| of t_i; with 2ab <= a^2 + b^2 a sufficient shell is
    need = 8*rad2 + 8*tmax2 + 4*wspread
    if shell_r2 < need:
        return {"error": "shell too small", "shell_r2": shell_r2, "need": need}
    # every Gamma-vector that could come closer than diam has |v| <= 2*diam
    gam = [v for v in gamma_coords(4*diam2 + 4*lam2, G) if v != (0, 0)]
    sep2 = None
    for V in cells:
        for v in gam:
            W = [_add(u, v) for u in V]
            d2 = poly_dist2(G, V, W)
            if sep2 is None or d2 < sep2:
                sep2 = d2
    # tiling check: the cell areas must add up to the covolume of Gamma
    detG = G[0][0]*G[1][1] - G[0][1]*G[1][0]
    area2 = sum(abs(sum(V[i][0]*V[(i+1) % len(V)][1] - V[(i+1) % len(V)][0]*V[i][1]
                        for i in range(len(V)))) for V in cells)
    tiling_ok = (area2 / 2) == 1                     # in basis coordinates, area 1
    return {"diam2": diam2, "sep2": sep2, "d2": sep2/diam2,
            "tiling_exact": tiling_ok, "detG": detG,
            "n_vertices": [len(V) for V in cells]}
