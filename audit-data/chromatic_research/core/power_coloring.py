"""Colourings by periodic power (Laguerre) tilings -- the non-lattice generalisation.

Every colouring used in this project so far has the same shape: pick a parent
lattice ``L``, cut space into the Voronoi cells of ``L``, and give the cell
``V0 + u`` the colour ``[u] in L/G``.  Admissibility is ``D(v) >= diam(V0)`` for
all ``v`` in ``G``, with ``D(v) = 2 dist(v/2, V0)``.

That scheme is a *special case* of the following one, which needs no parent
lattice at all.

    Let ``G`` be a lattice, ``t_1..t_k`` points of ``R^n`` and ``w_1..w_k`` real
    weights.  Give the site ``t_i + g`` (``g in G``) the weight ``w_i`` and take
    the power diagram of this ``G``-periodic site set.  Its cells are
    ``P_i + g``, they tile ``R^n``, and colouring a cell by the index ``i`` of
    its site uses ``k`` colours.

Two points of one colour lie either in one cell (distance ``<= diam P_i``) or in
``P_i + g`` and ``P_i + g'``; both cells are convex, so the distances they
realise fill the interval ``[dist(g - g', P_i - P_i), ...]``.  Hence with

    Delta = max_i diam(P_i),      gap = min_i min_{0 != g in G} dist(g, P_i - P_i)

the colouring omits every distance in ``(Delta, gap)``, i.e. after scaling

    chi(R^n, [1, gap/Delta]) <= k       whenever   d := gap/Delta >= 1.       (*)

Taking ``G subset L``, the ``t_i`` a transversal of ``L/G`` and all ``w_i = 0``
gives back the Voronoi scheme exactly: the power diagram is the Voronoi diagram
of ``L``, ``P_i - P_i = 2 V0`` and ``dist(g, 2 V0) = 2 dist(g/2, V0) = D(g)``.
So (*) contains the whole project as the ``w = 0``, ``T = L/G`` slice, and the
extra freedom is genuinely non-lattice: the sites need not form a group.

Why this can win.  The binding constraint is not the *volume* of a cell but its
*width* in the directions of the short vectors of ``G``: ``dist(g, P - P) >=
|g| - width(P, g/|g|)``.  A Voronoi cell of a lattice is forced to be the same
shape for every colour and to be "round" (it is a fundamental domain of a group
that contains ``G``); power cells may be anisotropic, may differ from colour to
colour, and are flattened exactly along the dangerous directions.

Everything reported by :func:`evaluate` is a *certified lower* bound for ``gap``
(a separating direction is exhibited for every pair) and the *exact* diameter of
each cell (maximum over vertex pairs of a polytope), so ``d`` is never
optimistic.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


# --------------------------------------------------------------------------- #
# lattice point enumeration                                                    #
# --------------------------------------------------------------------------- #

def lll_reduce(basis: np.ndarray, delta: float = 0.99) -> np.ndarray:
    """Plain LLL on the rows; keeps the lattice, shrinks the coefficient box."""
    B = np.array(basis, float, copy=True)
    n = B.shape[0]

    def gso(B):
        Bs = np.zeros_like(B)
        mu = np.zeros((n, n))
        for i in range(n):
            Bs[i] = B[i]
            for j in range(i):
                nj = Bs[j] @ Bs[j]
                mu[i, j] = (B[i] @ Bs[j]) / nj if nj > 1e-300 else 0.0
                Bs[i] -= mu[i, j] * Bs[j]
        return Bs, mu

    Bs, mu = gso(B)
    kk = 1
    guard = 0
    while kk < n and guard < 4000:
        guard += 1
        for j in range(kk - 1, -1, -1):
            if abs(mu[kk, j]) > 0.5:
                B[kk] -= round(mu[kk, j]) * B[j]
                Bs, mu = gso(B)
        if Bs[kk] @ Bs[kk] >= (delta - mu[kk, kk - 1] ** 2) * (Bs[kk - 1] @ Bs[kk - 1]):
            kk += 1
        else:
            B[[kk, kk - 1]] = B[[kk - 1, kk]]
            Bs, mu = gso(B)
            kk = max(kk - 1, 1)
    return B


def lattice_points(basis: np.ndarray, radius: float) -> np.ndarray:
    """All lattice points of norm ``<= radius`` (origin included), rows."""
    B = lll_reduce(np.asarray(basis, float))
    n = B.shape[0]
    # box of integer coefficients that certainly covers the ball
    dual = np.linalg.inv(B).T
    bounds = [int(math.floor(radius * np.linalg.norm(dual[i]) + 1e-9)) for i in range(n)]
    grids = np.meshgrid(*[np.arange(-b, b + 1) for b in bounds], indexing="ij")
    coeff = np.stack([g.ravel() for g in grids], axis=1).astype(float)
    pts = coeff @ B
    return pts[np.einsum("ij,ij->i", pts, pts) <= radius * radius + 1e-12]


# --------------------------------------------------------------------------- #
# one power cell                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Cell:
    """One power cell ``P_i``: half spaces ``A x <= b`` plus its vertices."""

    index: int
    site: np.ndarray
    A: np.ndarray
    b: np.ndarray
    vertices: np.ndarray
    circumradius: float                      # max |vertex - site|
    diameter: float
    volume: float = float("nan")
    complete: bool = True                    # neighbour enumeration provably enough


def _chebyshev_center(A: np.ndarray, b: np.ndarray):
    """Interior point of ``{A x <= b}`` and its inradius (``None`` if empty)."""
    n = A.shape[1]
    norms = np.linalg.norm(A, axis=1)
    # maximise r  s.t.  A x + r |a_j| <= b
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub = np.hstack([A, norms.reshape(-1, 1)])
    res = linprog(c, A_ub=A_ub, b_ub=b,
                  bounds=[(None, None)] * n + [(None, None)], method="highs")
    if not res.success:
        return None, -1.0
    return res.x[:n], float(res.x[-1])


def neighbourhood(sites: np.ndarray, weights: np.ndarray, shifts: np.ndarray):
    """All translated sites ``t_j + g`` with their weights, as flat arrays."""
    k, n = sites.shape
    S = (sites[:, None, :] + shifts[None, :, :]).reshape(-1, n)
    W = np.repeat(weights, shifts.shape[0])
    return S, W, (S * S).sum(axis=1)


def power_cell(sites: np.ndarray, weights: np.ndarray, cloud, i: int,
               enum_radius: float) -> Cell | None:
    """Power cell of site ``i`` against every translated site in ``cloud``."""
    S, W, S2 = cloud
    n = sites.shape[1]
    t_i, w_i = sites[i], weights[i]

    delta = S - t_i
    rho2 = np.einsum("ij,ij->i", delta, delta)
    if int(np.sum(rho2 <= 1e-12)) > 1:
        return None                                  # coincident sites: invalid
    keep = (rho2 > 1e-12) & (rho2 <= enum_radius * enum_radius)
    if not np.any(keep):
        return None
    # |x - t_i|^2 - w_i <= |x - s|^2 - w_s  <=>  2<x, s - t_i> <= |s|^2 - |t_i|^2 - w_s + w_i
    A = 2.0 * delta[keep]
    b = S2[keep] - float(t_i @ t_i) - W[keep] + w_i

    x0, inr = _chebyshev_center(A, b)
    if x0 is None or inr <= 1e-9:
        return "empty"                                # buried site: colour unused

    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    try:
        hs = HalfspaceIntersection(halfspaces, x0)
    except Exception:
        return None
    V = np.unique(np.round(hs.intersections, 10), axis=0)
    if V.shape[0] < n + 1:
        return None

    rc = float(np.max(np.linalg.norm(V - t_i, axis=1)))
    diffs = V[:, None, :] - V[None, :, :]
    diam = float(np.sqrt(np.max(np.einsum("ijk,ijk->ij", diffs, diffs))))

    span = float(weights.max() - weights.min())
    complete = (enum_radius * enum_radius - span) / (2.0 * enum_radius) > rc

    return Cell(index=i, site=t_i, A=A, b=b, vertices=V,
                circumradius=rc, diameter=diam, complete=complete)


# --------------------------------------------------------------------------- #
# certified distance from a vector to the difference body of a cell            #
# --------------------------------------------------------------------------- #

def certified_gap(V: np.ndarray, g: np.ndarray, iters: int = 50,
                  enough: float = math.inf) -> float:
    """``dist(g, P - P)`` for ``P = conv(V)``, as a *certified lower* bound.

    ``dist(g, P - P) = min { |y| : y in conv(Q) }`` with ``Q = {g - a + b}``
    over vertex pairs ``a, b`` of ``P``.  Wolfe's minimum-norm-point algorithm
    solves this in finitely many steps; the linear oracle over the (quadratic
    size, never materialised) set ``Q`` is cheap because

        argmin_{q in Q} <q, x> = g - argmax_a <a, x> + argmin_b <b, x>.

    For any iterate ``x in conv(Q)`` and its oracle answer ``q``, convexity
    gives ``|y*| >= <q, x> / |x|``: every value returned is a valid lower bound
    for the true distance, so the width it feeds is never optimistic.  For a
    unit ``e = x / |x|`` this is exactly the separating-slab bound
    ``<g, e> - width(P, e)``.
    """
    def oracle(x):
        proj = V @ x
        return g - V[int(np.argmax(proj))] + V[int(np.argmin(proj))]

    x = oracle(g)
    S = [x.copy()]
    lam = np.array([1.0])
    best = -math.inf
    for _ in range(iters):
        nx = float(np.linalg.norm(x))
        if nx < 1e-12:
            return 0.0
        q = oracle(x)
        best = max(best, float(q @ x) / nx)
        if best >= enough or float(q @ x) >= nx * nx - 1e-12 * max(1.0, nx * nx):
            return max(best, 0.0)
        S.append(q.copy())
        lam = np.append(lam, 0.0)
        for _ in range(64):                       # minor loop of Wolfe
            M = np.array(S)
            m = M.shape[0]
            # min-norm point of the affine hull of S, in barycentric coordinates
            K = np.ones((m + 1, m + 1))
            K[:m, :m] = M @ M.T
            K[m, m] = 0.0
            rhs = np.zeros(m + 1)
            rhs[m] = 1.0
            try:
                alpha = np.linalg.solve(K, rhs)[:m]
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, rhs, rcond=None)[0][:m]
            if np.all(alpha > 1e-12):
                lam, x = alpha, alpha @ M
                break
            neg = alpha <= 1e-12
            ratios = lam[neg] / np.maximum(lam[neg] - alpha[neg], 1e-300)
            theta = float(min(1.0, ratios.min()))
            lam = lam + theta * (alpha - lam)
            keep = lam > 1e-12
            if not np.any(keep):
                keep = np.zeros(m, bool)
                keep[int(np.argmax(lam))] = True
            S = [S[i] for i in range(m) if keep[i]]
            lam = lam[keep]
            lam = lam / lam.sum()
            x = lam @ np.array(S)
    return max(best, 0.0)


# --------------------------------------------------------------------------- #
# full evaluation                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    width: float                       # d = gap / diameter, admissible iff >= 1
    gap: float
    diameter: float
    colours: int
    cells: list = field(default_factory=list)
    worst: tuple = ()
    sound: bool = True
    reason: str = ""


def evaluate(gbasis, sites, weights, *, enum_mult: float = 3.0,
             gap_mult: float = 3.0, want: float = 0.0) -> Report:
    """Certified width ``d`` of the power colouring ``(G, T, w)``.

    ``want`` allows early exit: as soon as the running bound cannot reach
    ``want`` the evaluation stops (used inside optimisation loops).
    """
    G = np.asarray(gbasis, float)
    T = np.atleast_2d(np.asarray(sites, float))
    w = np.asarray(weights, float).ravel()
    k, n = T.shape
    det = abs(float(np.linalg.det(G)))
    scale = (det / k) ** (1.0 / n)

    reach = 2.0 * float(np.max(np.linalg.norm(T, axis=1))) if k else 0.0
    for attempt in range(4):
        radius = enum_mult * scale * (1.6 ** attempt)
        # a neighbour of site i is t_j + g with |t_j + g - t_i| <= radius, so the
        # shift g may have to reach |t_i - t_j| further than the cell itself
        shifts = lattice_points(G, radius + reach)
        cloud = neighbourhood(T, w, shifts)
        cells = []
        ok = True
        for i in range(k):
            c = power_cell(T, w, cloud, i, radius)
            if c is None:
                return Report(0.0, 0.0, 0.0, k, sound=False,
                              reason=f"site {i} coincides with another")
            if c == "empty":
                continue                     # buried site: that colour is unused
            if not c.complete:
                ok = False
                break
            cells.append(c)
        if ok and cells:
            break
    else:
        return Report(0.0, 0.0, 0.0, k, sound=False,
                      reason="neighbour enumeration did not close")

    used = len(cells)                     # buried sites do not consume a colour
    diameter = max(c.diameter for c in cells)
    if diameter <= 0:
        return Report(0.0, 0.0, 0.0, used, sound=False, reason="null diameter")

    # Short vectors of G.  dist(g, P - P) >= |g| - diam(P), so once the running
    # minimum is below |g| - diam no longer vector can bind: sort and break.
    # P - P is symmetric, so only one of +-g is needed.
    gammas = lattice_points(G, gap_mult * diameter)
    norms = np.linalg.norm(gammas, axis=1)
    gammas = gammas[norms > 1e-9]
    norms = norms[norms > 1e-9]
    order = np.argsort(norms)
    gammas, norms = gammas[order], norms[order]
    seen, uniq = set(), []
    for idx, g in enumerate(gammas):
        key = tuple(np.round(g, 7))
        if tuple(-x for x in key) in seen:
            continue
        seen.add(key)
        uniq.append(idx)
    gammas, norms = gammas[uniq], norms[uniq]

    gap = math.inf
    worst = ()
    for c in cells:
        for g, nrm in zip(gammas, norms):
            if nrm - c.diameter >= gap:
                break                                       # sorted: all later too
            val = certified_gap(c.vertices, g, enough=gap)
            if val < gap:
                gap, worst = val, (c.index, tuple(np.round(g, 6)))
                if want and gap < want * diameter:
                    return Report(gap / diameter, gap, diameter, used, cells, worst)
    return Report(gap / diameter, gap, diameter, used, cells, worst)


# --------------------------------------------------------------------------- #
# the Voronoi slice, for cross-validation                                      #
# --------------------------------------------------------------------------- #

def row_echelon(H) -> np.ndarray:
    """Integer row echelon form ``T = U H`` (``U`` unimodular), positive diagonal."""
    A = np.array(np.rint(H), dtype=np.int64)
    n = A.shape[0]
    r = 0
    for c in range(n):
        while True:
            piv = [i for i in range(r, n) if A[i, c] != 0]
            if len(piv) <= 1:
                break
            piv.sort(key=lambda i: abs(A[i, c]))
            i0 = piv[0]
            for i in piv[1:]:
                A[i] -= (A[i, c] // A[i0, c]) * A[i0]
        piv = [i for i in range(r, n) if A[i, c] != 0]
        if not piv:
            continue
        i0 = piv[0]
        A[[r, i0]] = A[[i0, r]]
        if A[r, c] < 0:
            A[r] = -A[r]
        r += 1
    return A


def transversal(lbasis: np.ndarray, hermite: np.ndarray) -> np.ndarray:
    """Coset representatives of ``L / G`` for ``G = H L``, reduced towards 0.

    ``T = U H`` upper triangular gives the box ``0 <= c_i < T_ii`` of coefficient
    vectors; each representative is then Babai-reduced modulo ``G`` so that the
    sites sit in one fundamental cell instead of running off to infinity.
    """
    L = np.asarray(lbasis, float)
    H = np.asarray(hermite, float)
    G = H @ L
    T = row_echelon(H)
    diag = [int(T[i, i]) for i in range(T.shape[0])]
    Ginv = np.linalg.inv(G)
    reps = []
    for coeff in itertools.product(*[range(d) for d in diag]):
        u = np.dot(coeff, L)
        reps.append(u - np.rint(u @ Ginv) @ G)
    return np.array(reps, float)
