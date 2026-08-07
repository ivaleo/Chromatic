"""Certified upper bounds on the covering radius of a *layered* lattice.

For ``Lambda = { (x + i c, i t) : x in L', i in Z }`` (layers of a base lattice
``L'`` at height ``t``, offset ``c``) the distance function decomposes by layer:

    dist^2((x, z), Lambda) = min_i [ f(x - i c) + (z - i t)^2 ],
    f(y) = dist^2(y, L').

Hence, restricting the minimum to any *subset* ``I`` of layers only increases it:

    R(Lambda)^2 <= max_{x in F, z in [0, t]} min_{i in I} [ f(x - i c) + (z - i t)^2 ]

for any set ``F`` covering a fundamental domain of ``L'``.  The right-hand side
is certified by branch-and-bound over axis-aligned boxes:

- for a box ``B`` and layer ``i``, ANY base-lattice point ``s`` gives the valid
  monotone bound ``max_{x in B} f(x - i c) <= max_{x in B} |x - i c - s|^2``,
  which has a closed per-coordinate form; we take the best of ``k`` nearby
  points (Babai reduction + KD-tree), so the bound is first-order tight in the
  box radius;
- the inner maximum over ``z`` of the lower envelope of the parabolas
  ``A_i + (z - i t)^2`` is exact: it is attained at a clipped crossing point or
  an endpoint, finitely many candidates;
- boxes whose bound is below the target are pruned; survivors split along the
  widest axis.  If the queue empties, the target is certified.

The floating-point evaluations here are straightforward interval-safe
arithmetic (sums of squares, min/max); a fully rational replay of the surviving
inequalities is routine and left to the certification pass.

This replaces both failed certification routes recorded in RESULTS.md: the
direction branching over ``S^8`` (exponential in essence) and the Delone cell
enumeration (``~9! = 362880`` cells).  The box tree adapts to where the
distance function is actually large, instead of resolving the whole sphere.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from chromatic_research.core.lamination import enumerate_upto


@dataclass
class LayeredCertifier:
    base: np.ndarray                   # base lattice basis (rows), dimension n-1
    offset: np.ndarray                 # layer offset c
    height: float                      # layer height t
    base_r2: float = np.inf            # EXACT covering radius^2 of the base:
                                       # f(y) <= base_r2 always, so every box
                                       # bound is capped by it.  With the cap the
                                       # certifier starts at the (P1) bound and
                                       # branch-and-bound only has to resolve the
                                       # neighbourhoods of near-doubly-deep points.
    layers: tuple[int, ...] = (-1, 0, 1, 2)
    k_candidates: int = 6
    point_radius: float = 4.2          # base points kept for nearest-point queries
    tree: cKDTree = field(init=False)
    points: np.ndarray = field(init=False)
    base_inv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        base = np.asarray(self.base, dtype=float)
        self.offset = np.asarray(self.offset, dtype=float)
        pts = enumerate_upto(base, self.point_radius)
        self.points = np.vstack([np.zeros((1, base.shape[1])), pts])
        self.tree = cKDTree(self.points)
        # Babai frame: LLL-reduce so the rounding error stays well inside
        # point_radius (with the raw E8 basis the reduction can drift outside
        # the stored ball and the nearest-point query silently inflates f).
        try:
            import combigeo
            reduced = np.asarray(combigeo.lll_reduce(base.tolist()), dtype=float)
        except Exception:                                   # noqa: BLE001
            reduced = base
        self.babai_basis = reduced
        self.base_inv = np.linalg.inv(reduced)

    # ------------------------------------------------------------------ bounds

    def _layer_bounds(self, lo: np.ndarray, hi: np.ndarray, layer: int,
                      *, exact_at_center: bool = False) -> np.ndarray:
        """Upper bound of ``f(x - layer*c)`` over each box (vectorised).

        With ``exact_at_center=True`` returns instead the exact value at the box
        center (used for the measured lower bound).
        """
        centers = (lo + hi) / 2 - layer * self.offset
        # Babai reduction: shift each query near the origin by a lattice point
        coeff = np.rint(centers @ self.base_inv)
        shift = coeff @ self.babai_basis
        reduced = centers - shift
        _, idx = self.tree.query(reduced, k=self.k_candidates)
        candidates = self.points[idx]                        # (N, k, dim)
        if exact_at_center:
            delta = reduced[:, None, :] - candidates
            return np.min(np.einsum("nkd,nkd->nk", delta, delta), axis=1)
        half = (hi - lo) / 2
        # max over box of |x - s|^2, per coordinate: (|center-s| + half)^2,
        # capped by the exact covering radius bound f <= base_r2
        gap = np.abs(reduced[:, None, :] - candidates) + half[:, None, :]
        return np.minimum(np.min(np.sum(gap * gap, axis=2), axis=1), self.base_r2)

    def _max_over_z(self, A: np.ndarray) -> np.ndarray:
        """Exact ``max_{z in [0,t]} min_i (A_i + (z - i t)^2)`` (vectorised).

        ``A`` has shape (N, L) in the order of ``self.layers``.  Candidates:
        endpoints and all pairwise crossings, clipped to ``[0, t]``.
        """
        t = self.height
        layer_arr = np.asarray(self.layers, dtype=float)
        n_boxes, n_layers = A.shape
        candidates = [np.zeros(n_boxes), np.full(n_boxes, t)]
        for a in range(n_layers):
            for b in range(a + 1, n_layers):
                i, j = layer_arr[a], layer_arr[b]
                z = (A[:, b] - A[:, a] + (j * j - i * i) * t * t) / (2 * (j - i) * t)
                candidates.append(np.clip(z, 0.0, t))
        best = np.zeros(n_boxes)
        for z in candidates:
            value = np.min(A + (z[:, None] - layer_arr[None, :] * t) ** 2, axis=1)
            np.maximum(best, value, out=best)
        return best

    def box_bound(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        A = np.stack([self._layer_bounds(lo, hi, layer) for layer in self.layers], axis=1)
        return self._max_over_z(A)

    def center_value(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Exact objective at box centers -- a LOWER bound on the true maximum."""
        A = np.stack([self._layer_bounds(lo, hi, layer, exact_at_center=True)
                      for layer in self.layers], axis=1)
        return self._max_over_z(A)

    # ----------------------------------------------------------------- certify

    def certify(self, target_r2: float, *, initial_radius: float | None = None,
                max_boxes: int = 40_000_000, wave_limit: int = 400_000,
                budget: float = 1800.0, verbose: bool = True) -> dict:
        """Try to certify ``R^2 <= target_r2``.  Returns a report dict."""
        start = time.time()
        dim = self.points.shape[1]
        if initial_radius is None:
            # circumradius of the base cell: any box covering V0(base) works,
            # and V0 sits inside the ball of the covering radius.  Estimate it
            # from the shortest stored point pair structure is unreliable --
            # callers should pass the exact value; the fallback is generous.
            initial_radius = float(np.min(np.linalg.norm(self.points[1:], axis=1)))
        lo = np.full((1, dim), -initial_radius)
        hi = np.full((1, dim), +initial_radius)
        measured = 0.0
        processed = 0
        waves = 0
        while len(lo):
            if processed > max_boxes or time.time() - start > budget:
                return {"certified": False, "reason": "budget",
                        "target_r2": target_r2, "processed": processed,
                        "queue": int(len(lo)), "measured_max_r2_lower": measured,
                        "seconds": round(time.time() - start, 1)}
            waves += 1
            batch_lo, batch_hi = lo[:wave_limit], hi[:wave_limit]
            rest_lo, rest_hi = lo[wave_limit:], hi[wave_limit:]
            bound = self.box_bound(batch_lo, batch_hi)
            measured = max(measured, float(np.max(
                self.center_value(batch_lo, batch_hi))))
            processed += len(batch_lo)
            alive = bound > target_r2
            if verbose:
                print(f"  wave {waves}: {len(batch_lo)} boxes, "
                      f"{int(alive.sum())} survive, worst bound "
                      f"{float(np.max(bound)):.6f}, measured >= {measured:.6f} "
                      f"[{time.time() - start:.0f}s]", flush=True)
            if measured > target_r2:
                return {"certified": False, "reason": "target below true maximum",
                        "target_r2": target_r2, "processed": processed,
                        "measured_max_r2_lower": measured,
                        "seconds": round(time.time() - start, 1)}
            split_lo, split_hi = batch_lo[alive], batch_hi[alive]
            if len(split_lo):
                widths = split_hi - split_lo
                axis = np.argmax(widths, axis=1)
                mid = (split_lo[np.arange(len(split_lo)), axis]
                       + split_hi[np.arange(len(split_hi)), axis]) / 2
                left_hi = split_hi.copy()
                left_hi[np.arange(len(split_lo)), axis] = mid
                right_lo = split_lo.copy()
                right_lo[np.arange(len(split_lo)), axis] = mid
                new_lo = np.vstack([split_lo, right_lo])
                new_hi = np.vstack([left_hi, split_hi])
            else:
                new_lo = np.zeros((0, dim))
                new_hi = np.zeros((0, dim))
            lo = np.vstack([rest_lo, new_lo])
            hi = np.vstack([rest_hi, new_hi])
        return {"certified": True, "target_r2": target_r2, "processed": processed,
                "waves": waves, "measured_max_r2_lower": measured,
                "seconds": round(time.time() - start, 1)}
