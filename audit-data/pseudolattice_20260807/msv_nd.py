r"""МСВ в произвольной размерности (обобщение msv_core на n ≥ 3).

Отличия от 3D-ядра:
- разностное тело K = V − V никогда не материализуется: опорная функция
  h_K(u) = max(Vu) − min(Vu) и линейный оракул Франка–Вульфа считаются прямо
  по вершинам ячейки (как в power_coloring другой сессии);
- подрешётки Γ индекса j: в 4D полных HNF десятки тысяч, поэтому в поиске
  используется ПОРТФЕЛЬ (детерминированная подвыборка по сиду + элита,
  накопленная по ходу прогона). Репортуемое d — максимум по подмножеству,
  т.е. валидная НИЖНЯЯ оценка качества конструкции (сертификат не слабеет);
  для финалистов есть полный перебор (exhaustive=True).
- сертификат расстояния прежний: lb = ⟨u,γ⟩ − h_K(u) с u из ФВ-проекции,
  корректен при любой сходимости ФВ.

Классификация значений: measured (сертифицированные сепарации/точные
диаметры, float); рационализация — отдельный шаг.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import HalfspaceIntersection, QhullError


# --------------------------------------------------------------------------- #
# HNF в размерности n                                                          #
# --------------------------------------------------------------------------- #

def _divisor_tuples(index: int, dim: int):
    if dim == 1:
        yield (index,)
        return
    for d in range(1, index + 1):
        if index % d == 0:
            for rest in _divisor_tuples(index // d, dim - 1):
                yield (d,) + rest


def hnf_count(index: int, dim: int) -> int:
    tot = 0
    for diag in _divisor_tuples(index, dim):
        m = 1
        for i, d in enumerate(diag):
            m *= d ** i
        tot += m
    return tot


def hnf_iter(index: int, dim: int):
    """Все верхнетреугольные HNF (строки — базис Γ в координатах P)."""
    for diag in _divisor_tuples(index, dim):
        ranges = []
        for j in range(dim):
            for i in range(j):
                ranges.append(range(diag[j]))       # H[i][j] ∈ [0, diag[j])
        for offs in itertools.product(*ranges):
            H = np.zeros((dim, dim), dtype=np.int64)
            for j in range(dim):
                H[j, j] = diag[j]
            pos = 0
            for j in range(dim):
                for i in range(j):
                    H[i, j] = offs[pos]
                    pos += 1
            yield H


def adjugate_int(H: np.ndarray) -> np.ndarray:
    """Целочисленная адъюгата: H @ adj = det·I (через float-инверсию + проверку)."""
    H = np.asarray(H, dtype=np.int64)
    det = int(round(np.linalg.det(H.astype(float))))
    adj = np.rint(np.linalg.inv(H.astype(float)) * det).astype(np.int64)
    assert np.array_equal(H @ adj, det * np.eye(len(H), dtype=np.int64)), H
    return adj


def hnf_portfolio(index: int, dim: int, max_size: int = 1500,
                  seed: int = 0) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Детерминированный портфель HNF: вся масса, если она ≤ max_size, иначе
    равномерная подвыборка по сиду. Возвращает [(H, adj, det)]."""
    total = hnf_count(index, dim)
    rng = np.random.default_rng(seed + 1000 * index + dim)
    if total <= max_size:
        keep = None
    else:
        keep = set(rng.choice(total, size=max_size, replace=False).tolist())
    out = []
    for i, H in enumerate(hnf_iter(index, dim)):
        if keep is None or i in keep:
            out.append((H, adjugate_int(H), index))
    return out


# --------------------------------------------------------------------------- #
# ячейка узла и сертифицированные расстояния (оракул по вершинам)              #
# --------------------------------------------------------------------------- #

@dataclass
class CellND:
    vertices: np.ndarray          # вершины ячейки (сдвинута в 0)
    diam: float

    def h_K(self, u: np.ndarray) -> float:
        p = self.vertices @ u
        return float(p.max() - p.min())

    def fast_lb(self, p: np.ndarray) -> float:
        npn = float(np.linalg.norm(p))
        if npn < 1e-12:
            return 0.0
        return max(0.0, npn - self.h_K(p / npn))


def lattice_points_in_ball_nd(B: np.ndarray, radius: float):
    dim = len(B)
    inv_min_sv = 1.0 / np.linalg.svd(B, compute_uv=False)[-1]
    N = int(math.ceil(radius * inv_min_sv)) + 1
    rng = np.arange(-N, N + 1)
    grid = np.array(np.meshgrid(*([rng] * dim), indexing="ij")).reshape(dim, -1).T
    xyz = grid @ B
    norms = np.linalg.norm(xyz, axis=1)
    mask = (norms <= radius + 1e-12) & (norms > 1e-12)
    return grid[mask].astype(np.int64), xyz[mask]


def site_cell_nd(B: np.ndarray, frac_sites: np.ndarray, i: int,
                 rcut0: float | None = None, max_grow: int = 7) -> CellND | None:
    dim = B.shape[0]
    sites = frac_sites @ B
    t = sites[i]
    rcut = rcut0 or 3.0 * abs(np.linalg.det(B)) ** (1.0 / dim)
    for _ in range(max_grow):
        n_int, _ = lattice_points_in_ball_nd(B, rcut + 1e-9)
        shifts = np.vstack([np.zeros((1, dim)), n_int @ B])
        neigh = []
        for j in range(len(sites)):
            pts = sites[j] + shifts - t
            nrm = np.linalg.norm(pts, axis=1)
            keep = (nrm > 1e-9) & (nrm <= rcut)
            neigh.append(pts[keep])
        neigh = np.vstack(neigh)
        if len(neigh) < dim + 1:
            rcut *= 1.6
            continue
        b = -0.5 * np.einsum("ij,ij->i", neigh, neigh)
        hs = None
        for opts in (None, "QJ"):
            try:
                hs = HalfspaceIntersection(np.hstack([neigh, b[:, None]]),
                                           np.zeros(dim), qhull_options=opts)
                break
            except (QhullError, ValueError):
                continue
        if hs is None:
            rcut *= 1.6
            continue
        verts = np.unique(np.round(hs.intersections, 9), axis=0)
        rmax = float(np.max(np.linalg.norm(verts, axis=1)))
        if 2.0 * rmax + 1e-9 <= rcut:
            sq = np.einsum("ij,ij->i", verts, verts)
            d2 = sq[:, None] + sq[None, :] - 2.0 * verts @ verts.T
            diam = float(math.sqrt(max(0.0, d2.max())))
            return CellND(vertices=verts, diam=diam)
        rcut = max(rcut * 1.4, 2.0 * rmax + 1e-6)
    return None


def cert_dist_nd(p: np.ndarray, cell: CellND, iters: int = 120,
                 tol: float = 1e-12) -> float:
    """ФВ над K по оракулу вершин; сертификат — опорное направление."""
    V = cell.vertices
    # старт: разность ближайших опорных вершин вдоль p
    d0 = V @ p
    x = V[int(np.argmax(d0))] - V[int(np.argmin(d0))]
    for _ in range(iters):
        g = x - p
        dV = V @ g
        s = V[int(np.argmin(dV))] - V[int(np.argmax(dV))]     # argmin ⟨g,·⟩ на K
        d = s - x
        gd = float(g @ d)
        if gd >= -tol:
            break
        gamma = min(1.0, -gd / float(d @ d))
        if gamma <= 0:
            break
        x = x + gamma * d
    u = p - x
    nu = float(np.linalg.norm(u))
    if nu <= 1e-14:
        return 0.0
    u /= nu
    return max(0.0, float(u @ p) - cell.h_K(u))


class LazyDistTableND:
    def __init__(self, n_int, xyz, cell: CellND, fw_iters: int = 120):
        order = np.argsort(np.linalg.norm(xyz, axis=1))
        self.n_int = n_int[order]
        self.xyz = xyz[order]
        self.norms = np.linalg.norm(self.xyz, axis=1)
        self.cell = cell
        self.rK = cell.diam
        self.fw_iters = fw_iters
        self._fast = np.full(len(self.xyz), -1.0)
        self._tight: dict[int, float] = {}

    def fast_lb(self, idx):
        if self._fast[idx] < 0.0:
            self._fast[idx] = self.cell.fast_lb(self.xyz[idx])
        return self._fast[idx]

    def dist_tight(self, idx):
        if idx not in self._tight:
            fw = cert_dist_nd(self.xyz[idx], self.cell, iters=self.fw_iters)
            self._tight[idx] = max(fw, self.fast_lb(idx))
        return self._tight[idx]

    def best_gap(self, prepared) -> tuple[float, np.ndarray | None]:
        best, bestH = -1.0, None
        for H, adj, det in prepared:
            mm = self.n_int @ adj
            member = np.all(mm % det == 0, axis=1)
            idxs = np.nonzero(member)[0]
            if len(idxs) == 0:
                return math.inf, H
            gap = math.inf
            for idx in idxs:
                if self.norms[idx] - self.rK >= gap:
                    break
                if idx in self._tight:
                    gap = min(gap, self._tight[idx])
                else:
                    lb = self.fast_lb(idx)
                    if lb < gap:
                        gap = min(gap, self.dist_tight(idx))
                if gap <= best:
                    break
            if gap > best:
                best, bestH = gap, H
        return best, bestH


_ELITE: dict[tuple[int, int], list] = {}     # (dim, j) -> [(H, adj, det)]


def evaluate_nd(B: np.ndarray, frac_sites: np.ndarray, j_list: list[int],
                portfolios: dict, d_cap: float = 1.30,
                fw_iters: int = 120, elite_max: int = 40) -> dict:
    dim = B.shape[0]
    s = len(frac_sites)
    cells = []
    for i in range(s):
        c = site_cell_nd(B, frac_sites, i)
        if c is None:
            return {"ok": False, "reason": "cell_failed"}
        cells.append(c)
    diam_max = max(c.diam for c in cells)
    gaps, hs = [], []
    for i, c in enumerate(cells):
        j = j_list[i]
        rtab = d_cap * diam_max + c.diam + 1e-9
        n_int, xyz = lattice_points_in_ball_nd(B, rtab)
        table = LazyDistTableND(n_int, xyz, c, fw_iters=fw_iters)
        elite = _ELITE.get((dim, j), [])
        gap, H = table.best_gap(elite + portfolios[j])
        if not math.isfinite(gap):
            return {"ok": False, "reason": "table_radius_too_small"}
        gaps.append(gap)
        hs.append(H)
        if H is not None:                       # пополнить элиту
            key = (dim, j)
            el = _ELITE.setdefault(key, [])
            if not any(np.array_equal(H, e[0]) for e in el):
                el.insert(0, (H, adjugate_int(H), j))
                del el[elite_max:]
    d = min(gaps) / diam_max
    return {"ok": True, "d": float(d), "k": int(sum(j_list)),
            "gaps": [float(g) for g in gaps],
            "diams": [float(c.diam) for c in cells],
            "diam_max": float(diam_max),
            "hnfs": [None if H is None else H.tolist() for H in hs]}


def exhaustive_gap(B: np.ndarray, frac_sites: np.ndarray, site_i: int, j: int,
                   d_cap: float = 1.45, fw_iters: int = 400,
                   chunk: int = 4000, log=None) -> tuple[float, np.ndarray]:
    """Полный перебор всех HNF индекса j для одной орбиты (для финалистов)."""
    dim = B.shape[0]
    cells = [site_cell_nd(B, frac_sites, i) for i in range(len(frac_sites))]
    diam_max = max(c.diam for c in cells)
    c = cells[site_i]
    rtab = d_cap * diam_max + c.diam + 1e-9
    n_int, xyz = lattice_points_in_ball_nd(B, rtab)
    table = LazyDistTableND(n_int, xyz, c, fw_iters=fw_iters)
    best, bestH = -1.0, None
    buf = []
    done = 0
    for H in hnf_iter(j, dim):
        buf.append((H, adjugate_int(H), j))
        if len(buf) >= chunk:
            g, hh = table.best_gap(buf)
            if g > best:
                best, bestH = g, hh
            done += len(buf)
            buf = []
            if log:
                log(done, best)
    if buf:
        g, hh = table.best_gap(buf)
        if g > best:
            best, bestH = g, hh
    return best, bestH
