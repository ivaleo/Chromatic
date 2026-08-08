r"""МСВ: пер-орбитные косетные раскраски мульти-решёток (multi-site Voronoi).

Псевдо-решётчатая конструкция, строго расширяющая решёточную схему Иванова.

    Точечное множество  S = ⋃_{i=1..s} (P + t_i)  — периодическое, s орбит
    на период P (решётка).  Пространство режется на ячейки Вороного S.
    Каждой орбите i назначается своя подрешётка Γ_i ⊆ P индекса j_i, и ячейка
    узла t_i + g получает цвет (i, [g] ∈ P/Γ_i).  Всего  k = Σ_i j_i  цветов —
    достижимо любое k ≥ s, в отличие от «одна Γ на всех» (k = s·j).

Ключевое упрощение относительно общего периодического случая: одноцветные
ячейки всегда лежат в ОДНОЙ орбите, т.е. являются трансляциями одной и той же
ячейки V_i.  Поэтому вся отделимость сворачивается в

    gap_i = min_{γ ∈ Γ_i \ 0} dist(γ, K_i),      K_i = V_i − V_i,

а качество раскраски — это           d = min_i gap_i / max_i diam V_i,
и при d ≥ 1 верно  χ(ℝⁿ, [1, d]) ≤ k  (после нормировки diam → 1).

При s = 1 конструкция в точности совпадает с решёточной схемой проекта:
dist(γ, V₀ − V₀) = 2·dist(γ/2, V₀) = D(γ) — лемма Иванова.

Статус вычислений: сепарация сертифицируется опорным направлением
(для каждой пары предъявляется единичный вектор u с
⟨u, γ⟩ − h_K(u) = lb ≤ dist(γ, K); h_K — опорная функция по вершинам, точная),
диаметр — точный максимум по парам вершин политопа-НАДмножества ячейки
(надмножество ⇒ диаметр не занижен, сепарация не завышена).  Т.е. отчётное d
консервативно: истинное d конструкции ≥ отчётного с точностью до float-округления.

Соотношение с параллельной работой по силовым (Laguerre) разбиениям
(core/power_coloring.py, другая сессия): класс периодических МСВ-раскрасок --
это срез w = 0 силовых, но с косетной структурой цветов (одноцветность внутри
орбиты) вместо «каждая ячейка периода — свой цвет»; поиск здесь структурный
(кристаллографические анзацы + HNF-подрешётки + локальная оптимизация),
а не CMA по всем параметрам.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial import QhullError


# --------------------------------------------------------------------------- #
# перечисления                                                                 #
# --------------------------------------------------------------------------- #

def hnf_list(index: int, dim: int = 3) -> list[np.ndarray]:
    """Все HNF-матрицы (верхнетреугольные, строки — базис подрешётки в
    координатах P) определителя `index`.  Для dim=3, index=7 их 57."""
    out = []
    if dim != 3:
        raise NotImplementedError("dim=3 only")
    for a in sorted(d for d in range(1, index + 1) if index % d == 0):
        rest = index // a
        for b in sorted(d for d in range(1, rest + 1) if rest % d == 0):
            c = rest // b
            for h01 in range(b):
                for h02 in range(c):
                    for h12 in range(c):
                        out.append(np.array([[a, h01, h02],
                                             [0, b, h12],
                                             [0, 0, c]], dtype=np.int64))
    return out


def adjugate3(H: np.ndarray) -> np.ndarray:
    """Целочисленная адъюгата 3×3: H @ adj = det·I (строки — кроссы столбцов)."""
    H = np.asarray(H, dtype=np.int64)
    c1, c2, c3 = H[:, 0], H[:, 1], H[:, 2]
    return np.stack([np.cross(c2, c3), np.cross(c3, c1), np.cross(c1, c2)])


def prepare_hnfs(index: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """[(H, adj(H), det)] — прекомпьют для быстрой проверки членства."""
    out = []
    for H in hnf_list(index):
        out.append((H, adjugate3(H), index))
    return out


def lattice_points_in_ball(B: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Все n ∈ ℤ³ с |n·B| ≤ radius (кроме 0).  Возвращает (n_int, xyz)."""
    inv_min_sv = 1.0 / np.linalg.svd(B, compute_uv=False)[-1]
    N = int(math.ceil(radius * inv_min_sv)) + 1
    rng = np.arange(-N, N + 1)
    grid = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    xyz = grid @ B
    norms = np.linalg.norm(xyz, axis=1)
    mask = (norms <= radius + 1e-12) & (norms > 1e-12)
    return grid[mask].astype(np.int64), xyz[mask]


# --------------------------------------------------------------------------- #
# ячейка Вороного узла периодического множества                                #
# --------------------------------------------------------------------------- #

@dataclass
class Cell:
    vertices: np.ndarray          # вершины V_i − t_i (ячейка вокруг нуля)
    diam: float                   # точный max по парам вершин
    K_fw: np.ndarray              # экстремальные точки K = V − V (для Франка–Вульфа)
    K_support: np.ndarray         # ВСЕ разности вершин (для опорного сертификата)
    rK: float                     # max |точка K|  (= diam)
    F_normals: np.ndarray | None = None   # направления-кандидаты (нормали фасет K)
    F_support: np.ndarray | None = None   # их ТОЧНЫЕ опоры по K_support

    def fast_lb(self, p: np.ndarray) -> float:
        """Быстрая нижняя оценка dist(p, K): max по фасетным направлениям
        (опоры пересчитаны точно по K_support) и по радиальному направлению
        p/|p|.  Оба слагаемых — корректные опорные оценки."""
        lb = 0.0
        if self.F_normals is not None:
            lb = float(np.max(self.F_normals @ p - self.F_support))
        np_ = float(np.linalg.norm(p))
        if np_ > 1e-12:
            u = p / np_
            lb = max(lb, np_ - float(np.max(self.K_support @ u)))
        return max(0.0, lb)


def site_cell(B: np.ndarray, frac_sites: np.ndarray, i: int,
              rcut0: float | None = None, max_grow: int = 7) -> Cell | None:
    """Ячейка Вороного узла t_i множества  ⋃_j (P + t_j),  сдвинутая в 0.

    Гарантия точности: возвращается, только если все узлы на расстоянии
    ≥ 2·max|vertex| учтены как полупространства (тогда ячейка точная);
    иначе радиус наращивается, при неудаче — None."""
    sites = frac_sites @ B
    t = sites[i]
    rcut = rcut0 or 3.0 * abs(np.linalg.det(B)) ** (1.0 / 3.0)
    for _ in range(max_grow):
        neigh = []
        n_int, _ = lattice_points_in_ball(B, rcut + 1e-9)
        shifts = np.vstack([np.zeros((1, 3)), n_int @ B])
        for j in range(len(sites)):
            pts = sites[j] + shifts - t
            nrm = np.linalg.norm(pts, axis=1)
            keep = (nrm > 1e-9) & (nrm <= rcut)
            neigh.append(pts[keep])
        neigh = np.vstack(neigh)
        if len(neigh) < 4:
            rcut *= 1.6
            continue
        # полупространства ⟨x, n⟩ ≤ |n|²/2, формат qhull: A x + b ≤ 0
        A = neigh
        b = -0.5 * np.einsum("ij,ij->i", neigh, neigh)
        hs = None
        for opts in (None, "QJ"):
            try:
                hs = HalfspaceIntersection(np.hstack([A, b[:, None]]),
                                           np.zeros(3),
                                           qhull_options=opts)
                break
            except (QhullError, ValueError):
                continue
        if hs is None:
            rcut *= 1.6
            continue
        verts = np.unique(np.round(hs.intersections, 9), axis=0)
        rmax = float(np.max(np.linalg.norm(verts, axis=1)))
        if 2.0 * rmax + 1e-9 <= rcut:
            diffs = (verts[:, None, :] - verts[None, :, :]).reshape(-1, 3)
            diam = float(np.max(np.linalg.norm(diffs, axis=1)))
            F_normals = F_support = None
            try:                       # оболочка — ускоритель ФВ и фасетный барьер
                Khull = ConvexHull(diffs, qhull_options="QJ")
                K_fw = diffs[Khull.vertices]
                nrm = Khull.equations[:, :3]
                nn = np.linalg.norm(nrm, axis=1)
                F_normals = nrm / nn[:, None]
                F_support = np.max(F_normals @ diffs.T, axis=1)  # точная опора
            except (QhullError, ValueError):
                K_fw = diffs
            return Cell(vertices=verts, diam=diam, K_fw=K_fw,
                        K_support=diffs, rK=diam,
                        F_normals=F_normals, F_support=F_support)
        rcut = max(rcut * 1.4, 2.0 * rmax + 1e-6)
    return None


# --------------------------------------------------------------------------- #
# сертифицированное расстояние до политопа (Франк–Вульф + опорное направление) #
# --------------------------------------------------------------------------- #

def cert_dist_to_hull(p: np.ndarray, Kv: np.ndarray,
                      K_support: np.ndarray | None = None,
                      iters: int = 400, tol: float = 1e-12) -> float:
    """Нижняя сертифицированная оценка dist(p, conv K_support).

    Франк–Вульф с away-шагами по Kv (экстремальные точки) приближает проекцию
    x̂; сертификат — опорное направление u = (p−x̂)/|p−x̂|:
    lb = ⟨u,p⟩ − max_{v ∈ K_support}⟨u,v⟩ ≤ dist.  Максимум берётся по ПОЛНОМУ
    набору точек K_support (не по qhull-оболочке), поэтому lb корректна при
    любом качестве x̂ и любых огрехах qhull (0 при p ∈ K)."""
    if K_support is None:
        K_support = Kv
    # старт: ближайшая вершина
    d2 = np.einsum("ij,ij->i", Kv - p, Kv - p)
    lam = np.zeros(len(Kv))
    lam[int(np.argmin(d2))] = 1.0
    x = Kv[int(np.argmin(d2))].astype(float)
    for _ in range(iters):
        g = x - p                             # градиент 0.5|x-p|²
        dots = Kv @ g
        s_idx = int(np.argmin(dots))          # FW-вершина
        active = np.nonzero(lam > 1e-14)[0]
        a_idx = int(active[np.argmax(dots[active])])  # away-вершина
        gap_fw = float(g @ (x - Kv[s_idx]))
        if gap_fw < tol:
            break
        d_fw = Kv[s_idx] - x
        d_aw = x - Kv[a_idx]
        if g @ d_fw <= g @ d_aw or lam[a_idx] >= 1.0 - 1e-14:
            dvec, gmax = d_fw, 1.0
            is_fw = True
        else:
            dvec, gmax = d_aw, lam[a_idx] / (1.0 - lam[a_idx] + 1e-300)
            is_fw = False
        denom = float(dvec @ dvec)
        if denom <= 0:
            break
        gamma = min(gmax, max(0.0, float(-(g @ dvec)) / denom))
        if gamma <= 0:
            break
        if is_fw:
            lam *= (1.0 - gamma)
            lam[s_idx] += gamma
        else:
            lam *= (1.0 + gamma)
            lam[a_idx] -= gamma
        x = lam @ Kv
    u = p - x
    nu = float(np.linalg.norm(u))
    if nu <= 1e-14:
        return 0.0
    u /= nu
    return max(0.0, float(u @ p - np.max(K_support @ u)))


# --------------------------------------------------------------------------- #
# оценка конструкции                                                           #
# --------------------------------------------------------------------------- #

class LazyDistTable:
    """Ленивая таблица cert-расстояний от точек P∩шар до K_i."""

    def __init__(self, n_int: np.ndarray, xyz: np.ndarray, cell: Cell,
                 fw_iters: int = 150):
        order = np.argsort(np.linalg.norm(xyz, axis=1))
        self.n_int = n_int[order]
        self.xyz = xyz[order]
        self.norms = np.linalg.norm(self.xyz, axis=1)
        self.cell = cell
        self.rK = cell.rK
        self.fw_iters = fw_iters
        self._fast = np.full(len(self.xyz), -1.0)   # фасетные lb (лениво)
        self._tight: dict[int, float] = {}

    def fast_lb(self, idx: int) -> float:
        if self._fast[idx] < 0.0:
            self._fast[idx] = self.cell.fast_lb(self.xyz[idx])
        return self._fast[idx]

    def dist_tight(self, idx: int) -> float:
        if idx not in self._tight:
            fw = cert_dist_to_hull(self.xyz[idx], self.cell.K_fw,
                                   self.cell.K_support, iters=self.fw_iters)
            self._tight[idx] = max(fw, self.fast_lb(idx))
        return self._tight[idx]

    def best_gap_over_hnfs(self, prepared) -> tuple[float, np.ndarray | None]:
        """max по подрешёткам Γ (прекомпьют prepare_hnfs) от
        min_{γ∈Γ∩шар} cert_dist(γ).  Все использованные значения — валидные
        нижние оценки, поэтому результат — сертифицированная нижняя оценка
        gap лучшей Γ."""
        best, bestH = -1.0, None
        for H, adj, det in prepared:
            mm = self.n_int @ adj
            member = np.all(mm % det == 0, axis=1)
            idxs = np.nonzero(member)[0]
            if len(idxs) == 0:          # ни одной точки Γ в шаре — шар мал
                return math.inf, H
            gap = math.inf
            for idx in idxs:
                if self.norms[idx] - self.rK >= gap:
                    break               # дальше точки не могут понизить минимум
                if idx in self._tight:
                    gap = min(gap, self._tight[idx])
                else:
                    lb = self.fast_lb(idx)
                    if lb < gap:        # только почти-связывающие уточняем ФВ
                        gap = min(gap, self.dist_tight(idx))
                if gap <= best:         # эта Γ уже не лучше текущей
                    break
            if gap > best:
                best, bestH = gap, H
        return best, bestH


def evaluate(B: np.ndarray, frac_sites: np.ndarray, j_list: list[int],
             hnf_lists: dict,
             d_cap: float = 1.8, fw_iters: int = 150) -> dict:
    """Качество лучшей МСВ-раскраски на структуре (B, sites) с индексами j_list.

    Возвращает dict: d, gaps, diams, HNFs, k.  d — консервативная оценка снизу
    (сертифицированные сепарации / точные диаметры надмножеств ячеек)."""
    s = len(frac_sites)
    assert len(j_list) == s
    prepared = {}
    for j in set(j_list):
        items = hnf_lists[j]
        prepared[j] = items if isinstance(items[0], tuple) else \
            [(np.asarray(H, dtype=np.int64), adjugate3(H), j) for H in items]
    cells = []
    for i in range(s):
        c = site_cell(B, frac_sites, i)
        if c is None:
            return {"ok": False, "reason": "cell_failed"}
        cells.append(c)
    diam_max = max(c.diam for c in cells)
    out_gaps, out_H = [], []
    for i, c in enumerate(cells):
        rtab = d_cap * diam_max + c.rK + 1e-9
        n_int, xyz = lattice_points_in_ball(B, rtab)
        table = LazyDistTable(n_int, xyz, c, fw_iters=fw_iters)
        gap, H = table.best_gap_over_hnfs(prepared[j_list[i]])
        if not math.isfinite(gap):
            return {"ok": False, "reason": "table_radius_too_small"}
        out_gaps.append(gap)
        out_H.append(H)
    d = min(out_gaps) / diam_max
    return {"ok": True, "d": float(d), "k": int(sum(j_list)),
            "gaps": [float(g) for g in out_gaps],
            "diams": [float(c.diam) for c in cells],
            "diam_max": float(diam_max),
            "hnfs": [H.tolist() for H in out_H]}


# --------------------------------------------------------------------------- #
# независимая проверка: прямое сравнение ячеек в коробке                       #
# --------------------------------------------------------------------------- #

def independent_check(B: np.ndarray, frac_sites: np.ndarray, j_list: list[int],
                      hnfs: list[np.ndarray], box: int = 2,
                      slsqp_iters: int = 200) -> dict:
    """Независимая проверка d: явная генерация всех ячеек в коробке
    [-box..box]³ периодов, раскраска, минимальная одноцветная межъячеечная
    дистанция через SLSQP-проекцию пар политопов (другой алгоритм, чем FW)
    и точный диаметр.  Возвращает измеренное d_check (не сертификат —
    контроль согласия)."""
    from scipy.optimize import minimize

    s = len(frac_sites)
    cells = [site_cell(B, frac_sites, i) for i in range(s)]
    if any(c is None for c in cells):
        return {"ok": False}
    sites = frac_sites @ B
    dets = [int(round(float(np.linalg.det(H)))) for H in hnfs]
    adjs = [adjugate3(H) for H in hnfs]
    items = []           # (color_id, center, verts_abs)
    rng_box = range(-box, box + 1)
    for i in range(s):
        for n in itertools.product(rng_box, repeat=3):
            n = np.array(n, dtype=np.int64)
            coset = tuple((n @ adjs[i]) % dets[i])
            color = (i, coset)
            center = sites[i] + n @ B
            items.append((color, center, cells[i].vertices + center))
    # min одноцветная межъячеечная дистанция
    dmin = math.inf
    argpair = None
    for a in range(len(items)):
        ca, pa, va = items[a]
        for bidx in range(a + 1, len(items)):
            cb, pb, vb = items[bidx]
            if ca != cb:
                continue
            gap0 = np.linalg.norm(pb - pa) - cells[ca[0]].diam
            if gap0 >= dmin:
                continue
            # SLSQP: min |x−y|², x ∈ conv(va), y ∈ conv(vb) (по барицентрическим)
            na, nb = len(va), len(vb)
            def obj(z):
                x = z[:na] @ va
                y = z[na:] @ vb
                return float(np.sum((x - y) ** 2))
            cons = [{"type": "eq", "fun": lambda z: np.sum(z[:na]) - 1.0},
                    {"type": "eq", "fun": lambda z: np.sum(z[na:]) - 1.0}]
            z0 = np.concatenate([np.full(na, 1.0 / na), np.full(nb, 1.0 / nb)])
            res = minimize(obj, z0, bounds=[(0.0, 1.0)] * (na + nb),
                           constraints=cons, method="SLSQP",
                           options={"maxiter": slsqp_iters, "ftol": 1e-14})
            dist = math.sqrt(max(res.fun, 0.0))
            if dist < dmin:
                dmin, argpair = dist, (a, bidx)
    diam_max = max(c.diam for c in cells)
    return {"ok": True, "d_check": float(dmin / diam_max),
            "min_pair_dist": float(dmin), "diam_max": float(diam_max),
            "n_cells": len(items), "argpair": argpair}
