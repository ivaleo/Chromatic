"""Кластерные G-периодические раскраски с мультипликативным зазором.

Новый класс псевдо-решётчатых раскрасок (сессия fable_pseudolattice_20260807).

Конструкция. Мелкая решётка ``L`` (базис ``B``, строки — векторы) задаёт
мозаику ячеек Вороного ``V0 + t``, ``t in L``. Период ``G = H @ B`` — подрешётка
индекса ``N = |det H|``. Раскраска — произвольное G-периодическое отображение
``phi: L/G -> {0..k-1}`` (НЕ обязательно по смежным классам промежуточной
решётки: классы цвета могут быть невыпуклыми объединениями ячеек-кластеров).

Исчисление интервалов. Ячейки центрально-симметричны, поэтому разностное тело
``V0 - V0 = 2 V0``, и множество расстояний между ячейками ``V0`` и ``V0 + w``
есть в точности отрезок ``[D(w), H(w)]``:

    D(w) = 2 dist(w/2, V0)          (лемма Иванова; 0, если w/2 внутри V0)
    H(w) = max_{v вершина V0} |w + 2 v|

Класс цвета реализует множество расстояний
``U(phi) = [0, diam V0]  ∪  U_{одноцветные пары} [D, H]``.
Наибольший мультипликативный зазор ``(a, b)`` в ``U(phi)`` даёт после
нормировки ``chi(R^n, [1, l]) <= k`` для всех ``l < b/a``. Для косетной
раскраски ``phi = L/Gamma`` это в точности ``d = D_min/diam``.

Поиск: скан порога кластера ``a`` (критические значения — концы ``H``) ×
бинарный поиск ``b`` по значениям ``D`` с CP-SAT-проверкой k-раскрашиваемости
конфликтного графа: ребро (i,j), если какой-то интервал пары пересекает (a,b).

Честность статусов: все найденные phi перепроверяются verify_coloring двумя
независимыми процедурами (GJK combigeo + SLSQP-проекция на H-представление
ячейки из qhull) и сопровождаются сертифицированной нижней границей D через
опорный функционал по вершинам. Отрицательные результаты CP-SAT (UNSAT) —
экраны для данного (L, G, cutoff, сетки, eps), не доказательства невозможности.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

import combigeo


# --------------------------------------------------------------------------- #
# геометрия ячейки                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class CellGeom:
    basis: np.ndarray            # строки — базис L
    cell: object                 # combigeo.VoronoiCell
    verts: np.ndarray            # вершины V0
    diam: float                  # diam V0
    hull_A: np.ndarray           # H-представление V0: A x <= bvec (qhull)
    hull_b: np.ndarray

    @classmethod
    def build(cls, basis) -> "CellGeom":
        from scipy.spatial import ConvexHull

        B = np.asarray(basis, float)
        cell = combigeo.voronoi_cell(B.tolist())
        V = np.asarray(cell.vertices, float)
        hull = ConvexHull(V)
        A = hull.equations[:, :-1]
        bvec = -hull.equations[:, -1]
        return cls(B, cell, V, float(cell.diameter), A, bvec)


def pair_interval(geom: CellGeom, w: np.ndarray) -> tuple[float, float]:
    """(D(w), H(w)) — интервал расстояний между V0 и V0 + w (GJK-ветка)."""
    lo = 2.0 * combigeo.distance_to_cell((0.5 * np.asarray(w, float)).tolist(),
                                         geom.cell)
    hi = float(np.sqrt(((w + 2.0 * geom.verts) ** 2).sum(axis=1).max()))
    return float(lo), hi


def pair_lo_slsqp(geom: CellGeom, w: np.ndarray) -> tuple[float, float]:
    """Независимая проверка D(w): SLSQP-проекция w/2 на V0 (H-представление)
    + сертифицированная нижняя граница через опорный функционал по вершинам.

    Возвращает (2*dist_slsqp, 2*dist_cert): вторая величина — строгая нижняя
    граница D(w) с точностью до float-погрешности вершин (~1e-12)."""
    from scipy.optimize import minimize

    p = 0.5 * np.asarray(w, float)
    if np.all(geom.hull_A @ p <= geom.hull_b + 1e-12):
        return 0.0, 0.0
    res = minimize(lambda y: ((y - p) ** 2).sum(),
                   x0=np.zeros(len(p)),
                   jac=lambda y: 2.0 * (y - p),
                   constraints=[{"type": "ineq",
                                 "fun": lambda y: geom.hull_b - geom.hull_A @ y,
                                 "jac": lambda y: -geom.hull_A}],
                   method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-14})
    y = res.x
    dist = float(np.linalg.norm(y - p))
    u = (p - y)
    nu = np.linalg.norm(u)
    if nu < 1e-14:
        return 2.0 * dist, 0.0
    u /= nu
    support = float((geom.verts @ u).max())
    cert = max(0.0, float(u @ p) - support)
    return 2.0 * dist, 2.0 * cert


# --------------------------------------------------------------------------- #
# смежные классы и перечисление пар                                            #
# --------------------------------------------------------------------------- #


def cosets_of_hnf(H: np.ndarray) -> np.ndarray:
    """Представители L/G для целочисленной матрицы перехода H (G = H @ L
    в координатах L): все целые точки x в [0, d_i) по столбцовой HNF-редукции.

    Универсально (не требует треугольности): берём вектора из box
    прямоугольника Смита через редукцию по модулю H."""
    H = np.asarray(H, dtype=np.int64)
    n = H.shape[0]
    from sympy import Matrix

    Hs = Matrix(H.tolist())
    det = int(Hs.det())
    if det == 0:
        raise ValueError("вырожденная матрица перехода")
    N = abs(det)
    adj = np.array(Hs.adjugate().tolist(), dtype=object)   # H^{-1} = adj/det

    def canon(x):
        # точная редукция в фундаментальный параллелепипед [0,1)^n:
        # q = floor(x H^{-1}) покомпонентно (int // int в python — floor)
        t = [int(v) for v in (np.asarray(x, dtype=object) @ adj)]
        q = np.array([tt // det for tt in t], dtype=object)
        c = np.asarray(x, dtype=object) - q @ H
        return tuple(int(v) for v in c)

    seen: dict[tuple, np.ndarray] = {}
    # BFS от нуля по единичным шагам гарантированно покрывает все классы
    frontier = [np.zeros(n, dtype=np.int64)]
    seen[canon(frontier[0])] = frontier[0]
    steps = [np.eye(n, dtype=np.int64)[i] * s for i in range(n) for s in (1, -1)]
    while frontier and len(seen) < N:
        nxt = []
        for x in frontier:
            for s in steps:
                key = canon(x + s)
                if key not in seen:
                    seen[key] = np.array(key, dtype=np.int64)
                    nxt.append(np.array(key, dtype=np.int64))
        frontier = nxt
    if len(seen) != N:
        raise RuntimeError(f"найдено {len(seen)} классов вместо {N}")
    return np.stack(list(seen.values()))


def _lattice_points_in_ball(basis: np.ndarray, radius: float) -> np.ndarray:
    """Целые коэффициенты c: |c @ basis| <= radius (простое box-перечисление
    по LLL-редуцированному базису)."""
    Bred = np.asarray(combigeo.lll_reduce(basis.tolist()), float)
    # box-границы через двойственный базис: |c_i| <= radius * |b*_i|
    Ginv = np.linalg.inv(Bred @ Bred.T)
    bounds = np.sqrt(np.diag(Ginv)) * radius
    ranges = [np.arange(-int(math.floor(b + 1e-9)), int(math.floor(b + 1e-9)) + 1)
              for b in bounds]
    pts = []
    for c in itertools.product(*ranges):
        v = np.asarray(c, float) @ Bred
        if v @ v <= radius * radius + 1e-9:
            pts.append(v)
    return np.asarray(pts)


@dataclass
class PairData:
    """Интервалы расстояний для пары классов (i <= j)."""
    intervals: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class ConflictData:
    geom: CellGeom
    H: np.ndarray                        # матрица перехода периода
    cosets: np.ndarray                   # N x n целых представителей
    cutoff: float                        # |w| <= cutoff перечислено
    claim_cap: float                     # b выше этого не заявляем
    pairs: dict[tuple[int, int], PairData]
    build_seconds: float

    @property
    def n_classes(self) -> int:
        return len(self.cosets)


def build_conflicts(basis, H, *, cutoff_mult: float = 3.2,
                    heartbeat: float = 15.0) -> ConflictData:
    """Полное перечисление интервалов пар классов до |w| <= cutoff.

    cutoff = cutoff_mult * diam(V0); любой не перечисленный w имеет
    D(w) >= |w| - diam > cutoff - diam =: claim_cap, поэтому все зазоры с
    b <= claim_cap корректны."""
    t0 = time.time()
    geom = CellGeom.build(basis)
    H = np.asarray(H, dtype=np.int64)
    cosets = cosets_of_hnf(H)
    Gbasis = H.astype(float) @ geom.basis
    cutoff = cutoff_mult * geom.diam
    claim_cap = cutoff - geom.diam

    # редукция Бабая: представители классов могут быть далеко от нуля, поэтому
    # каждую разность приводим к delta_red = delta mod G с |delta_red| <=
    # (1/2) sum |b_i| (b_i — LLL-базис G); окно перечисления g берём от неё
    Gred = np.asarray(combigeo.lll_reduce(Gbasis.tolist()), float)
    Gred_inv = np.linalg.inv(Gred)
    N = len(cosets)
    deltas_red = {}
    for i in range(N):
        for j in range(i, N):
            delta = (cosets[j] - cosets[i]).astype(float) @ geom.basis
            dred = delta - np.round(delta @ Gred_inv) @ Gred
            deltas_red[(i, j)] = dred
    max_dr = max(np.linalg.norm(d) for d in deltas_red.values())
    if max_dr > cutoff:
        raise RuntimeError(
            f"|delta_red|={max_dr:.3f} > cutoff={cutoff:.3f}: увеличьте "
            f"cutoff_mult (иначе пара классов останется без интервалов)")

    gpts = _lattice_points_in_ball(Gbasis, cutoff + max_dr + 1e-9)
    pairs: dict[tuple[int, int], PairData] = {}
    last_beat = t0
    for i in range(N):
        for j in range(i, N):
            delta = deltas_red[(i, j)]
            ws = gpts + delta
            norms = np.linalg.norm(ws, axis=1)
            sel = ws[norms <= cutoff]
            if len(sel) == 0 and not (i == j and np.linalg.norm(delta) < 1e-9):
                raise RuntimeError(f"пара ({i},{j}) без интервалов при "
                                   f"|delta_red|={np.linalg.norm(delta):.3f}")
            ivs = []
            for w in sel:
                if i == j and np.linalg.norm(w) < 1e-9:
                    continue
                lo, hi = pair_interval(geom, w)
                ivs.append((lo, hi))
            if ivs:
                pairs[(i, j)] = PairData(ivs)
        if time.time() - last_beat > heartbeat:
            print(f"    [conflicts] класс {i + 1}/{N}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            last_beat = time.time()
    return ConflictData(geom, H, cosets, cutoff, claim_cap, pairs,
                        time.time() - t0)


# --------------------------------------------------------------------------- #
# раскраска конфликтного графа                                                 #
# --------------------------------------------------------------------------- #


def _edges_at(conf: ConflictData, a: float, b: float,
              eps: float) -> tuple[list[tuple[int, int]], bool]:
    """Рёбра конфликтов на уровне (a, b); второй элемент False, если какой-то
    самоконфликт (i, i) делает уровень невыполнимым."""
    edges = []
    for (i, j), pd in conf.pairs.items():
        conflict = any(lo < b - eps and hi > a + eps for lo, hi in pd.intervals)
        if conflict:
            if i == j:
                return [], False
            edges.append((i, j))
    return edges, True


def _greedy_colorable(n: int, edges: list[tuple[int, int]], k: int) -> list[int] | None:
    """DSATUR-жадная попытка (быстрый positive-путь перед CP-SAT)."""
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    colors = [-1] * n
    for _ in range(n):
        best, best_sat = -1, (-1, -1)
        for v in range(n):
            if colors[v] != -1:
                continue
            sat = len({colors[u] for u in adj[v] if colors[u] != -1})
            key = (sat, len(adj[v]))
            if key > best_sat:
                best_sat, best = key, v
        used = {colors[u] for u in adj[best]}
        c = next((c for c in range(k) if c not in used), None)
        if c is None:
            return None
        colors[best] = c
    return colors


def _cpsat_colorable(n: int, edges: list[tuple[int, int]], k: int,
                     timeout_s: float) -> tuple[str, list[int] | None]:
    """CP-SAT-проверка k-раскрашиваемости. Возвращает (status, phi)."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    x = [m.new_int_var(0, k - 1, f"x{i}") for i in range(n)]
    for i, j in edges:
        m.add(x[i] != x[j])
    m.add(x[0] == 0)                      # слом симметрии
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_s
    solver.parameters.num_search_workers = 4
    st = solver.solve(m)
    if st == cp_model.OPTIMAL or st == cp_model.FEASIBLE:
        return "SAT", [solver.value(v) for v in x]
    if st == cp_model.INFEASIBLE:
        return "UNSAT", None
    return "UNKNOWN", None


def colorable(conf: ConflictData, a: float, b: float, k: int, *,
              eps: float = 1e-9, timeout_s: float = 10.0
              ) -> tuple[str, list[int] | None]:
    edges, ok = _edges_at(conf, a, b, eps)
    if not ok:
        return "SELF", None
    phi = _greedy_colorable(conf.n_classes, edges, k)
    if phi is not None:
        return "SAT", phi
    return _cpsat_colorable(conf.n_classes, edges, k, timeout_s)


# --------------------------------------------------------------------------- #
# зазор реализованного множества                                               #
# --------------------------------------------------------------------------- #


def realized_gap(conf: ConflictData, phi: list[int],
                 *, floor: float | None = None) -> dict:
    """Наибольший мультипликативный зазор множества U(phi) в (0, claim_cap].

    floor: минимальный нижний конец зазора (по умолчанию diam V0)."""
    base = conf.geom.diam if floor is None else floor
    ivs = [(0.0, base)]
    for (i, j), pd in conf.pairs.items():
        if phi[i] == phi[j]:
            ivs.extend(pd.intervals)
    ivs.sort()
    best = {"ratio": 1.0, "a": base, "b": base}
    cur_hi = 0.0
    for lo, hi in ivs:
        if lo > cur_hi:
            b_eff = min(lo, conf.claim_cap)
            if b_eff > cur_hi and cur_hi > 0:
                r = b_eff / cur_hi
                if r > best["ratio"]:
                    best = {"ratio": r, "a": cur_hi, "b": b_eff}
        cur_hi = max(cur_hi, hi)
        if cur_hi >= conf.claim_cap:
            break
    return best


# --------------------------------------------------------------------------- #
# главный поиск                                                                #
# --------------------------------------------------------------------------- #


def search_best(conf: ConflictData, k: int, *,
                a_cap_mult: float = 2.0,
                max_a_grid: int = 160,
                eps: float = 1e-9,
                cpsat_timeout: float = 10.0,
                budget_s: float = 600.0,
                heartbeat: float = 20.0,
                verbose: bool = True) -> dict:
    """Максимизация зазора b/a по phi при k цветах.

    Скан a по критическим H-значениям в [diam, a_cap], для каждого a —
    бинарный поиск максимального b по критическим D-значениям, CP-SAT внутри.
    Возвращает словарь с лучшим phi, зазором и журналом статусов."""
    t0 = time.time()
    diam = conf.geom.diam
    a_cap = a_cap_mult * diam

    his = sorted({hi for pd in conf.pairs.values() for _, hi in pd.intervals
                  if diam - 1e-12 <= hi <= a_cap})
    a_grid = [diam] + [h + 1e-9 for h in his]
    if len(a_grid) > max_a_grid:
        idx = np.linspace(0, len(a_grid) - 1, max_a_grid).astype(int)
        a_grid = [a_grid[i] for i in sorted(set(idx))]

    los_all = sorted({lo for pd in conf.pairs.values()
                      for lo, _ in pd.intervals if lo > 1e-9})

    best = {"ratio": 0.0, "phi": None, "a": None, "b": None}
    log = []
    unknowns = 0
    last_beat = t0
    for a in a_grid:
        if time.time() - t0 > budget_s:
            log.append({"event": "budget_exhausted", "a": a})
            break
        if best["ratio"] > 0 and conf.claim_cap / a <= best["ratio"]:
            break                                   # выше по a лучше не станет
        los = [x for x in los_all if x > a] + [conf.claim_cap]
        lo_i, hi_i = 0, len(los) - 1
        # сперва проверить максимально возможное b — если SAT, бинпоиск не нужен
        st, phi = colorable(conf, a, los[hi_i], k,
                            eps=eps, timeout_s=cpsat_timeout)
        if st == "SAT":
            feas_b, feas_phi = los[hi_i], phi
        else:
            # SELF (самоконфликт) зависит от b так же, как UNSAT: при меньшем
            # b интервал может выйти из зазора — продолжаем бинпоиск
            if st == "UNKNOWN":
                unknowns += 1
            hi_i = len(los) - 2
            feas_b, feas_phi = None, None
            while lo_i <= hi_i:
                mid = (lo_i + hi_i) // 2
                st, phi = colorable(conf, a, los[mid], k,
                                    eps=eps, timeout_s=cpsat_timeout)
                if st == "SAT":
                    feas_b, feas_phi = los[mid], phi
                    lo_i = mid + 1
                else:
                    if st == "UNKNOWN":
                        unknowns += 1
                    hi_i = mid - 1
        if feas_phi is not None:
            g = realized_gap(conf, feas_phi)
            log.append({"a": a, "b_search": feas_b, "gap": g})
            if g["ratio"] > best["ratio"]:
                best = {"ratio": g["ratio"], "phi": feas_phi,
                        "a": g["a"], "b": g["b"]}
                if verbose:
                    print(f"    [search] a={a:.4f}: зазор {g['ratio']:.6f} "
                          f"({g['a']:.4f}..{g['b']:.4f})", flush=True)
        else:
            log.append({"a": a, "status": "no_feasible_b"})
        if time.time() - last_beat > heartbeat:
            print(f"    [search] a-скан {a:.3f}/{a_cap:.3f}, "
                  f"best={best['ratio']:.4f}, {time.time() - t0:.0f}s",
                  flush=True)
            last_beat = time.time()
    return {"best": best, "log": log, "unknown_count": unknowns,
            "a_grid_size": len(a_grid), "seconds": time.time() - t0,
            "params": {"k": k, "a_cap_mult": a_cap_mult, "eps": eps,
                       "cpsat_timeout": cpsat_timeout, "budget_s": budget_s,
                       "cutoff": conf.cutoff, "claim_cap": conf.claim_cap}}


# --------------------------------------------------------------------------- #
# независимая верификация найденной раскраски                                  #
# --------------------------------------------------------------------------- #


def verify_coloring(conf: ConflictData, phi: list[int]) -> dict:
    """Пересчёт зазора phi двумя независимыми процедурами.

    Возвращает зазор по GJK-интервалам, зазор по SLSQP + сертифицированным
    нижним границам D и максимальное расхождение процедур."""
    geom = conf.geom
    gjk_gap = realized_gap(conf, phi)
    # пересборка интервалов одноцветных пар SLSQP-ветвью
    max_dev = 0.0
    ivs_cert = [(0.0, geom.diam)]
    Gbasis = conf.H.astype(float) @ geom.basis
    Gred = np.asarray(combigeo.lll_reduce(Gbasis.tolist()), float)
    Gred_inv = np.linalg.inv(Gred)
    mono = [(i, j) for (i, j) in conf.pairs if phi[i] == phi[j]]
    dred = {}
    for i, j in mono:
        delta = (conf.cosets[j] - conf.cosets[i]).astype(float) @ geom.basis
        dred[(i, j)] = delta - np.round(delta @ Gred_inv) @ Gred
    max_dr = max((np.linalg.norm(d) for d in dred.values()), default=0.0)
    gpts = _lattice_points_in_ball(Gbasis, conf.cutoff + max_dr + 1e-9)
    for (i, j) in mono:
        delta = dred[(i, j)]
        ws = gpts + delta
        norms = np.linalg.norm(ws, axis=1)
        sel = ws[norms <= conf.cutoff]
        for w in sel:
            if i == j and np.linalg.norm(w) < 1e-9:
                continue
            lo_gjk, hi = pair_interval(geom, w)
            lo_slsqp, lo_cert = pair_lo_slsqp(geom, w)
            max_dev = max(max_dev, abs(lo_gjk - lo_slsqp))
            ivs_cert.append((lo_cert, hi))
    # зазор по сертифицированным интервалам
    ivs_cert.sort()
    best = {"ratio": 1.0, "a": geom.diam, "b": geom.diam}
    cur_hi = 0.0
    for lo, hi in ivs_cert:
        if lo > cur_hi and cur_hi > 0:
            b_eff = min(lo, conf.claim_cap)
            if b_eff > cur_hi:
                r = b_eff / cur_hi
                if r > best["ratio"]:
                    best = {"ratio": r, "a": cur_hi, "b": b_eff}
        cur_hi = max(cur_hi, hi)
        if cur_hi >= conf.claim_cap:
            break
    return {"gjk_gap": gjk_gap, "cert_gap": best,
            "max_procedure_deviation": max_dev}
