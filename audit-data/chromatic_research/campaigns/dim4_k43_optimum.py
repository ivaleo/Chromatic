"""Оптимум эйзенштейнова семейства при индексе 43: точная алгебраическая модель.

Рационализованная точка теоремы (results/r4_k43_eisenstein_rational.json) ---
огрубление максимума d по трёхпараметрическому семейству \\S4.1. Здесь этот
максимум описывается точно.

Наблюдение, которое всё решает: в оптимуме РОВНО ЧЕТЫРЕ Z/3-орбиты векторов
Gamma (двенадцать пар +-v) дают одно и то же D. Четыре равные величины --- это
три уравнения на три параметра, поэтому точка изолирована и никакого условия
стационарности не нужно. В окрестности оптимума комбинаторика заморожена
(те же релевантные векторы, та же пара вершин диаметра, те же активные грани
проекций), поэтому diam^2 и все четыре D^2 --- явные рациональные функции
(b, u, s), а система --- три многочлена степеней 5, 6, 5.

Что делает скрипт:
  1. снимает комбинаторику в численном оптимуме (сетка + Нелдер--Мид);
  2. строит символические diam^2(b,u,s) и D_i^2(b,u,s);
  3. группирует связку по орбитам, разделяя их значением в возмущённой точке;
  4. решает систему nsolve'ом с 80 верными знаками;
  5. проверяет PSLQ, нет ли у d^2 минимального многочлена малой степени.

Запуск: python -m chromatic_research.campaigns.dim4_k43_optimum
"""

import itertools
import json

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import minimize

from chromatic_research.campaigns.dim4_eisenstein_scan import make_lattice, refine, scan
from chromatic_research.core.eisenstein4 import zw_submodules
from chromatic_research.paths import results_path

B_SYM, U_SYM, S_SYM = sp.symbols("b u s", positive=True)


def gram_sym():
    """S-инвариантная форма при a = 1 (см. (2) статьи)."""
    b, u, s = B_SYM, U_SYM, S_SYM
    return sp.Matrix([[1, sp.Rational(-1, 2), u, -u / 2 - s],
                      [sp.Rational(-1, 2), 1, -u / 2 + s, u],
                      [u, -u / 2 + s, b, -b / 2],
                      [-u / 2 - s, u, -b / 2, b]])


def numeric_optimum(passes=6):
    """Численный максимум d по семейству и отвечающая ему подрешётка.

    Доводка повторяется несколько раз: одного прохода Нелдера--Мида мало ---
    при остаточной ошибке 10^-8 четвёртая орбита связки ещё отделена, и модель
    получается недоопределённой (два уравнения на три параметра вместо трёх).
    """
    subs = zw_submodules(43)
    (_, p_grid), tops = scan(subs, 24, 10, 8)
    d_best, p_best = 0.0, p_grid
    for _, p in tops:
        d, p2 = refine(subs, p)
        if d > d_best:
            d_best, p_best = d, p2
    for _ in range(passes):
        d, p2 = refine(subs, p_best, tol=1e-14)
        if d >= d_best:
            d_best, p_best = d, p2
    a, rho, th = p_best
    c = rho * complex(np.cos(th), np.sin(th))
    b = (1 + abs(c) ** 2) / a
    return d_best, (b / a, c.real / a, np.sqrt(3) / 2 * c.imag / a), make_lattice(*p_best), subs


def combinatorics(Qf, sub_int, B):
    """Релевантные векторы, четвёрки граней вершин диаметра, активные множества."""
    def dot(x, y):
        return float(np.array(x, float) @ Qf @ np.array(y, float))

    rel = []
    grid = list(itertools.product(range(-3, 4), repeat=4))
    for coset in itertools.product((0, 1), repeat=4):
        if not any(coset):
            continue
        best, rep = None, None
        for c in grid:
            if any((c[i] - coset[i]) % 2 for i in range(4)):
                continue
            n2 = dot(c, c)
            if best is None or n2 < best - 1e-12:
                best, rep = n2, list(c)
        rel.append(rep)
    H = [[sgn * x for x in c] for c in rel for sgn in (1, -1)]
    A = np.array([np.array(h, float) @ Qf for h in H])
    off = np.array([dot(h, h) / 2 for h in H])

    verts = []
    for idx in itertools.combinations(range(len(H)), 4):
        M = A[list(idx)]
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        x = np.linalg.solve(M, off[list(idx)])
        if np.all(A @ x <= off + 1e-9):
            verts.append((x, idx))
    uniq = []
    for x, idx in verts:
        if not any(np.allclose(x, y, atol=1e-9) for y, _ in uniq):
            uniq.append((x, idx))
    diam2, pair = 0.0, None
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            w = uniq[i][0] - uniq[j][0]
            if w @ Qf @ w > diam2:
                diam2, pair = float(w @ Qf @ w), (i, j)

    Binv = np.linalg.inv(B)
    tied = []
    for c in itertools.product(range(-6, 7), repeat=4):
        if not any(c):
            continue
        v = np.array(c, float) @ sub_int
        if v @ v > (2.02 * np.sqrt(diam2)) ** 2:
            continue
        lam = np.rint(v @ Binv).astype(int)
        p = np.array(lam, float) / 2
        r = minimize(lambda x: float((x - p) @ Qf @ (x - p)), np.zeros(4),
                     jac=lambda x: 2 * (Qf @ (x - p)),
                     constraints=[{"type": "ineq", "fun": lambda x: off - A @ x,
                                   "jac": lambda x: -A}],
                     method="SLSQP", options={"maxiter": 5000, "ftol": 1e-18})
        act = [i for i in range(len(H)) if abs(off[i] - A[i] @ r.x) < 1e-7]
        tied.append((4 * float((r.x - p) @ Qf @ (r.x - p)) / diam2,
                     [int(t) for t in lam], act))
    tied.sort()
    best = tied[0][0]
    reps, seen = [], set()
    for val, lam, act in tied:
        if val - best > 1e-9 or tuple(-x for x in lam) in seen:
            continue
        seen.add(tuple(lam))
        reps.append((lam, act))
    return H, uniq[pair[0]][1], uniq[pair[1]][1], reps, diam2


def main():
    d_num, (b0, u0, s0), B, subs = numeric_optimum()
    print(f"численный оптимум d = {d_num:.12f}; (b,u,s) = ({b0!r}, {u0!r}, {s0!r})", flush=True)
    Qs = gram_sym()
    at0 = {B_SYM: sp.Float(b0, 30), U_SYM: sp.Float(u0, 30), S_SYM: sp.Float(s0, 30)}
    Qf = np.array(Qs.subs(at0).evalf(), float)

    T = None
    cell_best = -1.0
    import combigeo
    cell = combigeo.voronoi_cell(B.tolist())
    for S in subs:
        from chromatic_research.core.eisenstein4 import evaluate as ev
        val = ev(cell, B, S, cell.diameter)
        if val > cell_best:
            cell_best, T = val, S.astype(int)
    print("переход:", T.tolist(), flush=True)

    H, idxA, idxB, reps, diam2_num = combinatorics(Qf, T @ B, B)
    print(f"релевантных пар {len(H)//2}; пар в связке {len(reps)}; "
          f"diam^2(числ) = {diam2_num:.12f}", flush=True)

    def dot(x, y):
        return (sp.Matrix(x).T * Qs * sp.Matrix(y))[0, 0]

    def vertex(idx):
        M = sp.Matrix([[dot(H[t], [1 if i == j else 0 for i in range(4)])
                        for j in range(4)] for t in idx])
        return M.solve(sp.Matrix([dot(H[t], H[t]) / 2 for t in idx]))

    pA, pB = vertex(idxA), vertex(idxB)
    diam2 = sp.cancel(dot(list(pA - pB), list(pA - pB)))

    def D2(lam, act):
        w = [H[i] for i in act]
        M = sp.Matrix([[dot(wi, wj) for wj in w] for wi in w])
        p = [sp.Rational(t, 2) for t in lam]
        l = M.solve(sp.Matrix([dot(p, wi) - dot(wi, wi) / 2 for wi in w]))
        return 4 * (l.T * M * l)[0, 0]

    pert = {B_SYM: sp.Float(b0 + 0.004, 30), U_SYM: sp.Float(u0 - 0.003, 30),
            S_SYM: sp.Float(s0 + 0.002, 30)}
    orbits = {}
    for lam, act in reps:
        e = D2(lam, act)
        key = round(float(sp.N(e.subs(pert) / diam2.subs(pert), 25)), 12)
        orbits.setdefault(key, (e, lam))
    E = [v[0] for v in orbits.values()]
    print(f"различных орбит в связке: {len(E)}"
          f" (значит уравнений {len(E)-1} на 3 параметра)", flush=True)
    if len(E) != 4:
        raise RuntimeError(
            f"ожидались 4 орбиты (три уравнения на три параметра), найдено {len(E)}: "
            "численный оптимум не доведён — увеличьте passes")

    F = [sp.Poly(sp.numer(sp.cancel(sp.together(E[i] - E[0]))),
                 [B_SYM, U_SYM, S_SYM]) for i in range(1, len(E))]
    degs = [f.total_degree() for f in F]
    print("степени уравнений:", degs, flush=True)

    mp.mp.dps = 90
    sol = sp.nsolve([f.as_expr() for f in F], [B_SYM, U_SYM, S_SYM],
                    [sp.Float(b0, 25), sp.Float(u0, 25), sp.Float(s0, 25)],
                    prec=85, tol=mp.mpf(10) ** -80)
    bs, us, ss = (sp.N(x, 80) for x in sol)
    at = {B_SYM: bs, U_SYM: us, S_SYM: ss}
    d2 = sp.N(E[0].subs(at) / diam2.subs(at), 80)
    print(f"b* = {sp.N(bs, 42)}\nu* = {sp.N(us, 42)}\ns* = {sp.N(ss, 42)}", flush=True)
    print(f"d^2 = {sp.N(d2, 42)}\nd   = {sp.N(sp.sqrt(d2), 42)}")

    x = mp.mpf(str(d2))
    found = None
    for deg in (2, 3, 4, 5, 6, 8):
        r = mp.pslq([x ** k for k in range(deg + 1)], maxcoeff=10 ** 20, maxsteps=200000)
        if r:
            resid = abs(sum(r[k] * x ** k for k in range(deg + 1)))
            # настоящее соотношение обязано занулять с точностью счёта,
            # а не «на глазок»: артефакт PSLQ узнаётся по остатку
            if resid < mp.mpf(10) ** -70:
                found = (deg, [int(t) for t in r])
                break
    print("минимальный многочлен d^2:",
          found if found else "не найден при степени <= 8 и высоте <= 1e20", flush=True)

    out = {"index": 43, "transition": T.tolist(),
           "orbits_in_tie": len(E), "pairs_in_tie": len(reps),
           "system_degrees": degs,
           "b_star": str(sp.N(bs, 60)), "u_star": str(sp.N(us, 60)),
           "s_star": str(sp.N(ss, 60)),
           "d_squared": str(sp.N(d2, 60)), "d": str(sp.N(sp.sqrt(d2), 60)),
           "minimal_polynomial": found,
           "note": ("оптимум эйзенштейнова семейства k=43: изолированное решение "
                    "системы из трёх многочленов, комбинаторика заморожена")}
    path = results_path("dim4_k43_optimum.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("записано:", path, flush=True)


if __name__ == "__main__":
    main()
