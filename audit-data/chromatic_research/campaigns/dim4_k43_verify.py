"""Независимая перепроверка рекорда chi(R^4) <= 43 (кандидат Н. Глушковой).

Скрипт НИЧЕГО не берёт из voronoi4d/combigeo: вся геометрия строится с нуля
в точной рациональной арифметике, координаты — коэффициенты относительно
базиса Lambda, скалярное произведение <a,b> = a Q b^T. Это второй, полностью
независимый путь к тем же числам, что даёт cert_generic.py.

Что считается точно:
  1. релевантные векторы Вороного — по теореме Вороного: минимумы нормы в
     ненулевых смежных классах Lambda/2Lambda (перебор с запасом);
  2. вершины V0 — решение всех C(30,4) рациональных СЛАУ + фильтр неравенств;
     контроль: |вершин| = 120, центральная симметрия, объём = det(Lambda);
  3. diam(V0)^2 — максимум по парам вершин (точная дробь);
  4. D(Gamma)^2 = min по v in Gamma\\0 от 4*dist(v/2, V0)^2, где dist считается
     ТОЧНО: активное множество берётся из float-QP, а оптимальность
     подтверждается условиями ККТ в дробях (x* допустима, множители >= 0).
     Окно кандидатов |v| <= (1 + d_est)*diam полно, т.к. D(v) >= |v| - diam.

Запуск:  python -m chromatic_research.campaigns.dim4_k43_verify [вход.json] [выход.json]

По умолчанию проверяется заголовочная решётка (эйзенштейнов оптимум,
results/r4_k43_eisenstein_rational.json); первым аргументом можно подать любой
файл того же формата --- например results/r4_k43_rational.json, первую
найденную точку индекса 43.
"""

import itertools
import json
import sys
from fractions import Fraction as F

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from chromatic_research.paths import results_path

N = 4

DEFAULT_INPUT = "r4_k43_eisenstein_rational.json"


def load_input(path=None):
    """Читает форму Грама и переход; устанавливает Q, Qf, M как глобальные."""
    global Q_STR, M, Q, Qf
    src = json.load(open(path if path and "/" in str(path)
                         else results_path(path or DEFAULT_INPUT)))
    Q_STR = src["Q_fractions"]
    M = src["transition"]
    Q = [[F(s) for s in row] for row in Q_STR]
    Qf = np.array([[float(x) for x in row] for row in Q])


load_input()


def dot(a, b):
    """Точное <a, b> = a Q b^T для координатных векторов a, b."""
    return sum(a[i] * Q[i][j] * b[j] for i in range(N) for j in range(N))


def solve_exact(A, b):
    """Единственное решение A x = b в дробях; None, если его нет."""
    m, n = len(A), len(A[0])
    T = [list(map(F, A[i])) + [F(b[i])] for i in range(m)]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if T[i][c] != 0), None)
        if p is None:
            continue
        T[r], T[p] = T[p], T[r]
        pv = T[r][c]
        T[r] = [x / pv for x in T[r]]
        for i in range(m):
            if i != r and T[i][c] != 0:
                f = T[i][c]
                T[i] = [x - f * y for x, y in zip(T[i], T[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if T[i][-1] != 0 and all(x == 0 for x in T[i][:-1]):
            return None
    if r < n:
        return None
    x = [F(0)] * n
    for i, c in enumerate(piv):
        x[c] = T[i][-1]
    return x


def relevant_vectors(window=6):
    """Релевантные векторы Вороного: по паре +-v на каждый класс Lambda/2Lambda."""
    grid = list(itertools.product(range(-window, window + 1), repeat=N))
    rel, degenerate = [], []
    for coset in itertools.product((0, 1), repeat=N):
        if not any(coset):
            continue
        best, reps = None, []
        for c in grid:
            if any((c[i] - coset[i]) % 2 for i in range(N)):
                continue
            n2 = dot([F(x) for x in c], [F(x) for x in c])
            if best is None or n2 < best:
                best, reps = n2, [c]
            elif n2 == best:
                reps.append(c)
        if len(reps) == 2 and tuple(-x for x in reps[0]) == reps[1]:
            rel.append(list(reps[0]))
        else:
            degenerate.append((coset, reps))
    return rel, degenerate


def cell_vertices(half):
    """Все вершины V0 = {x : <x,v> <= off} точным перебором четвёрок граней."""
    verts, seen = [], set()
    for idx in itertools.combinations(range(len(half)), N):
        A = [[sum(Q[i][j] * half[t][0][j] for j in range(N)) for i in range(N)] for t in idx]
        x = solve_exact(A, [half[t][1] for t in idx])
        if x is None:
            continue
        if all(dot(x, v) <= off for v, off in half):
            key = tuple(x)
            if key not in seen:
                seen.add(key)
                verts.append(x)
    return verts


def exact_dist2(p, half, ambient):
    """Точный dist(p, V0)^2: активное множество из float-QP + проверка ККТ в дробях."""
    if all(dot(p, v) <= off for v, off in half):
        return F(0), []
    A_amb = np.array([np.array([float(t) for t in v]) @ Qf for v, _ in half])
    b_amb = np.array([float(off) for _, off in half])
    pf = np.array([float(t) for t in p])
    res = minimize(lambda x: float((x - pf) @ Qf @ (x - pf)), np.zeros(N),
                   jac=lambda x: 2 * (Qf @ (x - pf)),
                   constraints=[{"type": "ineq", "fun": lambda x: b_amb - A_amb @ x,
                                 "jac": lambda x: -A_amb}],
                   method="SLSQP", options={"maxiter": 3000, "ftol": 1e-16})
    active = [i for i in range(len(half)) if abs(b_amb[i] - A_amb[i] @ res.x) < 1e-7]

    def try_active(S):
        """x = p - sum lam_i v_i с <x, v_i> = off_i: ККТ для выпуклой задачи."""
        lam = solve_exact([[dot(half[j][0], half[i][0]) for j in S] for i in S],
                          [dot(p, half[i][0]) - half[i][1] for i in S])
        if lam is None or any(t < 0 for t in lam):
            return None
        x = list(p)
        for j, i in enumerate(S):
            x = [xx - lam[j] * vv for xx, vv in zip(x, half[i][0])]
        if not all(dot(x, v) <= off for v, off in half):
            return None
        w = [a - b for a, b in zip(p, x)]
        return dot(w, w)

    for pool in (active, range(len(half))):
        for r in range(1, N + 1):
            for S in itertools.combinations(pool, r):
                val = try_active(list(S))
                if val is not None:
                    return val, list(S)
    raise RuntimeError("ККТ-сертификат не найден")


def main():
    if len(sys.argv) > 1:
        load_input(sys.argv[1])
    B = np.linalg.cholesky(Qf)
    rel, degenerate = relevant_vectors()
    assert not degenerate, f"вырожденные классы Lambda/2Lambda: {degenerate}"
    half = [([F(s * x) for x in c], dot([F(s * x) for x in c], [F(s * x) for x in c]) / 2)
            for c in rel for s in (1, -1)]
    print(f"релевантных пар: {len(rel)} (ожидалось 15), полупространств: {len(half)}")

    verts = cell_vertices(half)
    print(f"вершин V0: {len(verts)}")
    sym = all(any(all(a == -b for a, b in zip(v, w)) for w in verts) for v in verts)
    vol = ConvexHull(np.array([[float(x) for x in v] for v in verts]) @ B).volume
    det = abs(float(np.linalg.det(B)))
    print(f"центральная симметрия: {sym}; объём V0 = {vol:.9f}, det(Lambda) = {det:.9f}")
    assert sym and abs(vol - det) < 1e-6 * det, "ячейка построена неверно"

    diam2 = max(dot([a - b for a, b in zip(u, w)], [a - b for a, b in zip(u, w)])
                for u, w in itertools.combinations(verts, 2))
    print(f"diam(V0)^2 = {diam2} = {float(diam2):.12f}")

    # окно кандидатов Gamma: |v| <= (1 + d_est) * diam, d_est = 1.01 > d
    d_est = F(101, 100)
    bound2 = float((1 + d_est) ** 2 * diam2)
    sub = np.array(M, float) @ B
    lim = int(np.ceil(np.sqrt(bound2) * np.abs(np.linalg.inv(sub)).sum(axis=0).max())) + 2
    cands = [c for c in itertools.product(range(-lim, lim + 1), repeat=N)
             if any(c) and (np.array(c, float) @ sub) @ (np.array(c, float) @ sub) <= bound2 * (1 + 1e-9)]
    print(f"окно коэффициентов Gamma: +-{lim}; кандидатов: {len(cands)}")

    D2, arg = None, None
    for c in cands:
        v = [sum(F(c[i]) * M[i][j] for i in range(N)) for j in range(N)]
        d2, S = exact_dist2([x / 2 for x in v], half, B)
        if D2 is None or 4 * d2 < D2:
            D2, arg = 4 * d2, (list(c), [str(x) for x in v], S)
    print(f"D(Gamma)^2 = {D2} = {float(D2):.12f}; минимум на v = {arg[1]}")

    ratio2 = D2 / diam2
    print(f"\nd^2 = {ratio2} = {float(ratio2):.12f}")
    print(f"d   = {float(ratio2) ** 0.5:.15f}")
    widths = {L: bool(ratio2 >= F(L) ** 2)
              for L in ("1", "1.003", "1.0037", "1.003714", "1.0038")}
    print("сертифицированные ширины:", widths)

    out = {"index": int(round(abs(float(np.linalg.det(np.array(M, float)))))),
           "Q_fractions": Q_STR, "transition": M,
           "diam2": str(diam2), "D2": str(D2), "d2": str(ratio2),
           "d_float": float(ratio2) ** 0.5, "n_relevant_pairs": len(rel),
           "n_vertices": len(verts), "volume_ok": True, "widths": widths,
           "argmin_gamma_coeffs": arg[0], "argmin_lambda_coords": arg[1],
           "note": "независимая точная перепроверка раскраски индекса 43 в R^4"}
    path = results_path(sys.argv[2] if len(sys.argv) > 2 else "dim4_k43_verify.json")
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("записано:", path)


if __name__ == "__main__":
    main()
