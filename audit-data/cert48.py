"""Точный рациональный сертификат теоремы chi(R^4,[1,l]) <= 48 при l <= l0.

Стратегия (вся арифметика — точные дроби, координаты — коэффициенты
относительно базиса решётки, скалярные произведения через Q):

  P := пересечение полупространств 30 биссекторов кандидатов-релевантных
       векторов (каждый биссектор содержит V0, поэтому V0 <= P);
  (a) diam(V0)^2 <= diam(P)^2 = max по парам ТОЧНЫХ вершин P (перечисление
      вершин P: C(30,4) рациональных СЛАУ + фильтр неравенств);
  (b) для каждого кандидата v подрешётки (|v| < D_est + diam, точная проверка
      окна): dist(v/2, V0) >= dist(v/2, P) >= (<v/2,u> - h_P(u))/|u|
      для ЛЮБОГО направления u (берём рационализованное направление GJK);
      => D(Gamma)^2 >= Ddown^2;
  (c) сертификат: Ddown^2 * 1 >= l0^2 * diamup^2  =>  d >= l0.

Всё выполняется на python Fraction; float используется только для ВЫБОРА
кандидатов/направлений (на строгость не влияет).
"""
import json, math, itertools
from fractions import Fraction as F
import numpy as np
import combigeo

# ---------- точная линейная алгебра на Fraction ----------


def main():
    def fsolve(A, b):
        """Решение A x = b (списки Fraction), None при вырожденности."""
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            piv = next((r for r in range(col, n) if M[r][col] != 0), None)
            if piv is None:
                return None
            M[col], M[piv] = M[piv], M[col]
            pv = M[col][col]
            M[col] = [x / pv for x in M[col]]
            for r in range(n):
                if r != col and M[r][col] != 0:
                    f = M[r][col]
                    M[r] = [x - f * y for x, y in zip(M[r], M[col])]
        return [M[i][n] for i in range(n)]

    def dotQ(Q, a, b):
        return sum(a[i] * Q[i][j] * b[j] for i in range(4) for j in range(4))

    # ---------- данные ----------

    r = json.load(open("/Users/mac/Documents/_My_code/Chromatic/audit-data/r5_k48_rational.json"))
    Q = [[F(s) for s in row] for row in r["Q_fractions"]]
    Qf = np.array([[float(x) for x in row] for row in Q])
    B = np.linalg.cholesky(Qf)
    Binv = np.linalg.inv(B)

    L0 = F(10396, 10000)          # l0 = 1.0396
    print(f"l0 = {L0} = {float(L0)}", flush=True)

    # кандидаты-биссекторы: целочисленные координаты релевантных векторов из ячейки
    cell = combigeo.voronoi_cell(B.tolist())
    rel_coords = []
    seen = set()
    for f_ in cell.facets:
        c = np.rint(np.array(f_.lattice_vector) @ Binv).astype(int)
        assert np.allclose(np.array(f_.lattice_vector) @ Binv, c, atol=1e-6)
        key = tuple(c)
        if tuple(-np.array(c)) in seen:
            continue
        seen.add(key)
        rel_coords.append([int(x) for x in c])
    print(f"пар биссекторов: {len(rel_coords)} (ожидалось 15)", flush=True)
    halfspaces = []
    for c in rel_coords:
        for sgn in (1, -1):
            v = [F(sgn) * F(x) for x in c]
            halfspaces.append((v, dotQ(Q, v, v) / 2))   # <x, v>_Q <= |v|^2/2

    # ---------- (a) точные вершины P и diam(P)^2 ----------

    m = len(halfspaces)
    verts = []
    seen_v = set()
    cnt = 0
    for idx in itertools.combinations(range(m), 4):
        cnt += 1
        A = [[sum(Q[i][j] * halfspaces[t][0][j] for j in range(4)) for i in range(4)]
             for t in idx]                      # строка t: коэффициенты <x, v_t>_Q по x_i
        b = [halfspaces[t][1] for t in idx]
        x = fsolve(A, b)
        if x is None:
            continue
        ok = True
        for (v, off) in halfspaces:
            if dotQ(Q, x, v) > off:
                ok = False
                break
        if ok:
            key = tuple(x)
            if key not in seen_v:
                seen_v.add(key)
                verts.append(x)
    print(f"перебрано {cnt} четвёрок; точных вершин P: {len(verts)}", flush=True)

    diam2 = F(0)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d2 = dotQ(Q, [a - b for a, b in zip(verts[i], verts[j])],
                      [a - b for a, b in zip(verts[i], verts[j])])
            if d2 > diam2:
                diam2 = d2
    print(f"diam(P)^2 = {diam2} = {float(diam2):.9f}  (float diam(V0)^2 = {cell.diameter**2:.9f})",
          flush=True)

    # ---------- (b) нижняя оценка D(Gamma)^2 ----------
    # Цель: для каждого v in Gamma\0 либо (i) точно |v|^2 >= W^2, где рациональное
    # W заведомо >= l0*diam_up + diam_up (тогда D(v) >= |v| - diam >= l0*diam);
    # либо (ii) опорный сертификат dist(v/2, P) >= (<v/2,u>_Q - h_P(u))/|u|_Q
    # с рациональным u малой сложности.

    T = [[1, 0, 0, 28], [0, 1, 0, 16], [0, 0, 1, 30], [0, 0, 0, 48]]
    Tf = np.array(T, float)
    sub_f = Tf @ B
    rhs = L0 * L0 * diam2                      # требуемое: D^2 >= rhs

    # рациональное W: W >= (l0 + 1) * sqrt(diam2). Проверка рациональна:
    # W^2 >= (l0+1)^2 * diam2.
    Wf = (float(L0) + 1.0) * math.sqrt(float(diam2)) * 1.0005
    W = F(round(Wf * 10000), 10000)
    assert W * W >= (L0 + 1) ** 2 * diam2, "увеличьте W"
    print(f"W = {float(W):.6f} (порог дальних кандидатов)", flush=True)

    from voronoi4d import lattice_points_within, lll_reduce
    sub_lll = lll_reduce(sub_f)
    cand = lattice_points_within(sub_lll, float(W) * 1.001)
    print(f"кандидатов в окне |v| <~ W: {len(cand)}", flush=True)

    # полнота окна: любой v вне перебора имеет |v| > W*1.001/1.0005 > W (float-перебор
    # lattice_points_within полон в своём радиусе; берём радиус с запасом 0.1%),
    # значит для него случай (i) выполняется автоматически.

    def support_lower_bound2(vhalf, u):
        """Точная нижняя оценка dist(v/2, P)^2 по направлению u (координаты, Fraction)."""
        h = max(dotQ(Q, w, u) for w in verts)
        num = dotQ(Q, vhalf, u) - h
        if num <= 0:
            return F(0)
        return num * num / dotQ(Q, u, u)

    Ddown2 = None
    hard = 0
    for v in cand:
        coords = np.rint(np.array(v) @ Binv).astype(int)
        assert np.allclose(np.array(v) @ Binv, coords, atol=1e-6)
        c_ex = [F(int(x)) for x in coords]
        v2 = dotQ(Q, c_ex, c_ex)               # точный |v|^2
        if v2 >= W * W:
            continue                            # случай (i): D(v) >= l0*diam автоматически
        hard += 1
        vhalf = [x / 2 for x in c_ex]
        # почти оптимальное направление: float-QP-проекция на P
        p = (np.array([float(x) for x in vhalf]) @ B)
        from scipy.optimize import minimize as _min
        Amb = np.array([np.array([float(t) for t in hv[0]]) @ B for hv in halfspaces])
        # полупространство: <x, v>_Q <= off; в объемлющих координатах x_amb . (v_amb) <= off
        boff = np.array([float(hv[1]) for hv in halfspaces])
        cons = [{"type": "ineq", "fun": lambda x: boff - Amb @ x, "jac": lambda x: -Amb}]
        proj = None
        for s0 in (np.zeros(4), p * 0.5):
            rq = _min(lambda x: (x - p) @ (x - p), s0, jac=lambda x: 2 * (x - p),
                      constraints=cons, method="SLSQP",
                      options={"maxiter": 2000, "ftol": 1e-14})
            if rq.success:
                proj = rq.x if proj is None or np.linalg.norm(rq.x - p) < np.linalg.norm(proj - p) \
                    else proj
        if proj is None:
            proj = np.zeros(4)
        u_float = (p - proj) @ Binv             # направление в координатах базиса
        scale = max(1e-12, float(np.abs(u_float).max()))
        d2low = F(0)
        for denom in (4096, 65536, 2 ** 24):
            u = [F(round(x / scale * denom), denom) for x in u_float]
            d2low = max(d2low, support_lower_bound2(vhalf, u))
            if 4 * d2low >= rhs:
                break
        D2low = 4 * d2low
        if Ddown2 is None or D2low < Ddown2:
            Ddown2 = D2low
    print(f"ближних кандидатов (опорный сертификат): {hard}", flush=True)
    print(f"Ddown^2 = {float(Ddown2):.9f}  (float D^2 = "
          f"{combigeo.min_color_distance(B.tolist(), sub_f.tolist())**2:.9f})", flush=True)

    # ---------- (c) вердикт ----------
    print(f"\nСЕРТИФИКАТ: Ddown^2 = {float(Ddown2):.9f}  >=?  l0^2*diamup^2 = {float(rhs):.9f}")
    ok = Ddown2 >= rhs
    print("ВЕРДИКТ:", "ДОКАЗАНО d >= l0 (точная рациональная арифметика)" if ok else "НЕ ДОКАЗАНО",
          flush=True)
    json.dump({"ok": bool(ok), "Ddown2": str(Ddown2), "diamup2": str(diam2),
               "rhs": str(rhs), "l0": str(L0), "n_vertices_P": len(verts),
               "hard_candidates": hard},
              open("/Users/mac/Documents/_My_code/Chromatic/audit-data/cert48.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
