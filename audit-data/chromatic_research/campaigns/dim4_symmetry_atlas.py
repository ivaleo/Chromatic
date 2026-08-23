r"""Атлас симметрий R^4: все обозримые классы конечных автоморфизмов.

Обобщение приёма \S4.1 статьи. Для элемента S in GL_4(Z) конечного порядка
пространство S-инвариантных форм {Q : S Q S^T = Q} тем меньше, чем богаче
действие. Здесь перебираются представители классов Z[S]-модулей ранга 4,
для каждого считается размерность инвариантного пространства, и те классы,
где она мала (<= 5, то есть <= 4 существенных параметра), сканируются
случайным поиском с доводкой; d при этом считается ИСЧЕРПЫВАЮЩИМ перебором
всех подрешёток данного индекса (combigeo.find_optimal), так что
неинвариантные Gamma тоже покрыты.

Представители строятся блочно из сопровождающих матриц круговых многочленов
и матриц перестановок (регулярные представления --- это склеенные модули,
не разлагающиеся в прямую сумму):

    Phi_3 (+) Phi_3   --- Z[w]-модуль ранга 2 (эйзенштейнов, \S4.1)
    Phi_4 (+) Phi_4   --- Z[i]-модуль ранга 2 (гауссов)
    Phi_3 (+) I_2     --- Z[w] (+) Z^2
    C_3 (+) I_1       --- регулярный Z[C_3] (+) Z: склейка, не прямая сумма
    C_4               --- регулярный Z[C_4]
    Phi_5, Phi_8, Phi_12, Phi_10 --- Z[zeta_m]-модули ранга 1
    Phi_6 (+) Phi_3, Phi_6 (+) I_2, Phi_4 (+) I_2, ...

Статус результата --- отрицательный экран [Э] по указанной области.

Запуск: python -m chromatic_research.campaigns.dim4_symmetry_atlas [kmin-kmax] [N]
"""

import json
import sys
import time

import numpy as np
import combigeo
from scipy.optimize import minimize

from chromatic_research.paths import results_path

MAX_DIM = 5          # инвариантных параметров формы (со масштабом)
SEED = 20260823


def companion(coeffs):
    """Сопровождающая матрица многочлена x^d + coeffs[d-1] x^{d-1} + ... ."""
    d = len(coeffs)
    C = np.zeros((d, d), dtype=int)
    C[1:, :-1] = np.eye(d - 1, dtype=int)
    C[:, -1] = [-c for c in coeffs]
    return C


def cycle(n):
    """Матрица циклической перестановки порядка n."""
    P = np.zeros((n, n), dtype=int)
    for i in range(n):
        P[i, (i + 1) % n] = 1
    return P


def blocks(*mats):
    n = sum(m.shape[0] for m in mats)
    out = np.zeros((n, n), dtype=int)
    o = 0
    for m in mats:
        k = m.shape[0]
        out[o:o + k, o:o + k] = m
        o += k
    return out


PHI3 = companion([1, 1])                 # x^2 + x + 1
PHI4 = companion([1, 0])                 # x^2 + 1
PHI6 = companion([1, -1])                # x^2 - x + 1
PHI5 = companion([1, 1, 1, 1])           # x^4 + x^3 + x^2 + x + 1
PHI8 = companion([1, 0, 0, 0])           # x^4 + 1
PHI10 = companion([1, -1, 1, -1])        # x^4 - x^3 + x^2 - x + 1
PHI12 = companion([1, 0, -1, 0])         # x^4 - x^2 + 1
I1, I2 = np.eye(1, dtype=int), np.eye(2, dtype=int)

CANDIDATES = {
    "Z[w]^2 (эйзенштейнов)": blocks(PHI3, PHI3),
    "Z[i]^2 (гауссов)": blocks(PHI4, PHI4),
    "Z[w] (+) Z^2": blocks(PHI3, I2),
    "Z[C3] (+) Z (склейка)": blocks(cycle(3), I1),
    "Z[C4] (регулярный)": cycle(4),
    "Z[zeta5]": PHI5,
    "Z[zeta8]": PHI8,
    "Z[zeta10]": PHI10,
    "Z[zeta12]": PHI12,
    "Z[w] (+) Z[-w]": blocks(PHI3, PHI6),
    "Z[i] (+) Z^2": blocks(PHI4, I2),
    "Z[-w] (+) Z^2": blocks(PHI6, I2),
}


def invariant_basis(S):
    """Базис пространства {Q симм. : S Q S^T = Q} (список матриц 4x4)."""
    idx = [(i, j) for i in range(4) for j in range(i, 4)]
    rows = []
    for (i, j) in idx:
        E = np.zeros((4, 4))
        E[i, j] = E[j, i] = 1.0
        rows.append((S @ E @ S.T - E).ravel())
    A = np.array(rows).T
    _, s, Vt = np.linalg.svd(A)
    null = Vt[np.sum(s > 1e-9):]
    out = []
    for v in null:
        E = np.zeros((4, 4))
        for c, (i, j) in zip(v, idx):
            E[i, j] = E[j, i] = c
        out.append(E)
    return out


def order_of(S, cap=24):
    P = np.eye(len(S), dtype=int)
    for m in range(1, cap + 1):
        P = P @ S
        if np.array_equal(P, np.eye(len(S), dtype=int)):
            return m
    return None


COND_MAX = 60.0     # отсечка вырожденных форм: GJK на них падает в C++ (abort)


def evaluate(t, basis, lo, hi):
    """d по всем индексам [lo, hi] для формы sum t_i B_i (нормировка det = 1).

    Сильно вытянутые формы отбрасываются ДО обращения к C++: при большом числе
    обусловленности GJK не достигает сертификата оптимальности и бросает
    исключение из рабочего потока, а такое исключение убивает процесс целиком
    (uncaught в C++), а не превращается в питоновское. Отсечка по cond и
    предварительная сборка ячейки в главном потоке снимают этот риск; заодно
    вытянутые формы всё равно дают малое d и интереса не представляют.
    """
    Q = sum(c * B for c, B in zip(t, basis))
    Q = 0.5 * (Q + Q.T)
    ev = np.linalg.eigvalsh(Q)
    if ev.min() <= 1e-9:
        return None
    Q = Q / np.linalg.det(Q) ** 0.25
    try:
        B = np.array(combigeo.lll_reduce(np.linalg.cholesky(Q).tolist()))
    except (RuntimeError, np.linalg.LinAlgError):
        return None
    G = B @ B.T
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    if ev.min() <= 1e-9 or ev.max() / ev.min() > COND_MAX:
        return None
    try:
        cell = combigeo.voronoi_cell(B.tolist())          # проверка в главном потоке
        if len(cell.facets) < 8 or not np.isfinite(cell.diameter):
            return None
        res = combigeo.find_optimal_range(B.tolist(), lo, hi, 0, True, 1)
    except (RuntimeError, np.linalg.LinAlgError):
        return None
    return {k: res[k].normalized for k in range(lo, hi + 1)}


def canonical(S, basis):
    """Координаты канонической инвариантной формы Q0 = sum_j S^j (S^j)^T.

    Случайная точка инвариантного конуса почти всегда вырождена, поэтому
    выборка ведётся вокруг Q0 --- заведомо положительно определённой
    S-инвариантной формы.
    """
    m = order_of(S)
    Q0 = sum(np.linalg.matrix_power(S, j) @ np.linalg.matrix_power(S, j).T
             for j in range(m)).astype(float)
    A = np.array([B.ravel() for B in basis]).T
    return np.linalg.lstsq(A, Q0.ravel(), rcond=None)[0]


def main():
    lo, hi = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "31-43").split("-"))
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rng = np.random.default_rng(SEED)
    path = results_path("dim4_symmetry_atlas.json")
    atlas = {}
    if path.exists():                      # докатываем прерванный прогон
        prev = json.load(open(path))
        if prev.get("range") == [lo, hi]:
            atlas = prev.get("atlas", {})
            print(f"продолжаем: уже посчитано классов {len(atlas)}", flush=True)

    def flush():
        json.dump({"range": [lo, hi], "samples": N, "seed": SEED,
                   "max_invariant_dim": MAX_DIM, "atlas": atlas},
                  open(path, "w"), ensure_ascii=False, indent=1)

    for name, S in CANDIDATES.items():
        if name in atlas:
            continue
        m = order_of(S)
        basis = invariant_basis(S)
        r = len(basis)
        head = (f"{name}: порядок {m}, инвариантных параметров {r}, "
                f"char {np.poly(S).round(3).tolist()}")
        if m is None or r > MAX_DIM:
            print(head + "  --- пропущено (слишком велико)", flush=True)
            atlas[name] = {"order": m, "dim": r, "skipped": True}
            flush()
            continue
        t0, rows = time.time(), []
        base = canonical(S, basis)
        scale = max(1e-9, float(np.abs(base).max()))
        for i in range(N):
            sigma = 0.15 + 0.85 * (i % 8) / 7.0        # от лёгких до сильных сдвигов
            t = base + sigma * scale * rng.normal(size=r)
            val = evaluate(t, basis, lo, hi)
            if val:
                rows.append((val, t.tolist()))
        val = evaluate(base, basis, lo, hi)             # сама каноническая форма
        if val:
            rows.append((val, base.tolist()))
        if not rows:
            print(head + "  --- положительно определённых форм не найдено", flush=True)
            atlas[name] = {"order": m, "dim": r, "empty": True}
            flush()
            continue
        best = {}
        for k in range(lo, hi + 1):
            rows.sort(key=lambda x: -x[0][k])
            best[str(k)] = {"d": rows[0][0][k], "t": rows[0][1],
                            "admissible": bool(rows[0][0][k] >= 1.0)}
        # доводка только у самого перспективного индекса класса (экран, не рекорд)
        k_top = max(range(lo, hi + 1), key=lambda k: best[str(k)]["d"])

        def neg(tt):
            v = evaluate(tt, basis, k_top, k_top)
            return -v[k_top] if v else 0.0

        res = minimize(neg, np.array(best[str(k_top)]["t"]), method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-10,
                                "maxiter": 250, "maxfev": 250})
        if -res.fun > best[str(k_top)]["d"]:
            best[str(k_top)] = {"d": -res.fun, "t": [float(x) for x in res.x],
                                "admissible": bool(-res.fun >= 1.0)}
        line = "  ".join(f"{k}:{best[str(k)]['d']:.4f}" for k in range(lo, hi + 1))
        print(head + f"  [{time.time()-t0:.0f}s]\n    {line}", flush=True)
        atlas[name] = {"order": m, "dim": r, "samples": len(rows),
                       "requested": N, "best": best}
        flush()
    flush()
    print("записано:", path)


if __name__ == "__main__":
    main()
