"""Симметрийно-ограниченные семейства решёток в R^4.

Идея кампании (23.08.2026): рекордная решётка индекса 43 оказалась
эйзенштейновой --- элемент S порядка 3 из Aut(Lambda) имеет характеристический
многочлен (x^2+x+1)^2, то есть Z^4 --- это Z[w]-модуль ранга 2, и вся Aut ~ Z/6
сохраняет Gamma. Пространство S-инвариантных форм 4-мерно (против 10 у общей
формы), а с точностью до масштаба --- ТРЁХмерно. Поэтому вместо CMA-ES по
10 параметрам можно вести (почти) исчерпывающий скан по трём.

Общая конструкция: Lambda = O-модуль ранга r в R^4, где O --- порядок в мнимом
квадратичном поле (Z[w], Z[i]) при r=2 либо кольцо целых циклотомического поля
степени 4 (Z[z5], Z[z8], Z[z12]) при r=1. При r=1 форма жёсткая (решётка одна
с точностью до подобия), при r=2 --- три существенных параметра.

Инвариантные подрешётки --- это O-подмодули; их индексы суть нормы идеалов, а
сами подмодули для циклического фактора Lambda/Gamma = O/pi перечисляются явно.
"""

import itertools
import math

import numpy as np

# --------------------------------------------------------------------------------
# Z[w]: арифметика и эрмитовы формы ранга 2
# --------------------------------------------------------------------------------

OMEGA = complex(-0.5, 0.5 * math.sqrt(3.0))


def zw_mul(p, q):
    """Умножение в Z[w] в координатах (x, y) = x + y*w, w^2 = -1 - w."""
    (x1, y1), (x2, y2) = p, q
    return (x1 * x2 - y1 * y2, x1 * y2 + x2 * y1 - y1 * y2)


def zw_norm(p):
    x, y = p
    return x * x - x * y + y * y


def zw_elements_of_norm(k, lim=None):
    """Все pi in Z[w] с N(pi) = k."""
    lim = lim or int(math.isqrt(4 * k) + 2)
    return [(x, y) for x in range(-lim, lim + 1) for y in range(-lim, lim + 1)
            if zw_norm((x, y)) == k]


def hermitian_gram(a, b, c):
    """Грам Z-базиса (e1, w e1, e2, w e2) эрмитовой формы [[a, c], [conj c, b]].

    Вещественное скалярное произведение <u, v> = Re h(u, v); форма положительно
    определена при a > 0 и a*b > |c|^2.
    """
    H = np.array([[a, c], [np.conj(c), b]], dtype=complex)
    cols = np.array([[1, OMEGA, 0, 0], [0, 0, 1, OMEGA]], dtype=complex)
    G = np.real(np.conj(cols).T @ H @ cols)
    return 0.5 * (G + G.T)


def hnf_rows(gens, n=4):
    """Эрмитова нормальная форма по строкам; None, если ранг < n."""
    A = [list(map(int, g)) for g in gens]
    m, r = len(A), 0
    for c in range(n):
        if not any(A[i][c] for i in range(r, m)):
            continue
        while True:
            p = min((i for i in range(r, m) if A[i][c] != 0), key=lambda i: abs(A[i][c]))
            A[r], A[p] = A[p], A[r]
            clean = True
            for i in range(r + 1, m):
                if A[i][c]:
                    q = A[i][c] // A[r][c]
                    A[i] = [x - q * y for x, y in zip(A[i], A[r])]
                    if A[i][c]:
                        clean = False
            if clean:
                break
        if A[r][c] < 0:
            A[r] = [-x for x in A[r]]
        for i in range(r):
            q = A[i][c] // A[r][c]
            A[i] = [x - q * y for x, y in zip(A[i], A[r])]
        r += 1
        if r == m:
            break
    A = [row for row in A if any(row)]
    return np.array(A, dtype=float) if len(A) == n else None


def zw_submodules(k):
    """Z[w]-подмодули ранга 2 индекса k с циклическим фактором Lambda/Gamma.

    Фактор изоморфен O/pi при N(pi) = k, поэтому Gamma содержит pi*Lambda и
    отвечает 𝔽-прямой в Lambda/pi*Lambda: представители (1 : t) и (0 : 1).
    Возвращает список целочисленных базисов 4x4 (строки --- координаты в Lambda).
    """
    out, seen = [], set()
    w = (0, 1)
    for pi in zw_elements_of_norm(k):
        reps = [((1, 0), (t, 0)) for t in range(k)] + [((0, 0), (1, 0))]
        for u in reps:
            gens = [[*u[0], *u[1]],
                    [*zw_mul(w, u[0]), *zw_mul(w, u[1])]]
            for e in (((1, 0), (0, 0)), ((0, 0), (1, 0))):
                pe = (zw_mul(pi, e[0]), zw_mul(pi, e[1]))
                gens.append([*pe[0], *pe[1]])
                gens.append([*zw_mul(w, pe[0]), *zw_mul(w, pe[1])])
            basis = hnf_rows(gens)
            if basis is None or abs(round(float(np.linalg.det(basis)))) != k:
                continue
            key = tuple(map(tuple, basis.astype(int).tolist()))
            if key not in seen:
                seen.add(key)
                out.append(basis)
    return out


def zw_norm_indices(lo, hi):
    """Индексы в [lo, hi], представимые как нормы элементов Z[w]."""
    return [k for k in range(lo, hi + 1) if zw_elements_of_norm(k)]


# --------------------------------------------------------------------------------
# Z[i]: то же самое для порядка Гаусса
# --------------------------------------------------------------------------------

def zi_mul(p, q):
    (x1, y1), (x2, y2) = p, q
    return (x1 * x2 - y1 * y2, x1 * y2 + x2 * y1)


def zi_norm(p):
    x, y = p
    return x * x + y * y


def zi_elements_of_norm(k):
    lim = int(math.isqrt(k)) + 1
    return [(x, y) for x in range(-lim, lim + 1) for y in range(-lim, lim + 1)
            if zi_norm((x, y)) == k]


def gauss_gram(a, b, c):
    """Грам Z-базиса (e1, i e1, e2, i e2) эрмитовой формы над Z[i]."""
    H = np.array([[a, c], [np.conj(c), b]], dtype=complex)
    cols = np.array([[1, 1j, 0, 0], [0, 0, 1, 1j]], dtype=complex)
    G = np.real(np.conj(cols).T @ H @ cols)
    return 0.5 * (G + G.T)


def zi_submodules(k):
    """Z[i]-подмодули ранга 2 индекса k с циклическим фактором."""
    out, seen = [], set()
    im = (0, 1)
    for pi in zi_elements_of_norm(k):
        reps = [((1, 0), (t, 0)) for t in range(k)] + [((0, 0), (1, 0))]
        for u in reps:
            gens = [[*u[0], *u[1]], [*zi_mul(im, u[0]), *zi_mul(im, u[1])]]
            for e in (((1, 0), (0, 0)), ((0, 0), (1, 0))):
                pe = (zi_mul(pi, e[0]), zi_mul(pi, e[1]))
                gens.append([*pe[0], *pe[1]])
                gens.append([*zi_mul(im, pe[0]), *zi_mul(im, pe[1])])
            basis = hnf_rows(gens)
            if basis is None or abs(round(float(np.linalg.det(basis)))) != k:
                continue
            key = tuple(map(tuple, basis.astype(int).tolist()))
            if key not in seen:
                seen.add(key)
                out.append(basis)
    return out


# --------------------------------------------------------------------------------
# быстрый оценщик d
# --------------------------------------------------------------------------------

def lattice_points_within(basis, bound):
    """Все v = c @ basis с 0 < |v| <= bound (по одному из пары +-v); сферный декодер."""
    basis = np.asarray(basis, dtype=float)
    n = len(basis)
    bs = basis.astype(float).copy()
    mu = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            mu[i, j] = (basis[i] @ bs[j]) / (bs[j] @ bs[j])
            bs[i] = bs[i] - mu[i, j] * bs[j]
    bn2 = np.array([b @ b for b in bs])
    b2, out, coeffs = bound * bound, [], [0] * n

    def descend(level, partial2):
        if level == 0:
            for c in coeffs:
                if c > 0:
                    break
                if c < 0:
                    return
            else:
                return
            v = np.array(coeffs, dtype=float) @ basis
            if v @ v <= b2 + 1e-9:
                out.append(v)
            return
        j = level - 1
        center = sum(coeffs[i] * mu[i, j] for i in range(j + 1, n))
        rem = b2 - partial2
        if rem < -1e-9:
            return
        rad = math.sqrt(max(0.0, rem) / max(bn2[j], 1e-300))
        for c in range(math.ceil(-center - rad - 1e-9), math.floor(-center + rad + 1e-9) + 1):
            coeffs[j] = c
            descend(j, partial2 + (c + center) ** 2 * bn2[j])
        coeffs[j] = 0

    descend(n, 0.0)
    return out


def evaluate(cell, B, sub_int, diam, cap=None):
    """d = D/diam для подрешётки sub_int (целые координаты в Lambda).

    Ячейка строится один раз на решётку и переиспользуется --- это на два
    порядка быстрее, чем min_color_distance на каждую подрешётку.
    cap: если текущий минимум уже ниже cap, можно прервать (ранний выход).
    """
    import combigeo
    sub = np.asarray(sub_int, dtype=float) @ B
    sub = np.array(combigeo.lll_reduce(sub.tolist()))
    best = float("inf")
    for v in sorted(lattice_points_within(sub, 2.02 * diam), key=lambda v: v @ v):
        if float(np.linalg.norm(v)) - diam >= best:
            break
        best = min(best, 2.0 * combigeo.distance_to_cell((0.5 * v).tolist(), cell))
        if cap is not None and best < cap * diam:
            return best / diam
    return best / diam
