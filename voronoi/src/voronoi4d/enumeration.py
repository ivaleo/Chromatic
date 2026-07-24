"""Точный перебор коротких векторов решётки (sphere decoding).

Заменяет эвристическое окно коэффициентов ±limits: перечисляются ВСЕ векторы
решётки v с |v| <= bound (по одному из каждой пары ±v), с отсечением по
частичной норме в ортогонализованном базисе. Полнота не зависит от базиса,
но узкие границы перебора требуют LLL-приведённого базиса.
"""

import math

import numpy as np

# --------------------------------------------------------------------------------


def _gram_schmidt(basis):
    n = len(basis)
    b_star = basis.astype(float).copy()
    mu = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            denom = b_star[j] @ b_star[j]
            mu[i, j] = (basis[i] @ b_star[j]) / denom
            b_star[i] = b_star[i] - mu[i, j] * b_star[j]
    return b_star, mu


def lattice_points_within(basis, bound):
    """Все векторы решётки v = c @ basis с 0 < |v| <= bound, по одному из пары ±v.

    :param basis: базис решётки (numpy.ndarray, строки — векторы), желательно
                  LLL-приведённый (для узких границ перебора; полнота — при любом).
    :param bound: радиус перебора.
    :return: список numpy-векторов.
    """
    basis = np.asarray(basis, dtype=float)
    n = len(basis)
    b_star, mu = _gram_schmidt(basis)
    bn2 = np.array([b @ b for b in b_star])
    bound2 = bound * bound
    out = []
    coeffs = [0] * n

    def descend(level, partial2):
        if level == 0:
            # канонический представитель пары (v, -v): первый ненулевой
            # коэффициент положителен; нулевой вектор исключается
            for c in coeffs:
                if c > 0:
                    break
                if c < 0:
                    return
            else:
                return
            v = np.array(coeffs, dtype=float) @ basis
            if v @ v <= bound2 + 1e-9:
                out.append(v)
            return
        j = level - 1
        center = sum(coeffs[i] * mu[i, j] for i in range(j + 1, n))
        remaining2 = bound2 - partial2
        if remaining2 < -1e-9:
            return
        radius = math.sqrt(max(0.0, remaining2) / max(bn2[j], 1e-300))
        lo = math.ceil(-center - radius - 1e-9)
        hi = math.floor(-center + radius + 1e-9)
        for c in range(lo, hi + 1):
            coeffs[j] = c
            add2 = (c + center) ** 2 * bn2[j]
            descend(j, partial2 + add2)
        coeffs[j] = 0

    descend(n, 0.0)
    return out


def shortest_vector(basis):
    """Кратчайший ненулевой вектор решётки (точный, через lattice_points_within).

    :param basis: базис решётки (желательно LLL-приведённый).
    :return: numpy-вектор.
    """
    basis = np.asarray(basis, dtype=float)
    start = min(float(np.linalg.norm(row)) for row in basis)
    candidates = lattice_points_within(basis, start + 1e-9)
    return min(candidates, key=lambda v: float(v @ v))
