"""LLL-приведение базиса решётки.

Если установлена библиотека fpylll (быстрая C-реализация), используется она,
иначе — собственная реализация на чистом python. Единая точка входа — lll_reduce().
"""

import numpy as np

try:
    from fpylll import IntegerMatrix, LLL as _FpylllLLL

    HAS_FPYLLL = True
except ImportError:
    HAS_FPYLLL = False


# --------------------------------------------------------------------------------


def gram_schmidt(basis):
    """Ортогонализация Грама-Шмидта.

    :param basis: список векторов базиса.
    :return: кортеж (b_star, mu), где b_star — ортогонализованные векторы,
             mu — строго нижнетреугольная матрица коэффициентов (диагональ
             нулевая и не используется; квадраты норм |b*_i|^2 lll_reduce_python
             считает отдельно в beta).
    """
    n = len(basis)
    b_star = [np.array(basis[i], dtype=float) for i in range(n)]
    mu = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            mu[i, j] = np.dot(basis[i], b_star[j]) / np.dot(b_star[j], b_star[j])
            b_star[i] -= mu[i, j] * b_star[j]

    return b_star, mu


def lll_reduce_python(basis, delta=0.75):
    """LLL-приведение базиса (чистый python).

    :param basis: матрица базиса (numpy.ndarray или список векторов).
    :param delta: параметр Ловаса (обычно 0.75).
    :return: приведённый базис (numpy.ndarray).
    """
    n = len(basis)
    basis = [np.array(vec, dtype=float) for vec in basis]
    b_star, mu = gram_schmidt(basis)
    beta = [np.dot(b_star[i], b_star[i]) for i in range(n)]

    k = 1
    while k < n:
        # Size Reduction
        for l in range(k - 1, -1, -1):
            if abs(mu[k, l]) > 0.5:
                q = round(mu[k, l])
                basis[k] -= q * basis[l]
                # пересчитываем Грама-Шмидта после изменения базиса
                b_star, mu = gram_schmidt(basis)
                beta = [np.dot(b_star[i], b_star[i]) for i in range(n)]

        # проверка условия Ловаса
        if beta[k] < (delta - mu[k, k - 1] ** 2) * beta[k - 1]:
            # Interchange: меняем векторы местами и пересчитываем всё
            basis[k], basis[k - 1] = basis[k - 1].copy(), basis[k].copy()
            b_star, mu = gram_schmidt(basis)
            beta = [np.dot(b_star[i], b_star[i]) for i in range(n)]
            k = max(1, k - 1)
        else:
            k += 1

    return np.array(basis)


def lll_reduce_fpylll(basis, delta=0.75, precision=12):
    """LLL-приведение через fpylll (работает с целыми числами).

    Матрица масштабируется до целых чисел с заданной точностью,
    приводится и масштабируется обратно. Если базис непредставим в этой
    точности (округление меняет решётку), возвращается None — вызывающая
    сторона обязана перейти на python-реализацию.

    :param basis: матрица базиса (numpy.ndarray).
    :param delta: параметр Ловаса (передаётся в fpylll).
    :param precision: количество десятичных знаков, сохраняемых при масштабировании.
    :return: приведённый базис (numpy.ndarray, float) или None.
    """
    arr = np.asarray(basis, dtype=float)
    scale = 10 ** precision
    scaled = np.round(arr * scale)
    # представимость: округление не должно возмутить решётку
    # (масштаб больших чисел тоже контролируем: целые части точны в double до 2^53)
    tol = 1e-9 * max(1.0, float(np.abs(arr).max()))
    if not np.allclose(scaled / scale, arr, rtol=0.0, atol=tol) or np.abs(scaled).max() > 2 ** 53:
        return None

    reduced = _FpylllLLL.reduction(
        IntegerMatrix.from_matrix([[int(v) for v in row] for row in scaled.tolist()]),
        delta=delta)
    rows, cols = reduced.nrows, reduced.ncols
    return np.array([[reduced[i, j] for j in range(cols)] for i in range(rows)],
                    dtype=float) / scale


def lll_reduce(basis, delta=0.75, precision=12):
    """LLL-приведение базиса: fpylll, если доступна и базис представим,
    иначе чистый python. delta действует в обоих путях.

    (До версии 1.1.0 fpylll-путь молча округлял базис до 3 знаков и игнорировал
    delta; см. AUDIT-2026-07-21.)

    :param basis: матрица базиса (numpy.ndarray).
    :param delta: параметр Ловаса (обычно 0.75).
    :param precision: точность целочисленного масштабирования (только для fpylll).
    :return: приведённый базис (numpy.ndarray, float).
    """
    if HAS_FPYLLL:
        reduced = lll_reduce_fpylll(basis, delta=delta, precision=precision)
        if reduced is not None:
            return reduced
    return lll_reduce_python(basis, delta=delta)
