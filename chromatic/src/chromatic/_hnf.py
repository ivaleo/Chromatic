"""Целочисленная эрмитова нормальная форма (HNF) для матриц перехода.

Конвенция всего воркспейса: верхнетреугольная HNF, диагональ > 0, элементы
выше диагонали приведены в [0, d_j) по модулю диагонали своего столбца.
Используется фасадом, чтобы выдавать `transition` в координатах базиса
ПОЛЬЗОВАТЕЛЯ (а не внутреннего LLL-приведённого кадра бэкенда).
"""

from __future__ import annotations

from typing import List, Sequence


def hnf_of(matrix: Sequence[Sequence[int]]) -> List[List[int]]:
    """HNF заданной целочисленной матрицы (унимодулярные операции над строками)."""
    c = [[int(x) for x in row] for row in matrix]
    n = len(c)
    if any(len(row) != n for row in c):
        raise ValueError("hnf_of: матрица не квадратная")

    for col in range(n):
        for row in range(col + 1, n):
            while c[row][col] != 0:
                if c[col][col] == 0:
                    c[col], c[row] = c[row], c[col]
                    continue
                # евклидов шаг: деление с округлением к нулю уменьшает |c[row][col]|
                q = int(c[row][col] / c[col][col])
                for j in range(n):
                    c[row][j] -= q * c[col][j]
                if c[row][col] != 0:
                    c[col], c[row] = c[row], c[col]
        if c[col][col] == 0:
            raise ValueError("hnf_of: вырожденная матрица")
        if c[col][col] < 0:
            c[col] = [-x for x in c[col]]
        d = c[col][col]
        for row in range(col):
            q = c[row][col] // d  # floor-деление: остаток попадает в [0, d)
            if q:
                for j in range(n):
                    c[row][j] -= q * c[col][j]
    return c


def transition_in_user_basis(sub_basis: Sequence[Sequence[float]],
                             basis: Sequence[Sequence[float]],
                             tol: float = 1e-6) -> List[List[int]]:
    """Матрица перехода подрешётки в координатах базиса пользователя.

    Строки sub_basis раскладываются по строкам basis; коэффициенты обязаны быть
    целыми (иначе ValueError), результат приводится к HNF.
    """
    import numpy as np

    b = np.asarray(basis, dtype=float)
    s = np.asarray(sub_basis, dtype=float)
    coeffs = s @ np.linalg.inv(b)
    rounded = np.rint(coeffs)
    if not np.allclose(coeffs, rounded, rtol=0.0, atol=tol):
        raise ValueError(
            "transition_in_user_basis: sub_basis не является подрешёткой basis "
            "(нецелые коэффициенты разложения)")
    return hnf_of(rounded.astype(int).tolist())
