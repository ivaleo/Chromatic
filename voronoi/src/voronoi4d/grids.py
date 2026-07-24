"""Генерация базисов решёток для перебора.

- generate_grids() — случайные вещественные решётки;
- generate_integer_grids() — полный перебор целочисленных решёток.

Все базисы LLL-приводятся и нормализуются к каноническому виду,
чтобы исключить дубликаты.
"""

import random

import numpy as np
from itertools import product, combinations

from .lll import lll_reduce

DEFAULT_PRECISION = 3  # точность округления элементов матрицы

# --------------------------------------------------------------------------------


def normalize_rows(matrix):
    """Делает первый ненулевой элемент каждой строки неотрицательным.

    :param matrix: исходная матрица (numpy.ndarray).
    :return: новая матрица.
    """
    matrix = matrix.copy()
    for i in range(matrix.shape[0]):
        for val in matrix[i]:
            if val != 0:
                if val < 0:
                    matrix[i] *= -1
                break  # выходим после первого ненулевого
    return matrix


def canonical_form_by_rows(matrix):
    """Приводит матрицу к каноническому виду через лексикографическую сортировку строк.

    :param matrix: исходная матрица (numpy.ndarray).
    :return: новая матрица с отсортированными строками.
    """
    idx = np.lexsort(matrix.T[::-1])  # инвертируем столбцы для нужного порядка
    return matrix[idx]


# --------------------------------------------------------------------------------
# случайные вещественные решётки


def generate_random_matrix(a=2, precision=DEFAULT_PRECISION):
    """Генерирует случайную невырожденную матрицу 4x4.

    Первая строка фиксирована: (1, 0, 0, 0). В остальных строках первый
    элемент из [0, 1], остальные из [-a, a].

    :param a: границы генерации элементов.
    :param precision: точность округления.
    :return: невырожденная матрица (numpy.ndarray).
    """
    first_row = [1.0, 0.0, 0.0, 0.0]

    while True:
        matrix = [first_row]
        for _ in range(3):
            first = round(random.uniform(0, 1), precision)
            rest = [round(random.uniform(-a, a), precision) for _ in range(3)]
            matrix.append([first] + rest)

        mat = np.array(matrix, dtype=float)

        # проверяем невырожденность
        if not np.isclose(np.linalg.det(mat), 0, atol=1e-8):
            return mat


def check_grid(grid, norm_factor=2.0, cos_limit=0.5):
    """Проверяет качество базиса решётки.

    Условия: длина каждого вектора >= 1, отношение длин векторов
    не более norm_factor, |косинус| угла между векторами не более cos_limit.

    :param grid: базис решётки (numpy.ndarray).
    :return: True, если базис проходит все проверки.
    """
    for i in range(len(grid) - 1):
        vi = grid[i]
        ni = np.linalg.norm(vi)
        for j in range(i + 1, len(grid)):
            vj = grid[j]
            nj = np.linalg.norm(vj)

            if ni < 1 or nj < 1:
                return False
            if ni / nj > norm_factor or nj / ni > norm_factor:
                return False

            cos = vi @ vj / (ni * nj)
            if abs(cos) > cos_limit:
                return False

    return True


def generate_grids(norm_limit=2.0, norm_factor=2.0, cos_limit=0.5, det_limit=1.0,
                   max_attempts=5000, precision=DEFAULT_PRECISION):
    """Генерирует список случайных решёток, прошедших проверку качества.

    :param norm_limit: границы генерации элементов матрицы.
    :param norm_factor: максимальное отношение длин векторов базиса.
    :param cos_limit: максимальный |косинус| угла между векторами.
    :param det_limit: минимальный определитель.
    :param max_attempts: количество попыток генерации.
    :param precision: точность округления.
    :return: список базисов (numpy.ndarray) в каноническом виде.
    """
    list_grids = []

    for _ in range(max_attempts):
        matrix = generate_random_matrix(norm_limit, precision=precision)

        # LLL-приведение и нормализация к каноническому виду
        lll_grid = lll_reduce(matrix, precision=precision)
        lll_grid = normalize_rows(lll_grid)
        lll_grid = canonical_form_by_rows(lll_grid)

        det = np.linalg.det(lll_grid)
        if det > det_limit and check_grid(lll_grid, norm_factor, cos_limit):
            list_grids.append(lll_grid)

    return list_grids


# --------------------------------------------------------------------------------
# полный перебор целочисленных решёток


def generate_integer_grids(coeff_range=(-1, 2), show_progress=True):
    """Перебирает целочисленные решётки 4x4 с фиксированной первой строкой (1, 0, 0, 0).

    Остальные три строки выбираются из всех канонических целочисленных векторов
    (первый ненулевой элемент положителен) с координатами из coeff_range.

    :param coeff_range: кортеж (min, max) — диапазон координат range(min, max).
    :param show_progress: показывать прогресс-бар (требуется tqdm).
    :return: список уникальных невырожденных базисов в каноническом виде.
    """
    # все векторы-кандидаты
    list_coords = [var for var in product(range(*coeff_range), repeat=4)]
    list_coords.remove((1, 0, 0, 0))  # первая строка фиксирована

    # оставляем только канонические векторы (первый ненулевой элемент положителен)
    candidates = []
    for vec in list_coords:
        first_non_zero = next((x for x in vec if x != 0), None)
        if first_non_zero is not None and first_non_zero > 0:
            candidates.append(vec)

    all_combinations = combinations(candidates, 3)

    if show_progress:
        try:
            from math import comb
            from tqdm import tqdm

            total = comb(len(candidates), 3)
            all_combinations = tqdm(all_combinations, total=total, desc="Обработка комбинаций")
        except ImportError:
            pass  # tqdm не установлен — работаем без прогресс-бара

    list_grids = []

    for vectors in all_combinations:
        g = np.vstack(([1, 0, 0, 0], *vectors)).astype(float)

        if np.isclose(np.linalg.det(g), 0):  # вырожденная матрица
            continue

        # LLL-приведение и нормализация к каноническому виду
        lll_grid = lll_reduce(g, precision=0)
        lll_grid = normalize_rows(lll_grid)
        lll_grid = canonical_form_by_rows(lll_grid)

        # проверяем уникальность
        if not any(np.allclose(lll_grid, m) for m in list_grids):
            list_grids.append(lll_grid)

    return list_grids


# --------------------------------------------------------------------------------


def min_max_det(list_grids, verbose=True):
    """Вычисляет минимальный и максимальный |определитель| по списку решёток.

    :param list_grids: список базисов.
    :param verbose: печатать решётки при обновлении минимума/максимума.
    :return: кортеж (min_det, max_det).
    """
    min_det = float("inf")
    max_det = 0.0

    for grid in list_grids:
        det = np.abs(np.linalg.det(grid))
        if det < min_det:
            min_det = det
            if verbose:
                print(grid, "\nminDet =", min_det)
        if det > max_det:
            max_det = det
            if verbose:
                print(grid, "\nmaxDet =", max_det)

    return min_det, max_det
