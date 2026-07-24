"""Сохранение результатов поиска и построение графиков."""

import datetime

# --------------------------------------------------------------------------------


def save_result(grid, det, mat, center, dist, output_file="results.txt"):
    """Дописывает результат поиска в файл.

    :param grid: исходная решётка (numpy.ndarray).
    :param det: определитель матрицы перехода.
    :param mat: матрица перехода (numpy.ndarray).
    :param center: координаты точки (numpy.ndarray).
    :param dist: расстояние до многогранника.
    :param output_file: путь к файлу результатов.
    """
    with open(output_file, "a") as f:
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Grid: {grid.tolist()}\n")
        f.write(f"Determinant: {det}\n")
        f.write(f"Matrix:\n{mat.tolist()}\n")
        f.write(f"Center: {center.tolist()}\n")
        f.write(f"Distance: {dist:.6f}\n")
        f.write("-" * 50 + "\n")


def plot_results(d_values, n_values, title="Зависимость N от d"):
    """Строит график зависимости хроматического числа N от запрещённого расстояния d.

    N = |det(M)| — количество цветов раскраски, d — максимальное нормированное
    запрещённое расстояние, достижимое этим количеством цветов.
    Требует установленный matplotlib (pip install voronoi4d[plot]).

    :param d_values: список нормированных расстояний d.
    :param n_values: список количеств цветов N.
    :param title: заголовок графика.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(d_values, n_values, "bo-", linewidth=2, markersize=8, markerfacecolor="red")

    plt.title(title, fontsize=14)
    plt.xlabel("d", fontsize=12)
    plt.ylabel("N", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()
