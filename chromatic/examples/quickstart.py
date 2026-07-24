"""Быстрый старт фасада chromatic.

Запуск:  python examples/quickstart.py
Требует хотя бы один установленный бэкенд (combigeo или voronoi4d).
"""

import chromatic

D4 = [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]


def main() -> None:
    print("Доступные бэкенды:", chromatic.available_backends())

    # явный выбор бэкенда по имени
    backend_name = "combigeo" if "combigeo" in chromatic.available_backends() else "voronoi4d"
    backend = chromatic.get_backend(backend_name)
    print(f"Используем бэкенд: {backend_name}\n")

    # ячейка Вороного решётки D4 (24-ячейка)
    cell = backend.build_cell(D4)
    print(cell)
    if cell.f_vector:
        print("f-вектор:", cell.f_vector)
    print("диаметр:", cell.diameter, "\n")

    # поиск оптимальной подрешётки для 49 цветов
    res = backend.find_optimal(D4, 49)
    print(res)
    print("матрица перехода (HNF):")
    if res.transition:
        for row in res.transition:
            print("  ", row)
    print(f"\nЗапрещённое расстояние d = {res.normalized:.6f} "
          f"({'пригодна' if res.feasible else 'непригодна'})")

    # кросс-валидация (если доступны оба бэкенда)
    if {"voronoi4d", "combigeo"} <= set(chromatic.available_backends()):
        print("\nКросс-валидация бэкендов на D4, индекс 49:")
        report = chromatic.compare_backends(D4, [49])
        print(" ", report, "— совпадают" if report.agree else "— РАСХОЖДЕНИЯ")


if __name__ == "__main__":
    main()
