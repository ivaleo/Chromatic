"""Полная кросс-валидация бэкендов на пригодной раскраске.

Прогоняет ОБА бэкенда (combigeo и voronoi4d) на D4 для числа цветов 49 —
известной пригодной раскраски (d = 1.080123) — и сверяет результаты.

С версии 1.1.0 voronoi4d на индексе 49 считает ~7 секунд (точный префильтр по кратчайшему вектору); без fpylll — до пары минут.
Для скорости установите fpylll: pip install fpylll cysignals.

Запуск:  python examples/cross_validate.py
Требует оба бэкенда.
"""

import time

import chromatic

D4 = [[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]]


def main() -> None:
    avail = chromatic.available_backends()
    if not {"voronoi4d", "combigeo"} <= set(avail):
        raise SystemExit(f"нужны оба бэкенда, доступны: {avail}")

    index = 49
    print(f"Кросс-валидация D4, {index} цветов (известная пригодная раскраска)\n")

    t = time.time()
    report = chromatic.compare_backends(D4, [index])
    dt = time.time() - t

    rv = report.results_voronoi4d[index]
    rc = report.results_combigeo[index]

    print(f"voronoi4d: d = {rv.normalized:.6f}  D = {rv.min_distance:.6f}  "
          f"feasible = {rv.feasible}")
    print(f"combigeo:  d = {rc.normalized:.6f}  D = {rc.min_distance:.6f}  "
          f"feasible = {rc.feasible}  (examined {rc.examined})")
    print()

    if report.agree:
        print(f"✓ бэкенды СОГЛАСНЫ (за {dt:.1f}s)")
    else:
        print("✗ РАСХОЖДЕНИЯ:")
        for d in report.discrepancies:
            print(f"  k={d.num_colors} {d.field}: voronoi4d={d.voronoi4d} "
                  f"combigeo={d.combigeo} diff={d.diff}")


if __name__ == "__main__":
    main()
