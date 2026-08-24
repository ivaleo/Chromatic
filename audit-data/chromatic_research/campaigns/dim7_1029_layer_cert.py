"""Независимая точная перепроверка χ(ℝ⁷) ≤ 1029 двухслойным сертификатом.

Оценка 1029 уже имеет статус [С]: она доказана полным точным перечислением
всех 30 368 вершин семимерной ячейки Вороного
(:mod:`chromatic_research.campaigns.dim7_1029_exact`).  Здесь та же теорема
получается СОВЕРШЕННО ДРУГИМ путём — через слоёную структуру конструкции
(ламинирование ``E6*/343``) и точный кусочный сертификат радиуса покрытия
(:mod:`chromatic_research.core.exact_layer_cert`).

Что именно доказывается.  Связывающее расстояние ``D_min^2 = 7`` даётся
планарной теоремой об эйзенштейновых конструкциях, поэтому для ``ell = 1``
достаточно ``diam^2 < 7``, то есть ``R^2 < 7/4``.  Сертификат ограничивает
``R^2`` сверху, не перечисляя вершин семимерной ячейки вообще: он работает в
ШЕСТИмерной базе и опирается только на выпуклость мажоранты ``phi`` на кусках
общего измельчения двух сдвинутых разбиений Вороного.

Ценность перепроверки в том, что два пути не делят ни одной вычислительной
детали: первый перечисляет вершины ``V_0`` в ℝ⁷, второй — вершины плоских
кусков в ℝ⁶ и вообще не строит семимерную ячейку.

Usage::

    python -m chromatic_research.campaigns.dim7_1029_layer_cert
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as Fr
from pathlib import Path

from chromatic_research.core.exact_layer_cert import (
    certify_covering_radius,
    geometry_from_gram,
)
from chromatic_research.paths import results_path

# Та же рациональная форма, что и в dim7_1029_exact (Q = INTEGER_GRAM / 10000).
INTEGER_GRAM = [
    [30000, 15000, 0, -22500, 15000, 7500, -8632],
    [15000, 30000, 22500, 0, 7500, 15000, -4197],
    [0, 22500, 45000, 22500, 0, 22500, 1601],
    [-22500, 0, 22500, 45000, -22500, 0, 6617],
    [15000, 7500, 0, -22500, 30000, 15000, -5275],
    [7500, 15000, 22500, 0, 15000, 30000, 4434],
    [-8632, -4197, 1601, 6617, -5275, 4434, 17897],
]
DENOMINATOR = 10_000

# База — E6* в нормировке lambda_1^2 = 3; для E6* отношение 2R/lambda_1 = sqrt(2),
# то есть R^2 = lambda_1^2 / 2 = 3/2 (см. таблицу точных ширин в RESULTS.md).
BASE_R2 = Fr(3, 2)

# Порог: D_min^2 = 7 (планарная теорема), нужно diam^2 = 4 R^2 < 7.
TARGET_R2 = Fr(7, 4)

# Значение из независимого сертификата dim7_1029_exact, для сверки.
KNOWN_DIAM2 = Fr(96858928789031597, 14761819462500000)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--exact-max", action="store_true",
                        help="считать точный максимум мажоранты, а не только "
                             "проверять порог (дольше)")
    args = parser.parse_args(argv)

    geometry = geometry_from_gram(INTEGER_GRAM, DENOMINATOR, BASE_R2)
    print(f"база: размерность {geometry.dim}, релевантных векторов "
          f"{len(geometry.relevant)}")
    print(f"сдвиг слоя c = {[str(v) for v in geometry.offset]}")
    print(f"высота^2 t^2 = {geometry.height2} = {float(geometry.height2):.9f}")
    print(f"кусков к проверке: {len(geometry.piece_shifts())}")
    print(f"цель: R^2 < {TARGET_R2} (то есть diam^2 < 7)")

    report = certify_covering_radius(
        geometry, None if args.exact_max else TARGET_R2)

    known_r2 = KNOWN_DIAM2 / 4
    max_phi = report["max_phi"]
    report_json = {
        "target_r2": str(TARGET_R2),
        "certified": report["certified"],
        "max_phi": str(max_phi),
        "max_phi_float": float(max_phi),
        "diam2_upper": str(4 * max_phi),
        "known_r2_from_vertex_enumeration": str(known_r2),
        "known_r2_float": float(known_r2),
        "pieces": report["pieces"],
        "empty_pieces": report["empty_pieces"],
        "max_constraints": report["max_constraints"],
        "max_vertices": report["max_vertices"],
        "seconds": report["seconds"],
        "exact_max_mode": bool(args.exact_max),
    }
    out = args.output or results_path("dim7_1029_layer_cert.json")
    out.write_text(json.dumps(report_json, indent=1, ensure_ascii=False) + "\n")

    print()
    print(f"max phi   = {float(max_phi):.12f}  (точно {max_phi})")
    print(f"diam^2 <= {float(4 * max_phi):.12f}")
    print(f"известный точный R^2 = {float(known_r2):.12f} "
          f"(перечисление вершин); мажоранта обязана быть не ниже: "
          f"{'ДА' if max_phi >= known_r2 else 'НЕТ — ПРОТИВОРЕЧИЕ'}")
    print(f"кусков {report['pieces']}, пустых {report['empty_pieces']}, "
          f"{report['seconds']} с")
    if report["certified"]:
        print("СЕРТИФИЦИРОВАНО: R^2 < 7/4, значит diam^2 < 7 = D_min^2,")
        print("то есть chi(R^7, [1, 1]) <= 1029 — независимо от перечисления вершин.")
    else:
        print(f"НЕ сертифицировано; провалившихся кусков: "
              f"{len(report['failed_pieces'])}")
    print(f"сохранено: {out}")
    return 0 if report["certified"] or args.exact_max else 2


if __name__ == "__main__":
    raise SystemExit(main())
