"""Единый запуск проверки пяти заголовочных утверждений.

    python -m chromatic_research.campaigns.verify_main_results          # уровень 1
    python -m chromatic_research.campaigns.verify_main_results --full   # + уровни 2/3

Пять строк вывода отвечают пяти доказанным оценкам

    chi(R^4) <= 43,  chi(R^5) <= 132,  chi(R^7) <= 1029,
    chi(R^9) <= 9604,  chi(R^10) <= 45619.

Уровень независимости (шкала П13 ревью
``journal/REVIEW-split-proposals-2026-09-03.md``):

* без флагов — **уровень 1**: итоговые неравенства пересчитываются из
  опубликованных дробей сертификатов ``audit-data/results/*.json`` на
  ``fractions.Fraction`` (без numpy и без кода конвейера); для 9604 и 45619
  пересчитываются сами аналитические суммы 6/7+1/9 и 6/7+4/31 и ширины
  блоков. Это проверяет арифметику *утверждения*, а не полноту списков
  вершин и опасных векторов — та проверяется полными верификаторами;
* ``--full`` — **уровни 2/3**: дополнительно запускаются полные точные
  верификаторы, заново строящие ячейку Вороного и все KKT-сертификаты:
  ``dim4_k43_verify`` (независимый пересчёт 43 без voronoi4d/combigeo),
  ``dim7_1029_exact`` (1029, полнота вершин по 1-скелету) и
  ``verify_exact_voronoi`` (аудит 132 без Qhull). Их вывод пишется во
  временный каталог; опубликованные артефакты не перезаписываются.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path

from chromatic_research.paths import results_path

CLAIMS = ("R4_43", "R5_132", "R7_1029", "R9_9604", "R10_45619")


@dataclass
class Check:
    claim: str
    ok: bool
    detail: str


def _load(name: str) -> dict:
    return json.loads(results_path(name).read_text())


# ---------------------------------------------------------------- R^4: 43


def check_r4_43() -> Check:
    cert = _load("cert43_eisenstein.json")          # опорный сертификат
    ver = _load("dim4_k43_verify.json")             # независимый пересчёт
    l0 = F(cert["l0"])
    ddown2, diamup2 = F(cert["Ddown2"]), F(cert["diamup2"])
    d2, big_d2, diam2 = F(ver["d2"]), F(ver["D2"]), F(ver["diam2"])
    conds = {
        "ok-флаг сертификата": bool(cert["ok"]),
        "l0 = 100411/100000": l0 == F(100411, 100000),
        "Ddown2 > l0^2*diamup2": ddown2 > l0 * l0 * diamup2,
        "индекс 43": ver["index"] == 43,
        "d2 = D2/diam2": d2 == big_d2 / diam2,
        "d2 > l0^2": d2 > l0 * l0,
        "опорная оценка не выше точной": ddown2 <= big_d2,
        "диаметры совпали": diamup2 == diam2,
    }
    bad = [k for k, v in conds.items() if not v]
    detail = (f"d^2 = {d2} = {float(d2):.9f}, запас d^2 - l0^2 = "
              f"{float(d2 - l0 * l0):.3e} > 0")
    return Check("R4_43", not bad, detail if not bad else "нарушено: " + ", ".join(bad))


# ---------------------------------------------------------------- R^5: 132


def check_r5_132() -> Check:
    cert = _load("metric_deform_a5_132_refined_certificate.json")
    sep, ci = cert["separation"], cert["certified_interval"]
    ell = F(ci["upper_endpoint"])
    m2, diam2 = F(sep["minimum_distance_squared"]), F(sep["diameter_squared"])
    margin = m2 - ell * ell * diam2
    conds = {
        "оценка 132": int(cert["certified_upper_bound"]) == 132,
        "интервал valid": bool(ci["valid"]) is True,
        "ell = 101/100": ell == F(101, 100),
        "запас совпал с записанным": margin == F(ci["squared_margin"]),
        "запас > 0": margin > 0,
        "diam^2 = 4 R^2": diam2 == 4 * F(cert["voronoi"]["covering_radius_squared"]),
    }
    bad = [k for k, v in conds.items() if not v]
    detail = (f"D_min^2 - (101/100)^2 diam^2 = {float(margin):.6f} > 0, "
              f"d = {float(m2 / diam2) ** 0.5:.9f}")
    return Check("R5_132", not bad, detail if not bad else "нарушено: " + ", ".join(bad))


# ---------------------------------------------------------------- R^7: 1029


def check_r7_1029() -> Check:
    d = _load("dim7_1029_exact.json")
    ell = F(d["ell"])
    dmin2, diam2, r2 = (F(d["minimum_distance_squared_exact"]),
                        F(d["diameter_squared_exact"]), F(d["radius_squared_exact"]))
    d2, margin = F(d["normalized_distance_squared_exact"]), F(d["margin_exact"])
    conds = {
        "индекс 1029": d["index"] == 1029,
        "interval_valid": d["interval_valid"] is True,
        "D_min^2 = 7": dmin2 == 7,
        "d^2 = 7/diam^2": d2 == dmin2 / diam2,
        "d^2 > 1": d2 > 1,
        "ell = 103/100": ell == F(103, 100),
        "запас = 7 - ell^2 diam^2": margin == dmin2 - ell * ell * diam2,
        "запас > 0": margin > 0,
        "diam^2 = 4 R^2": diam2 == 4 * r2,
    }
    bad = [k for k, v in conds.items() if not v]
    detail = f"d^2 = {d2}, d = {float(d2) ** 0.5:.9f}, запас {float(margin):.3e} > 0"
    return Check("R7_1029", not bad, detail if not bad else "нарушено: " + ", ".join(bad))


# ------------------------------------------------- аналитические блоки


def e8_block_width_squared() -> F:
    """Блок (3+omega)E_8 в нормировке lambda_1^2 = 3: diam^2 = 6, D^2 = 7."""
    diam2 = F(6)          # 2R = 2 при lambda_1^2 = 2; масштаб 3/2 даёт 6
    dmin2 = F(7)          # планарная граница: D^2 >= (7/3) lambda_1^2 = 7
    return dmin2 / diam2


def planar_block_width_squared() -> F:
    """Блок (5+2omega)A_2, индекс 19, в нормировке lambda_1 = 1.

    Точка (5+2omega)/2 = (2, sqrt3/2); ближайшая вершина шестиугольника ---
    (1/2, 1/(2 sqrt3)).  Квадрат расстояния: (3/2)^2 + (sqrt3/2 - 1/(2 sqrt3))^2,
    где вторая скобка равна (3-1)/(2 sqrt3) = 1/sqrt3, то есть её квадрат 1/3.
    D^2 = 4*dist^2, diam^2 = (2/sqrt3)^2 = 4/3.
    """
    dist2 = F(3, 2) ** 2 + F(1, 3)          # 9/4 + 1/3 = 31/12
    dmin2 = 4 * dist2                        # 31/3
    diam2 = F(4, 3)
    return dmin2 / diam2                     # 31/4


def check_r9_9604() -> Check:
    cost = 1 / e8_block_width_squared() + 1 / F(3) ** 2   # 6/7 + 1/9
    conds = {
        "6/7 + 1/9 = 61/63": cost == F(61, 63),
        "сумма < 1": cost < 1,
        "2401*4 = 9604": 2401 * 4 == 9604,
        "d_1^2 = 7/6": e8_block_width_squared() == F(7, 6),
    }
    bad = [k for k, v in conds.items() if not v]
    detail = f"6/7 + 1/9 = {cost} < 1, ell = sqrt(63/61) = {float(1 / cost) ** 0.5:.6f}"
    return Check("R9_9604", not bad, detail if not bad else "нарушено: " + ", ".join(bad))


def check_r10_45619() -> Check:
    cost = 1 / e8_block_width_squared() + 1 / planar_block_width_squared()  # 6/7 + 4/31
    conds = {
        "d_2^2 = 31/4": planar_block_width_squared() == F(31, 4),
        "6/7 + 4/31 = 214/217": cost == F(214, 217),
        "сумма < 1": cost < 1,
        "2401*19 = 45619": 2401 * 19 == 45619,
    }
    bad = [k for k, v in conds.items() if not v]
    detail = f"6/7 + 4/31 = {cost} < 1, ell = sqrt(217/214) = {float(1 / cost) ** 0.5:.6f}"
    return Check("R10_45619", not bad, detail if not bad else "нарушено: " + ", ".join(bad))


CHECKS = (check_r4_43, check_r5_132, check_r7_1029, check_r9_9604, check_r10_45619)


def run_all() -> list[Check]:
    return [c() for c in CHECKS]


# ------------------------------------------------------------ --full


def full_commands(tmp: Path) -> list[list[str]]:
    """Полные верификаторы; выход --- во временный каталог, артефакты не трогаются."""
    py = sys.executable
    return [
        [py, "-m", "chromatic_research.campaigns.dim4_k43_verify",
         str(results_path("r4_k43_eisenstein_rational.json")), str(tmp / "dim4_k43_verify.json")],
        [py, "-m", "chromatic_research.campaigns.dim7_1029_exact",
         "--output", str(tmp / "dim7_1029_exact.json")],
    ]


# Аудит 132 без Qhull (verify_exact_voronoi) перечисляет короткие векторы в том же
# доказуемо полном окне |v| < 2(1+ell)R, что и сертификат (ell берётся из поля
# certified_interval.upper_endpoint; при ell = 1 это классическое окно 4R), и
# сверяет их число и KKT-свидетели с сертификатом. Его протокол —
# results/metric_deform_a5_132_refined_independent_exact_audit.json
# (certificate_count 38).


def audit_132_command(tmp: Path) -> list[str]:
    return [sys.executable, "-m", "chromatic_research.campaigns.verify_exact_voronoi",
            str(results_path("metric_deform_a5_132_refined_certificate.json")),
            "--output", str(tmp / "a5_132_independent_audit.json")]


def run_full() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for cmd in full_commands(tmp) + [audit_132_command(tmp)]:
            print(">>", " ".join(cmd[1:]))
            code = subprocess.call(cmd)
            print("   код возврата", code)
            ok &= code == 0
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="дополнительно запустить полные точные верификаторы (минуты)")
    args = parser.parse_args(argv)

    results = run_all()
    for r in results:
        print(f"{'OK  ' if r.ok else 'FAIL'} {r.claim:<10} {r.detail}")
    all_ok = all(r.ok for r in results)
    if args.full:
        all_ok &= run_full()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
