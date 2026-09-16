"""Единый запуск проверки пяти заголовочных утверждений.

    python -m chromatic_research.campaigns.verify_main_results          # уровень 1
    python -m chromatic_research.campaigns.verify_main_results --full   # + уровни 2/3

Пять строк вывода отвечают пяти доказанным оценкам

    chi(R^4) <= 43,  chi(R^5) <= 132,  chi(R^7) <= 1029,
    chi(R^9) <= 7203,  chi(R^10) <= 45619.

Уровень независимости (шкала П13 ревью
``journal/REVIEW-2026-09-03-split-proposals.md``):

* без флагов — **уровень 1**: итоговые неравенства пересчитываются из
  опубликованных дробей сертификатов ``audit-data/results/*.json`` на
  ``fractions.Fraction`` (без numpy и без кода конвейера); для 45619
  пересчитываются сами аналитические суммы 6/7+1/9 и 6/7+4/31 и ширины
  блоков. Это проверяет арифметику *утверждения*, а не полноту списков
  вершин и опасных векторов — та проверяется полными верификаторами;
* ``--full`` — **уровни 2/3**: дополнительно запускаются полные точные
  верификаторы, заново строящие ячейку Вороного и все KKT-сертификаты:
  ``dim4_k43_verify`` (независимый пересчёт 43 без voronoi4d/combigeo),
  ``dim7_1029_exact`` (1029, полнота вершин по 1-скелету),
  ``dim9_7203_exact`` (7203, то же в девяти измерениях: ~1,5 ч) и
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


@dataclass
class Check:
    claim: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ExactCert:
    """Поля артефакта точного верификатора; дроби уже разобраны в Fraction."""

    index: int
    # не приводится к bool: проверки требуют именно `is True`, а не истинность
    interval_valid: object
    ell: F
    dmin2: F
    diam2: F
    r2: F
    d2: F
    margin: F


def _load(name: str) -> dict:
    return json.loads(results_path(name).read_text())


def _load_exact_cert(name: str) -> ExactCert:
    """Разбирает артефакт dim7/dim9: у них одинаковый набор ключей."""
    raw = _load(name)
    return ExactCert(
        index=raw["index"],
        interval_valid=raw["interval_valid"],
        ell=F(raw["ell"]),
        dmin2=F(raw["minimum_distance_squared_exact"]),
        diam2=F(raw["diameter_squared_exact"]),
        r2=F(raw["radius_squared_exact"]),
        d2=F(raw["normalized_distance_squared_exact"]),
        margin=F(raw["margin_exact"]),
    )


def _verdict(claim: str, conds: dict[str, bool], detail: str) -> Check:
    """Свёртка словаря «условие -> выполнено» в Check; при провале — их список."""
    bad = [name for name, held in conds.items() if not held]
    return Check(claim, not bad, detail if not bad else "нарушено: " + ", ".join(bad))


# ---------------------------------------------------------------- R^4: 43


def check_r4_43() -> Check:
    cert = _load("cert43_eisenstein.json")          # опорный сертификат
    ver = _load("dim4_k43_verify.json")             # независимый пересчёт
    l0 = F(cert["l0"])
    ddown2, diamup2 = F(cert["Ddown2"]), F(cert["diamup2"])
    d2, dmin2, diam2 = F(ver["d2"]), F(ver["D2"]), F(ver["diam2"])
    conds = {
        "ok-флаг сертификата": bool(cert["ok"]),
        "l0 = 100411/100000": l0 == F(100411, 100000),
        "Ddown2 > l0^2*diamup2": ddown2 > l0 * l0 * diamup2,
        "индекс 43": ver["index"] == 43,
        "d2 = D2/diam2": d2 == dmin2 / diam2,
        "d2 > l0^2": d2 > l0 * l0,
        "опорная оценка не выше точной": ddown2 <= dmin2,
        "диаметры совпали": diamup2 == diam2,
    }
    return _verdict("R4_43", conds,
                    f"d^2 = {d2} = {float(d2):.9f}, запас d^2 - l0^2 = "
                    f"{float(d2 - l0 * l0):.3e} > 0")


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
    return _verdict("R5_132", conds,
                    f"D_min^2 - (101/100)^2 diam^2 = {float(margin):.6f} > 0, "
                    f"d = {float(m2 / diam2) ** 0.5:.9f}")


# ---------------------------------------------------------------- R^7: 1029


def check_r7_1029() -> Check:
    c = _load_exact_cert("dim7_1029_exact.json")
    conds = {
        "индекс 1029": c.index == 1029,
        "interval_valid": c.interval_valid is True,
        "D_min^2 = 7": c.dmin2 == 7,
        "d^2 = 7/diam^2": c.d2 == c.dmin2 / c.diam2,
        "d^2 > 1": c.d2 > 1,
        "ell = 103/100": c.ell == F(103, 100),
        "запас = 7 - ell^2 diam^2": c.margin == c.dmin2 - c.ell * c.ell * c.diam2,
        "запас > 0": c.margin > 0,
        "diam^2 = 4 R^2": c.diam2 == 4 * c.r2,
    }
    return _verdict("R7_1029", conds,
                    f"d^2 = {c.d2}, d = {float(c.d2) ** 0.5:.9f}, "
                    f"запас {float(c.margin):.3e} > 0")


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


def check_r9_7203() -> Check:
    """Точный сертификат ламинирования E8/2401 (m=3); вытеснил 9604."""
    c = _load_exact_cert("dim9_7203_exact.json")
    # прежняя аналитическая ступень: 6/7 + 1/9 = 61/63 < 1 даёт 9604
    superseded = 1 / e8_block_width_squared() + 1 / F(3) ** 2
    conds = {
        "индекс 7203": c.index == 7203,
        "interval_valid": c.interval_valid is True,
        "D_min^2 = 7": c.dmin2 == 7,
        "d^2 = 7/diam^2": c.d2 == c.dmin2 / c.diam2,
        "d^2 > 1": c.d2 > 1,
        "R^2 < 7/4": c.r2 < F(7, 4),
        "diam^2 = 4 R^2": c.diam2 == 4 * c.r2,
        "запас = 7 - ell^2 diam^2": c.margin == c.dmin2 - c.ell * c.ell * c.diam2,
        "запас > 0": c.margin > 0,
        "шире вытесненной 9604": c.d2 > 1 / superseded,
        "цветов меньше 9604": 7203 < 2401 * 4,
    }
    return _verdict("R9_7203", conds,
                    f"d^2 = {c.d2}, d = {float(c.d2) ** 0.5:.9f} против "
                    f"sqrt(63/61) = {float(1 / superseded) ** 0.5:.9f}, "
                    f"запас {float(c.margin):.3e} > 0")


def check_r10_45619() -> Check:
    cost = 1 / e8_block_width_squared() + 1 / planar_block_width_squared()  # 6/7 + 4/31
    conds = {
        "d_2^2 = 31/4": planar_block_width_squared() == F(31, 4),
        "6/7 + 4/31 = 214/217": cost == F(214, 217),
        "сумма < 1": cost < 1,
        "2401*19 = 45619": 2401 * 19 == 45619,
    }
    return _verdict(
        "R10_45619", conds,
        f"6/7 + 4/31 = {cost} < 1, ell = sqrt(217/214) = {float(1 / cost) ** 0.5:.6f}")


CHECKS = (check_r4_43, check_r5_132, check_r7_1029, check_r9_7203, check_r10_45619)


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
        # ~1,5 ч: перечисление 1 654 230 вершин девятимерной ячейки и
        # замыкание по 1-скелету на 17,7 млн рёберных подмножеств
        [py, "-m", "chromatic_research.campaigns.dim9_7203_exact",
         "--output", str(tmp / "dim9_7203_exact.json")],
        # Аудит 132 без Qhull: перечисляет короткие векторы в том же доказуемо
        # полном окне |v| < 2(1+ell)R, что и сертификат (ell берётся из поля
        # certified_interval.upper_endpoint; при ell = 1 это классическое окно
        # 4R), и сверяет их число и KKT-свидетели с сертификатом. Его протокол —
        # results/metric_deform_a5_132_refined_independent_exact_audit.json
        # (certificate_count 38).
        [py, "-m", "chromatic_research.campaigns.verify_exact_voronoi",
         str(results_path("metric_deform_a5_132_refined_certificate.json")),
         "--output", str(tmp / "a5_132_independent_audit.json")],
    ]


def run_full() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        for cmd in full_commands(Path(tmpdir)):
            print(">>", " ".join(cmd[1:]))
            code = subprocess.call(cmd)
            print("   код возврата", code)
            ok = ok and code == 0
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
