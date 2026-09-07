"""Единый запуск проверки пяти заголовочных утверждений (уровень 1).

`verify_main_results` пересчитывает итоговые неравенства пяти результатов
из опубликованных дробей сертификатов и двух аналитических сумм; тест
закрепляет, что все пять строк проходят и что `main([])` возвращает 0.
"""

from fractions import Fraction

from chromatic_research.campaigns import verify_main_results as V


def test_each_claim_passes():
    results = V.run_all()
    ids = [r.claim for r in results]
    assert ids == ["R4_43", "R5_132", "R7_1029", "R9_7203", "R10_45619"]
    failed = [f"{r.claim}: {r.detail}" for r in results if not r.ok]
    assert not failed, "не прошли:\n  " + "\n  ".join(failed)


def test_analytic_sums_are_exact():
    assert V.check_r9_7203().ok
    assert V.check_r10_45619().ok
    assert V.planar_block_width_squared() == Fraction(31, 4)
    assert V.e8_block_width_squared() == Fraction(7, 6)


def test_main_returns_zero(capsys):
    assert V.main([]) == 0
    out = capsys.readouterr().out
    assert out.count("OK ") == 5 and "FAIL" not in out
