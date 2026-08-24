"""Контроли оболочечного пола.

Главное утверждение, которое здесь проверяется: на родителе E₈ пол совпадает с
рекордом, то есть 2401 оптимален. Доказательство короткое (Коши–Шварц плюс
целочисленность плюс постоянная Эрмита), поэтому тесты проверяют каждое звено
отдельно, а не только итоговое число.
"""

import math
from fractions import Fraction as Fr

import pytest

from chromatic_research.campaigns.shell_floor import (
    E8_GRAM,
    analyse,
    dot,
    a3_star_parent,
    e6_star_parent,
    e8_parent,
    max_inner_product,
    radial_witness,
    shells_up_to,
)


@pytest.fixture(scope="module")
def e8():
    return e8_parent()


def test_e8_gram_is_even_unimodular(e8):
    """E₈: чётная, det 1, минимум 2."""
    import numpy as np

    matrix = np.array(E8_GRAM, dtype=float)
    assert round(np.linalg.det(matrix)) == 1
    assert all(E8_GRAM[i][i] % 2 == 0 for i in range(8))
    shells = shells_up_to(e8, Fr(4))
    assert min(shells) == Fr(2)
    assert shells[Fr(2)] == 240          # корни E₈
    assert shells[Fr(4)] == 2160
    assert all(value.denominator == 1 and value.numerator % 2 == 0
               for value in shells)


def test_e8_window_shells(e8):
    """Окно [(2R+λ₁)², (2·diam)²] = [11.657, 16]: оболочки 12, 14, 16."""
    low = (2 * math.sqrt(float(e8.covering_sq))
           + math.sqrt(float(e8.lambda1_sq))) ** 2
    assert low == pytest.approx((2 + math.sqrt(2)) ** 2)
    shells = shells_up_to(e8, 4 * e8.diam_sq)
    window = [k for k in shells if float(k) >= low - 1e-12]
    assert window == [Fr(12), Fr(14), Fr(16)]
    assert shells[Fr(12)] == 60480
    assert shells[Fr(14)] == 82560


def test_cauchy_schwarz_integrality_bound():
    """⟨v,u⟩ ≤ ⌊√(N·M)⌋ при целых скалярных произведениях."""
    assert max_inner_product(Fr(12), Fr(2), 1) == Fr(4)     # √24 = 4.899
    assert max_inner_product(Fr(12), Fr(4), 1) == Fr(6)     # √48 = 6.928
    # знаменатель 3 (E₆*): √(8·4/3) = 3.266 ⇒ 3
    assert max_inner_product(Fr(8), Fr(4, 3), 3) == Fr(3)


def test_e8_shell_12_witness_is_quarter(e8):
    """Свидетель для оболочки 12 — ровно v/4, и он даёт D² ≤ 3."""
    shells = shells_up_to(e8, 4 * e8.diam_sq)
    witness = radial_witness(e8, Fr(12), shells)
    assert witness is not None
    scale, distance_sq, _ = witness
    assert distance_sq == Fr(3)
    assert distance_sq < e8.diam_sq
    # оценка D² = 4(1/2 − s)²·12 при s = 1/4 равна ровно 3
    assert 4 * (Fr(1, 2) - Fr(1, 4)) ** 2 * 12 == Fr(3)
    assert scale <= Fr(1, 4)


def test_e8_shell_12_witness_verified_directly(e8):
    """Прямая проверка v/4 ∈ V₀ на выборке векторов оболочки 12."""
    from chromatic_research.core.exact_layer_cert import enumerate_shifted

    zero = [Fr(0)] * 8
    small = [u for u in enumerate_shifted(e8.gram, zero, Fr(4)) if any(u)]
    shell = [v for v in enumerate_shifted(e8.gram, zero, Fr(12))
             if dot(e8.gram, v, v) == Fr(12)]
    assert len(shell) == 60480
    for vector in shell[:60]:
        quarter = [Fr(c, 4) for c in vector]
        for u in small:
            assert dot(e8.gram, quarter, u) <= dot(e8.gram, u, u) / 2


def test_e8_shell_14_is_not_forbidden(e8):
    """Оболочка 14 запрещаться не должна — на ней стоит сам рекорд."""
    shells = shells_up_to(e8, 4 * e8.diam_sq)
    assert radial_witness(e8, Fr(14), shells) is None


def test_e8_floor_equals_the_record(e8):
    """Пол 14⁴/16 = 2401 совпадает с рекордом (3+ω)E₈."""
    assert Fr(14) ** 4 / Fr(16) == Fr(2401)
    report = analyse(e8)
    assert report["first_allowed_shell"] == "14"
    assert report["index_floor"] == 2401
    assert report["record_is_optimal"] is True


def test_champion_attains_the_bound():
    """(3+ω)E₈: индекс N(3+ω)⁴ = 7⁴ и λ₁² = 7·2 = 14 — равенство в Эрмите."""
    assert 7 ** 4 == 2401
    assert 7 * 2 == 14
    # равенство возможно лишь потому, что Γ подобна E₈, а E₈ достигает γ₈ = 2
    assert Fr(14) ** 4 / Fr(16) == 7 ** 4


def test_a3_star_invariants():
    """ОЦК: λ₁² = 3/4, R² = 5/16 (вершина усечённого октаэдра), det = 1/2."""
    import numpy as np

    parent = a3_star_parent()
    matrix = np.array([[float(x) for x in row] for row in parent.gram])
    assert round(np.linalg.det(matrix), 12) == 0.25          # (det Λ)² = 1/4
    shells = shells_up_to(parent, Fr(5))
    assert min(shells) == Fr(3, 4)
    assert shells[Fr(3, 4)] == 8                              # (±½,±½,±½)
    assert shells[Fr(1)] == 6
    # окно: оболочки 4, 19/4, 5 — суммы трёх нечётных квадратов дают 19/4
    window = [k for k in shells if float(k) >= 3.9364]
    assert window == [Fr(4), Fr(19, 4), Fr(5)]
    assert shells[Fr(4)] == 6 and shells[Fr(19, 4)] == 24


def test_a3_star_floor_equals_the_record():
    """Пол на ОЦК равен 15 — то есть рекорд Кулсона неулучшаем на этом родителе."""
    report = analyse(a3_star_parent())
    assert report["first_allowed_shell"] == "19/4"
    assert report["index_floor"] == 15
    assert report["record_is_optimal"] is True
    # k² ≥ (19/4)³/(1/2) = 6859/32 = 214.34…, √ = 14.64… ⇒ 15
    assert Fr(19, 4) ** 3 / Fr(1, 2) == Fr(6859, 32)
    assert math.ceil(math.sqrt(6859 / 32)) == 15


def test_e6_star_floor_is_305():
    report = analyse(e6_star_parent())
    assert report["first_allowed_shell"] == "28/3"
    assert report["index_floor"] == 305
    assert report["record_is_optimal"] is False
    assert report["record_index"] == 343


def test_k12_floor_is_conditional_and_improves():
    """K₁₂: пол поднят до 177979, но результат условен по γ₁₂."""
    from chromatic_research.campaigns.shell_floor import k12_parent

    report = analyse(k12_parent())
    assert report["first_allowed_shell"] == "30"
    assert report["index_floor"] == 177_979
    assert report["record_index"] == 3 ** 12
    assert report["conditional"]                      # непустая пометка
    assert not report["record_is_optimal"]
    # k ≥ (30/4)⁶ = 7.5⁶ = 177978.515625
    assert math.ceil((Fr(30) / 4) ** 6) == 177_979


def test_leech_floor_is_unconditional():
    """Λ₂₄: γ₂₄ = 4 доказана (Кон–Кумар), поэтому пол безусловен."""
    from chromatic_research.campaigns.shell_floor import leech_parent

    report = analyse(leech_parent())
    assert report["first_allowed_shell"] == "26"
    assert report["record_index"] == 7 ** 12
    assert report["conditional"] == ""
    # k ≥ (26/4)^12 = 6.5^12 = 5688009063.1…
    assert report["index_floor"] == math.ceil(float((Fr(26) / 4) ** 12))
    assert report["index_floor"] < 7 ** 12


def test_leech_shell_26_is_not_closed():
    """Оболочка 26 у Лича НЕ закрывается — и это узкое место, а не недосмотр.

    Свидетель ``s = 2/9`` даёт ``D² ≤ 4(1/2−2/9)²·26 = 2600/324 = 8.0247``,
    что чуть БОЛЬШЕ ``diam² = 8``; а больший ``s`` запрещён минимальными
    векторами (``⟨v,u⟩ ≤ ⌊√104⌋ = 10``, откуда ``s ≤ 1/5``… с уточнением до 9
    получается ``s ≤ 2/9``).  Разрыв 0.3 %.
    """
    from chromatic_research.campaigns.shell_floor import leech_parent

    parent = leech_parent()
    assert 4 * (Fr(1, 2) - Fr(2, 9)) ** 2 * 26 == Fr(2600, 324)
    assert Fr(2600, 324) > parent.diam_sq
    shells = shells_up_to(parent, 4 * parent.diam_sq)
    assert radial_witness(parent, Fr(26), shells) is None


def test_forbidden_shell_bounds_are_strict(e8):
    """Оценка D² свидетеля обязана быть СТРОГО ниже diam²."""
    for parent in (e8, e6_star_parent(), a3_star_parent()):
        report = analyse(parent)
        for entry in report["forbidden_shells"]:
            numerator, _, denominator = entry["d2_bound"].partition("/")
            value = Fr(int(numerator), int(denominator) if denominator else 1)
            assert value < parent.diam_sq
