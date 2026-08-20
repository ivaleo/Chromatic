"""Точный сертификат ширины k = 15 в R^3: конструкции Кулсона и их деформация.

Историческая 15-раскраска Кулсона --- ОЦК-решётка (A3*) с подрешёткой индекса
15; здесь она реализована как альфа = 1/3 однопараметрического семейства

    G(alpha) = [[1, -a, 2a-1], [-a, 1, -a], [2a-1, -a, 1]],

в котором подрешётка T = [[1,0,11],[0,1,7],[0,0,15]] держится фиксированной.
Символьное решение KKT-систем связывающих орбит даёт замкнутые формы

    D^2_{(3,2,2)} = (4a^2+3a+1)/(a+1),   D^2_{(2,1,-1)} = 2(5a^2-6a+2)/(1-a),
    diam^2 = 2 - a,

откуда при a = 1/3 ширина равна ровно 1 (интервал вырожден --- основная
конструкция Кулсона), а максимум семейства достигается в корне кубики
14a^3 - 3a^2 - 10a + 3 = 0, a* = 0.313695331..., d* = 1.026598584...

В конце работы Кулсона приведена УЛУЧШЕННАЯ 15-раскраска с исключённым
интервалом (sqrt(22), sqrt(389/17)), то есть (1, sqrt(389/374)) после
нормировки, sqrt(389/374) = 1.019856339...  Это в точности точка a = 4/13
семейства: замкнутые формы дают diam^2 = 22/13 и D^2 = 389/221, что при
масштабе 13 совпадает с кулсоновскими 22 и 389/17 дословно.  Точка
сертифицируется здесь наравне с остальными.

Сертифицируется рациональная точка a = 3137/10000 (целочисленный грамиан
10^4*G), лежащая в 4.7e-6 от alpha*: полный точный конвейер --- вершины ячейки
над Q, радиус покрытия, окно |v| < 2(1+l)R, KKT-сертификат каждого вектора ---
как в verify_metric_candidate. Итог:

    chi(R^3, [1, l]) <= 15   при l <= 102659/100000 = 1.02659   [С]

Сам максимум семейства --- алгебраическое число: alpha* задаётся кубикой плюс
рациональным изолирующим интервалом, а d*^2 = (602 alpha*^2 - 87 alpha* + 143)/166
(замечание к версии 6). Функция alpha_star_bound() проверяет это в чистой
рациональной арифметике: изоляцию корня, монотонность связывающей ветви и
итоговое неравенство d* > 102659/100000.

Запуск::

    python -m chromatic_research.campaigns.dim3_k15
"""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction

import numpy as np
from sympy import Matrix, Rational

import combigeo
from chromatic_research.campaigns.verify_metric_candidate import (
    exact_positive_definite,
    exact_projection_certificate,
    exact_vertex_radius,
    fraction_text,
    voronoi_data,
)
from chromatic_research.core.prime_radon import hnf_columns
from chromatic_research.paths import results_path

T_ROWS = np.array([[1, 0, 11], [0, 1, 7], [0, 0, 15]], dtype=np.int64)


# --- точный максимум семейства: alpha* --- корень кубики в изолирующем окне ---
ALPHA_LO = Fraction(3_136_953, 10**7)
ALPHA_HI = Fraction(3_136_954, 10**7)
ELL_STAR = Fraction(102_659, 100_000)


def cubic(x: Fraction) -> Fraction:
    """p(x) = 14x^3 - 3x^2 - 10x + 3; alpha* --- его корень в (ALPHA_LO, ALPHA_HI)."""
    return 14 * x**3 - 3 * x**2 - 10 * x + 3


def branch_a(x: Fraction) -> Fraction:
    """r_A(x) = (4x^2+3x+1)/((x+1)(2-x)) --- квадрат ширины на ветви (3,2,2)."""
    return (4 * x**2 + 3 * x + 1) / ((x + 1) * (2 - x))


def alpha_star_bound() -> dict:
    """Строгая рациональная граница d(alpha*) > 102659/100000.

    Ни одного вычисления с плавающей точкой. Три шага:

    1. Изоляция корня: p(lo) > 0, p(hi) < 0 --- корень есть; p''(x) = 84x - 6 > 0
       на окне, а p'(hi) = 42 hi^2 - 6 hi - 10 < 0, значит p' < 0 всюду на окне,
       p строго убывает и корень един.  Это и есть определение alpha*.
    2. Монотонность: r_A'(x) = (7x^2 + 18x + 5)/((x-2)^2 (x+1)^2), числитель при
       x >= 0 положителен (корни -2.2546 и -0.3168), поэтому r_A возрастает и
       r_A(alpha*) > r_A(lo).
    3. Одно точное рациональное неравенство: r_A(lo) - ELL_STAR^2 > 0.

    В самой точке alpha* обе ветви совпадают (это и определяет кубику), поэтому
    d(alpha*)^2 = r_A(alpha*) > ELL_STAR^2.
    """
    p_lo, p_hi = cubic(ALPHA_LO), cubic(ALPHA_HI)
    dp_hi = 42 * ALPHA_HI**2 - 6 * ALPHA_HI - 10
    ddp_lo = 84 * ALPHA_LO - 6
    margin = branch_a(ALPHA_LO) - ELL_STAR**2
    assert p_lo > 0 and p_hi < 0, "изолирующее окно alpha* не меняет знак p"
    assert ddp_lo > 0 and dp_hi < 0, "p' не отрицательна на всём окне alpha*"
    assert margin > 0, "рациональная граница d* > 102659/100000 не проходит"
    return {
        "cubic": "14a^3 - 3a^2 - 10a + 3",
        "isolating_interval": [fraction_text(ALPHA_LO), fraction_text(ALPHA_HI)],
        "p_at_lo": fraction_text(p_lo),
        "p_at_hi": fraction_text(p_hi),
        "p_prime_at_hi": fraction_text(dp_hi),
        "p_double_prime_at_lo": fraction_text(ddp_lo),
        "branch_a_derivative_numerator": "7a^2 + 18a + 5",
        "branch_a_at_lo": fraction_text(branch_a(ALPHA_LO)),
        "ell": fraction_text(ELL_STAR),
        "rational_margin": fraction_text(margin),
        "width_squared_closed_form": "(602 a^2 - 87 a + 143)/166",
        "alpha_float": 0.3136953314651367,
        "width_float": 1.0265985837974323,
        "proves": "d(alpha*) > 102659/100000",
    }


def closed_forms(alpha: Fraction) -> dict[str, Fraction]:
    d2_a = (4 * alpha**2 + 3 * alpha + 1) / (alpha + 1)
    d2_b = 2 * (5 * alpha**2 - 6 * alpha + 2) / (1 - alpha)
    diam2 = 2 - alpha
    return {"D2_322": d2_a, "D2_21m1": d2_b, "diam2": diam2,
            "width2": min(d2_a, d2_b) / diam2}


def audit(alpha: Fraction, ell: Fraction | None) -> dict:
    denominator = alpha.denominator
    a_num = alpha.numerator
    gram_int = np.array(
        [[denominator, -a_num, 2 * a_num - denominator],
         [-a_num, denominator, -a_num],
         [2 * a_num - denominator, -a_num, denominator]], dtype=np.int64)
    exact_positive_definite(gram_int)
    basis = np.linalg.cholesky(gram_int.astype(np.float64) / denominator)

    facets, facet_normals, hull, _ = voronoi_data(basis)
    facet_coordinates = np.rint(
        facet_normals @ np.linalg.inv(basis)).astype(np.int64)
    assert np.allclose(facet_coordinates @ basis, facet_normals, atol=1e-9)
    assert all(len(active) == 3 for active in hull.dual_facets)
    radius_sq, _, _ = exact_vertex_radius(
        gram_int, denominator, facet_coordinates, hull, progress_every=0)
    radius_sq = Fraction(int(radius_sq.p), int(radius_sq.q))
    diam_sq = 4 * radius_sq
    forms = closed_forms(alpha)
    assert diam_sq == forms["diam2"], (diam_sq, forms["diam2"])

    # окно нарушителя: для цели d > l достаточно |v| < 2(1+l)R; при l=None
    # (калибровка Кулсона) берём то же окно с l = 1
    ell_eff = ell if ell is not None else Fraction(1)
    cutoff_sq = 4 * (1 + ell_eff) ** 2 * radius_sq
    p, q = cutoff_sq.numerator, cutoff_sq.denominator

    kernel = hnf_columns(T_ROWS.T.copy())
    sub = np.asarray(kernel).T @ basis
    reduced = np.asarray(combigeo.lll_reduce(sub.tolist()), dtype=np.float64)
    reduced_rows = np.rint(reduced @ np.linalg.inv(basis)).astype(np.int64)
    assert np.allclose(reduced_rows @ basis, reduced, atol=1e-7)

    sub_gram = Matrix(reduced_rows.tolist()) * Matrix(gram_int.tolist()) \
        * Matrix(reduced_rows.tolist()).T
    sub_inv = sub_gram.inv() * denominator
    box = []
    for i in range(3):
        entry = Rational(sub_inv[i, i])
        bound = Fraction(int(entry.p), int(entry.q)) * cutoff_sq
        box.append(math.isqrt(int(bound)))

    S = [[int(sub_gram[i, j]) for j in range(3)] for i in range(3)]
    vectors = []
    for z in itertools.product(*[range(-b, b + 1) for b in box]):
        if not any(z):
            continue
        norm_num = sum(z[i] * S[i][j] * z[j]
                       for i in range(3) for j in range(3))
        # нестрогое сравнение: граничные векторы (|v| = 2(1+l)R) включаются,
        # это надмножество окна нарушителя и потому всегда корректно
        if norm_num * q <= p * denominator:
            vectors.append(np.asarray(z, dtype=np.int64) @ reduced_rows)

    certificates = [
        exact_projection_certificate(
            coord, basis, facets, facet_coordinates, gram_int, denominator)
        for coord in vectors]
    distances = [Fraction(c["distance_squared"]) for c in certificates]
    min_d_sq = min(distances)
    width_sq = Fraction(min_d_sq, diam_sq)
    assert width_sq == forms["width2"], (width_sq, forms["width2"])

    record = {
        "alpha": f"{alpha.numerator}/{alpha.denominator}",
        "integer_gram": gram_int.tolist(),
        "denominator": denominator,
        "transition_rows": T_ROWS.tolist(),
        "index": 15,
        "facets": len(facets),
        "vertices": len(hull.intersections),
        "covering_radius_squared": f"{radius_sq.numerator}/{radius_sq.denominator}",
        "diameter_squared": f"{diam_sq.numerator}/{diam_sq.denominator}",
        "window_cutoff_squared": f"{p}/{q}",
        "coefficient_box": box,
        "vector_count": len(vectors),
        "minimum_distance_squared":
            f"{min_d_sq.numerator}/{min_d_sq.denominator}",
        "width_squared": f"{width_sq.numerator}/{width_sq.denominator}",
        "width": math.sqrt(float(width_sq)),
        "closed_forms_match": True,
        "all_projection_certificates": certificates,
    }
    if ell is not None:
        margin = min_d_sq - ell * ell * diam_sq
        assert margin > 0, f"interval l={ell} not certified: margin={margin}"
        record["certified_interval"] = {
            "valid": True,
            "upper_endpoint": f"{ell.numerator}/{ell.denominator}",
            "squared_margin": f"{margin.numerator}/{margin.denominator}",
            "squared_margin_float": float(margin),
        }
    else:
        record["degenerate_interval"] = bool(width_sq == 1)
    return record


def main() -> int:
    coulson = audit(Fraction(1, 3), None)
    print(f"Кулсон, ОЦК (alpha=1/3): width^2 = {coulson['width_squared']} "
          f"(вырожденный интервал: {coulson['degenerate_interval']}), "
          f"векторов в окне: {coulson['vector_count']}", flush=True)
    assert coulson["width_squared"] == "1/1"

    improved = audit(Fraction(4, 13), Fraction(1019, 1000))
    assert improved["width_squared"] == "389/374"
    # кулсоновские числа получаются масштабированием формы на 13
    improved["coulson_scaling"] = {
        "factor": 13,
        "diameter_squared": "22",
        "minimum_distance_squared": "389/17",
        "published_interval": "(sqrt(22), sqrt(389/17))",
        "normalised_interval": "(1, sqrt(389/374)) ~ (1, 1.019856339)",
    }
    assert Fraction(improved["diameter_squared"]) * 13 == 22
    assert (Fraction(improved["minimum_distance_squared"]) * 13
            == Fraction(389, 17))
    print(f"Кулсон, улучшенная (alpha=4/13): width = {improved['width']:.9f} "
          f"(width^2 = {improved['width_squared']}), масштаб 13 даёт "
          f"diam^2 = 22 и D^2 = 389/17 --- числа статьи Кулсона", flush=True)

    star = alpha_star_bound()
    print(f"alpha* (корень {star['cubic']} в {star['isolating_interval']}): "
          f"d* = {star['width_float']:.12f}, рациональная граница "
          f"d* > {star['ell']} доказана (запас {star['rational_margin']})",
          flush=True)

    optimal = audit(Fraction(3137, 10000), ELL_STAR)
    print(f"alpha=3137/10000: width = {optimal['width']:.9f} "
          f"(width^2 = {optimal['width_squared']}), "
          f"интервал l={ELL_STAR} подтверждён, векторов: "
          f"{optimal['vector_count']}", flush=True)

    legacy = audit(Fraction(16, 51), Fraction(513, 500))
    print(f"alpha=16/51 (прежняя точка): width = {legacy['width']:.9f}, "
          f"интервал l=513/500 подтверждён", flush=True)

    payload = {
        "method": "family G(a)=[[1,-a,2a-1],[-a,1,-a],[2a-1,-a,1]] with fixed "
                  "index-15 sublattice; exact vertices + exact KKT projections",
        "closed_forms": {
            "D2_orbit_322": "(4a^2+3a+1)/(a+1)",
            "D2_orbit_21m1": "2(5a^2-6a+2)/(1-a)",
            "diam2": "2-a",
            "optimal_alpha_cubic": "14a^3 - 3a^2 - 10a + 3 = 0",
            "optimal_alpha_float": 0.3136953314651367,
            "optimal_width_float": 1.0265985837974336,
            "coulson_bcc_alpha": "1/3",
            "coulson_improved_alpha": "4/13",
        },
        "alpha_star": star,
        "coulson_bcc": coulson,
        "coulson_improved": improved,
        "certified_optimum": optimal,
        "earlier_rational_point": legacy,
        "certified_upper_bound": 15,
    }
    path = results_path("dim3_k15_certificate.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"written {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
