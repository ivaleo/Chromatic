"""Пересчёт окна перечисления для интервальных сертификатов 132 и 1323.

Исходные сертификаты перечисляли короткие векторы подрешётки в окне
``|v| < 4R`` --- этого достаточно для цели ``d > 1``, но для интервала
``[1, l]`` нарушитель ограничен лишь ``|v| < 2(1+l)R`` (из
``D(v) >= |v| - 2R``).  Между двумя окнами лежит по одной паре векторов на
сертификат, и текст статьи (38 и 72 вектора) опережал артефакты (36 и 70).

Скрипт заново перечисляет расширенное окно точно:

1. проверяет положительную определённость и радиус покрытия по всем вершинам
   (та же машинерия, что в ``verify_metric_candidate``);
2. доказывает индивидуальные границы коэффициентов ``|z_i| <= b_i`` из
   ``z_i^2 <= |v|_Q^2 (S^{-1})_ii`` и перечисляет бокс в целочисленной
   арифметике со строгим сравнением ``|v|^2 < 4(1+l)^2 R^2``;
3. сверяет список с независимым C++-перечислителем Финке--Похста;
4. строит точный KKT-сертификат проекции для каждого вектора окна;
5. переписывает артефакт: новое окно, все сертификаты, блок
   ``certified_interval`` (для 1323 его не было вовсе).

Запуск::

    python -m chromatic_research.campaigns.certificate_window_fix a5
    python -m chromatic_research.campaigns.certificate_window_fix e7
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
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
from chromatic_research.paths import results_path

TARGETS = {
    "a5": ("metric_deform_a5_132_refined_certificate.json", Fraction(101, 100)),
    "e7": ("metric_deform_e7_1323_certificate.json", Fraction(1007, 1000)),
}


def parse_fraction(text: str) -> Fraction:
    return Fraction(text) if "/" in text else Fraction(int(text), 1)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "a5"
    name, ell = TARGETS[tag]
    path = results_path(name)
    payload = json.loads(path.read_text())

    denominator = int(payload["denominator"])
    gram_int = np.asarray(payload["integer_gram"], dtype=np.int64)
    n = gram_int.shape[0]
    kernel = np.asarray(payload["kernel_basis_columns"], dtype=np.int64)
    reduced_rows = np.asarray(payload["lll_kernel_basis_rows"], dtype=np.int64)

    exact_positive_definite(gram_int)
    basis = np.linalg.cholesky(gram_int.astype(np.float64) / denominator)

    facets, facet_normals, hull, _ = voronoi_data(basis)
    facet_coordinates = np.rint(
        facet_normals @ np.linalg.inv(basis)).astype(np.int64)
    assert np.allclose(facet_coordinates @ basis, facet_normals, atol=1e-7)
    assert all(len(active) == n for active in hull.dual_facets), "not simple"
    print(f"[{tag}] facets={len(facets)} vertices={len(hull.intersections)}; "
          "exact vertex audit...", flush=True)

    radius_sq, farthest, singular = exact_vertex_radius(
        gram_int, denominator, facet_coordinates, hull)
    stored_radius_sq = parse_fraction(
        payload["voronoi"]["covering_radius_squared"])
    radius_sq_frac = Fraction(int(radius_sq.p), int(radius_sq.q))
    assert radius_sq_frac == stored_radius_sq, (
        f"covering radius changed: {radius_sq_frac} != {stored_radius_sq}")
    print(f"[{tag}] R^2 confirmed = {payload['voronoi']['covering_radius_squared']}",
          flush=True)

    # окно нарушителя: |v| < 2(1+l)R, т.е. |v|^2 < 4(1+l)^2 R^2
    cutoff_sq = 4 * (1 + ell) ** 2 * radius_sq_frac
    old_cutoff_sq = 16 * radius_sq_frac

    # индивидуальные границы координат: z_i^2 <= |v|^2 (S/den)^{-1}_ii
    reduced_m = Matrix(reduced_rows.tolist())
    gram_m = Matrix(gram_int.tolist())
    sub_gram_int = reduced_m * gram_m * reduced_m.T          # = S * den
    sub_inv_times_den = sub_gram_int.inv() * denominator     # = S^{-1}
    bounds, box = [], []
    for i in range(n):
        entry = Rational(sub_inv_times_den[i, i])
        bound = Fraction(int(entry.p), int(entry.q)) * cutoff_sq
        bounds.append(bound)
        box.append(math.isqrt(int(bound)))                   # z_i^2 < bound
    print(f"[{tag}] ell={ell}  cutoff^2={float(cutoff_sq):.6f} "
          f"(old {float(old_cutoff_sq):.6f})  box={box}", flush=True)

    # точное перечисление бокса в целых числах
    S = [[int(sub_gram_int[i, j]) for j in range(n)] for i in range(n)]
    p, q = cutoff_sq.numerator, cutoff_sq.denominator
    vectors: list[np.ndarray] = []
    start = time.perf_counter()
    for z in itertools.product(*[range(-b, b + 1) for b in box]):
        if not any(z):
            continue
        norm_num = sum(z[i] * S[i][j] * z[j]
                       for i in range(n) for j in range(n))
        # |v|^2 = norm_num/den < p/q  <=>  norm_num*q < p*den (всё целое)
        if norm_num * q < p * denominator:
            vectors.append(np.asarray(z, dtype=np.int64) @ reduced_rows)
    print(f"[{tag}] box enumerated in {time.perf_counter()-start:.1f}s: "
          f"{len(vectors)} nonzero vectors", flush=True)

    # независимая C++-сверка тем же радиусом
    cpp = combigeo._vectors_near(
        (kernel.T @ basis).tolist(), [0.0] * n,
        math.sqrt(float(cutoff_sq)) + 1e-8)
    inv_basis = np.linalg.inv(basis)
    cpp_coords = set()
    for vec in cpp:
        if np.linalg.norm(vec) < 1e-10:
            continue
        coord = np.rint(np.asarray(vec) @ inv_basis).astype(np.int64)
        norm_num = int(coord @ gram_int @ coord)
        if norm_num * q < p * denominator:      # строгое точное окно
            cpp_coords.add(tuple(coord.tolist()))
    exact_coords = {tuple(v.tolist()) for v in vectors}
    assert cpp_coords == exact_coords, (
        f"enumerators disagree: exact-only={exact_coords - cpp_coords}, "
        f"cpp-only={cpp_coords - exact_coords}")

    # какие пары добавились против старого окна 16R^2
    extra = sorted(
        tuple(v.tolist()) for v in vectors
        if int(np.asarray(v) @ gram_int @ np.asarray(v))
        * old_cutoff_sq.denominator
        >= old_cutoff_sq.numerator * denominator)
    print(f"[{tag}] beyond 4R window: {extra}", flush=True)

    print(f"[{tag}] exact KKT audit of {len(vectors)} vectors...", flush=True)
    start = time.perf_counter()
    certificates = []
    for k, coord in enumerate(vectors, 1):
        certificates.append(exact_projection_certificate(
            coord, basis, facets, facet_coordinates, gram_int, denominator))
        if k % 10 == 0:
            print(f"  kkt {k}/{len(vectors)} "
                  f"elapsed={time.perf_counter()-start:.0f}s", flush=True)

    distances_sq = [parse_fraction(c["distance_squared"])
                    for c in certificates]
    min_d_sq = min(distances_sq)
    stored_min = parse_fraction(
        payload["separation"]["minimum_distance_squared"])
    assert min_d_sq == stored_min, (
        f"minimum changed: {min_d_sq} != {stored_min} — новая пара стала "
        "минимумом, текст статьи требует пересмотра")
    diam_sq = 4 * radius_sq_frac
    margin = min_d_sq - diam_sq
    interval_margin = min_d_sq - ell * ell * diam_sq
    assert margin > 0 and interval_margin > 0
    ratio = math.sqrt(float(Fraction(min_d_sq, diam_sq)))
    for coord in extra:
        i = [tuple(v.tolist()) for v in vectors].index(coord)
        d_over = math.sqrt(float(Fraction(distances_sq[i], diam_sq)))
        print(f"[{tag}] extra pair {coord}: D/diam = {d_over:.6f}", flush=True)

    witnesses = [c for c, d in zip(certificates, distances_sq)
                 if d == min_d_sq]
    payload["method"] = (
        "rational Gram + exact vertices + exact KKT projections; "
        "enumeration window |v| < 2(1+ell)R")
    payload["short_vector_certificate"] = {
        "window_definition": "|v|^2 < 4(1+ell)^2 R^2, ell = "
                             + fraction_text(Rational(ell.numerator,
                                                      ell.denominator)),
        "length_cutoff_squared": f"{p}/{q}" if q != 1 else str(p),
        "coefficient_bounds_squared": [
            (f"{b.numerator}/{b.denominator}" if b.denominator != 1
             else str(b.numerator)) for b in bounds],
        "coefficient_box": box,
        "exact_vector_count": len(vectors),
        "cpp_vector_count": len(cpp_coords),
        "beyond_4R_pairs": extra,
    }
    payload["separation"] = {
        "valid": True,
        "minimum_distance_squared": fraction_text(
            Rational(min_d_sq.numerator, min_d_sq.denominator)),
        "diameter_squared": fraction_text(
            Rational(diam_sq.numerator, diam_sq.denominator)),
        "squared_margin": fraction_text(
            Rational(margin.numerator, margin.denominator)),
        "squared_margin_float": float(margin),
        "distance_ratio": ratio,
        "minimum_witnesses": witnesses,
        "all_projection_certificates": certificates,
    }
    payload["certified_interval"] = {
        "valid": True,
        "upper_endpoint": f"{ell.numerator}/{ell.denominator}",
        "squared_margin": fraction_text(
            Rational(interval_margin.numerator, interval_margin.denominator)),
        "squared_margin_float": float(interval_margin),
    }
    payload["certified_upper_bound"] = int(payload["kernel_determinant"])
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[{tag}] rewritten: {path}  vectors={len(vectors)} "
          f"ratio={ratio:.12f}  interval ell={ell} margin="
          f"{float(interval_margin):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
