"""Инварианты двух коротких версий (14.09.2026): сообщение для Докладов РАН
(paper/note-dan/dan.tex) и короткая английская версия для arXiv
(paper/arxiv-en/bounds-en.tex).

Числа английской версии обязаны совпадать с точными сертификатами
audit-data/results; сообщение для Докладов не должно ссылаться на
неопубликованные работы и упоминать численные кандидаты.
"""

import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DAN = ROOT / "paper" / "note-dan" / "dan.tex"
EN = ROOT / "paper" / "arxiv-en" / "bounds-en.tex"
RES = ROOT / "audit-data" / "results"


def _load(stem):
    path = RES / f"{stem}.json"
    if not path.exists():
        pytest.skip(f"нет артефакта {stem}.json")
    return json.loads(path.read_text())


def _has_fraction(tex, value):
    """Дробь как \frac{p}{q} (возможно, с переносом строки между скобками)
    или как p/q."""
    f = Fraction(value)
    compact = "".join(tex.split())
    return (f"{f.numerator}}}{{{f.denominator}}}" in compact
            or f"{f.numerator}/{f.denominator}" in compact)


def _rows(matrix):
    return ["&".join(str(x) for x in row) for row in matrix]


def test_en_7203_data_matches_certificate():
    d = _load("dim9_7203_exact")
    tex = EN.read_text(encoding="utf-8")
    G, C = d["integer_gram"], d["sublattice_columns"]
    assert d["gram_denominator"] == 10000
    # Q = (3/2 A, g; g^T, 17982/10^4): верхний блок G равен 15000*A
    A = [[Fraction(G[i][j], 15000) for j in range(8)] for i in range(8)]
    assert all(x.denominator == 1 for row in A for x in row)
    for row in _rows([[int(x) for x in r] for r in A]):
        assert row in tex, f"строка A {row} не найдена"
    g = [G[i][8] for i in range(8)]
    assert ",\\,".join(str(x) for x in g) in tex, "столбец g для 7203"
    assert str(G[8][8]) in tex
    for row in _rows(C):
        assert row in tex, f"строка C {row} не найдена"
    for key in ("diameter_squared_exact", "radius_squared_exact",
                "normalized_distance_squared_exact", "margin_exact"):
        assert _has_fraction(tex, d[key]), key
    assert Fraction(d["minimum_distance_squared_exact"]) == 7
    assert d["facets"] == 752 and d["vertices"] == 1654230
    assert d["global_vectors"] == 280 and d["global_pairs"] == 140
    assert len(d["minimizers"]) == 120  # пар; в тексте «240 векторов (120 пар)»
    hist = {int(k): v for k, v in d["active_size_histogram"].items()}
    assert (hist[9], hist[15], hist[16]) == (1649640, 4320, 270)


def test_en_1029_data_matches_certificate():
    d = _load("dim7_1029_exact")
    tex = EN.read_text(encoding="utf-8")
    G, C = d["integer_gram"], d["sublattice_columns"]
    M = [[Fraction(G[i][j], 7500) for j in range(6)] for i in range(6)]
    assert all(x.denominator == 1 for row in M for x in row)
    for row in _rows([[int(x) for x in r] for r in M]):
        assert row in tex, f"строка M {row} не найдена"
    g = [G[i][6] for i in range(6)]
    assert ",\\,".join(str(x) for x in g) in tex, "столбец g для 1029"
    assert str(G[6][6]) in tex
    # C описана словами: диагональ (7,1,7,1,7,1,3), C12=C34=C56=-5
    expected = [[0] * 7 for _ in range(7)]
    for i, v in enumerate((7, 1, 7, 1, 7, 1, 3)):
        expected[i][i] = v
    for i in (0, 2, 4):
        expected[i][i + 1] = -5
    assert C == expected
    for key in ("diameter_squared_exact", "radius_squared_exact",
                "normalized_distance_squared_exact"):
        assert _has_fraction(tex, d[key]), key
    assert Fraction(d["minimum_distance_squared_exact"]) == 7
    assert d["facets"] == 254 and d["vertices"] == 30368
    assert d["global_vectors"] == 74 and d["global_pairs"] == 37
    assert len(d["minimizers"]) == 27  # пар; в тексте «$27$ pairs»
    assert "$27$ pairs" in tex


def test_en_43_and_132_data_match_certificates():
    tex = EN.read_text(encoding="utf-8")
    d4 = _load("dim4_k43_verify")
    for key in ("diam2", "D2", "d2"):
        assert _has_fraction(tex, d4[key]), key
    assert d4["transition"] == [[1, 0, 0, 41], [0, 1, 0, 14],
                                [0, 0, 1, 37], [0, 0, 0, 43]]
    assert d4["n_vertices"] == 120 and d4["n_relevant_pairs"] == 15
    d5 = _load("metric_deform_a5_132_refined_certificate")
    for row in _rows(d5["integer_gram"]):
        assert row in tex, f"строка Грама 132 {row}"
    assert _has_fraction(tex, d5["voronoi"]["covering_radius_squared"])
    assert _has_fraction(tex, d5["separation"]["minimum_distance_squared"])
    assert _has_fraction(tex, d5["certified_interval"]["squared_margin"])
    assert d5["certified_interval"]["upper_endpoint"] == "101/100"
    assert d5["voronoi"]["facets"] == 62 and d5["voronoi"]["vertices"] == 720


def test_short_versions_state_only_proven_bounds():
    pat = re.compile(r"\\chi\\?\(\\R\^\{?(9|10)\}?\)\s*\\le\s*(28812|21609)")
    for path in (DAN, EN):
        t = path.read_text(encoding="utf-8")
        assert not pat.search(t), f"кандидат записан как оценка: {path.name}"
    dan = DAN.read_text(encoding="utf-8")
    body = dan[dan.index(r"\maketitle"):]
    for banned in ("28812", "21609", "1323", r"\cite{Full}", r"\cite{Bounds}",
                   r"\cite{Widths}", "tabular", "includegraphics"):
        assert banned not in body, f"в сообщении для Докладов лишнее: {banned}"
    for token in ("45619", "7203", "2401", r"\udk{", r"\presentedby{",
                  r"\begin{altabstract}", r"\begin{altkeywords}"):
        assert token in dan, f"в сообщении для Докладов нет {token}"
    # библиография Докладов: не более 25 источников, самоцитирование <= 30 %
    first_bib = dan.split(r"\begin{thebibliography}")[1]
    items = re.findall(r"\\bibitem\{(\w+)\}", first_bib.split(r"\end{thebibliography}")[0])
    assert 0 < len(items) <= 25
    assert sum(k.startswith("Ivanov") for k in items) / len(items) <= 0.3
