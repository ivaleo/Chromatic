"""Инварианты разделения рукописи на статью 1, статью 2 и краткое сообщение.

План разделения --- journal/PLAN-2026-09-03-paper-split.md, ревью
предложений --- journal/REVIEW-split-proposals-2026-09-03.md. Полная
рукопись paper/chi4-43.tex не меняется и остаётся электронным дополнением;
здесь закрепляется, что три производных документа не «подтягивают» обратно
то, ради чего их отделяли: численные кандидаты в статью 1, запись χ≤ для
кандидатов в статью 2, побочные сюжеты в двухстраничную заметку.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A1 = ROOT / "paper" / "article1"
A2 = ROOT / "paper" / "article2"
FULL = ROOT / "paper" / "chi4-43.tex"


def _texts(d):
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
            for p in d.rglob("*.tex")}


def test_article1_contains_only_proven_bounds():
    banned = ("7203", "28812", "21609", "1323", r"\stN", r"\stM", "[Э]",
              "CMA", "min-conflicts", "Radon")
    hits = [f"{n}: {b}" for n, t in _texts(A1).items() for b in banned if b in t]
    assert not hits, "в статье 1 численные/поисковые следы:\n  " + "\n  ".join(hits)
    main = (A1 / "bounds.tex").read_text(encoding="utf-8")
    title = main[main.index(r"\title{"):main.index(r"\author{")]
    for k in ("43", "132", "1029", "9604", "45619"):
        assert rf"\le{k}$" in title, f"{k} нет в заголовке статьи 1"


def test_article1_prints_the_1029_construction():
    """Конструкция 1029 выписана явно (ревью П2), а не только в JSON."""
    import json
    data = json.loads((ROOT / "audit-data" / "results" / "dim7_1029_exact.json")
                      .read_text())
    mat = (A1 / "sections" / "mat1029.tex").read_text(encoding="utf-8")
    for row in data["integer_gram"]:
        assert "&".join(str(x) for x in row) in mat, f"строка G {row} не выписана"
    for row in data["sublattice_columns"]:
        assert "&".join(str(x) for x in row) in mat, f"строка C {row} не выписана"


def test_article2_never_states_candidates_as_bounds():
    pat = re.compile(r"\\chi\\?\(\\R\^\{?(9|10)\}?\)\s*\\le\s*(7203|28812|21609)")
    hits = [n for n, t in _texts(A2).items() if pat.search(t)]
    assert not hits, f"кандидат записан как оценка χ≤: {hits}"
    for n, t in _texts(A2).items():
        assert "[Э]" not in t, f"метка экрана в статье 2: {n}"



def test_full_version_declares_its_role():
    """Полная рукопись объявлена дополнением и знает о трёх документах."""
    main = FULL.read_text(encoding="utf-8")
    assert r"\input{split-preface}" in main
    preface = (ROOT / "paper" / "split-preface.tex").read_text(encoding="utf-8")
    for token in ("paper/article1/bounds.tex", "paper/article2/widths.tex",
                  ):
        assert token in preface, f"предисловие не указывает на {token}"
