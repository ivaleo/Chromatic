"""Инварианты разделения рукописи на статью 1, статью 2 и краткие версии.

План разделения --- journal/PLAN-2026-09-03-paper-split.md, ревью
предложений --- journal/REVIEW-2026-09-03-split-proposals.md. Полная
рукопись, из которой выделены статьи, с 16.09.2026 в репозитории не хранится
(она в истории git); здесь закрепляется, что статьи не «подтягивают» обратно
то, ради чего их отделяли: численные кандидаты в статью 1, запись χ≤ для
кандидатов в статью 2 (краткие версии проверяет test_paper_short_versions).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A1 = ROOT / "paper" / "article1"
A2 = ROOT / "paper" / "article2"


def _texts(d):
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
            for p in d.rglob("*.tex")}


def test_article1_contains_only_proven_bounds():
    banned = ("28812", "21609", "1323", r"\stN", r"\stM", "[Э]",
              "CMA", "min-conflicts", "Radon")
    hits = [f"{n}: {b}" for n, t in _texts(A1).items() for b in banned if b in t]
    assert not hits, "в статье 1 численные/поисковые следы:\n  " + "\n  ".join(hits)
    main = (A1 / "bounds.tex").read_text(encoding="utf-8")
    title = main[main.index(r"\title{"):main.index(r"\author{")]
    for k in ("43", "132", "1029", "7203", "45619"):
        assert rf"\le{k}$" in title, f"{k} нет в заголовке статьи 1"


def test_article1_prints_the_exact_constructions():
    """Конструкции 1029 и 7203 выписаны явно (ревью П2), а не только в JSON."""
    import json
    import pytest
    for stem, tex in (("dim7_1029_exact", "mat1029"),
                      ("dim9_7203_exact", "mat7203")):
        path = ROOT / "audit-data" / "results" / f"{stem}.json"
        if not path.exists():
            pytest.skip(f"нет артефакта {stem}.json")
        data = json.loads(path.read_text())
        mat = (A1 / "sections" / f"{tex}.tex").read_text(encoding="utf-8")
        for row in data["integer_gram"]:
            assert "&".join(str(x) for x in row) in mat, f"{tex}: строка G {row}"
        for row in data["sublattice_columns"]:
            assert "&".join(str(x) for x in row) in mat, f"{tex}: строка C {row}"


def test_article2_never_states_candidates_as_bounds():
    pat = re.compile(r"\\chi\\?\(\\R\^\{?(9|10)\}?\)\s*\\le\s*(28812|21609)")
    hits = [n for n, t in _texts(A2).items() if pat.search(t)]
    assert not hits, f"кандидат записан как оценка χ≤: {hits}"
    for n, t in _texts(A2).items():
        assert "[Э]" not in t, f"метка экрана в статье 2: {n}"


def test_articles_do_not_cite_removed_full_version():
    """Полная рукопись убрана 16.09.2026: ссылки ведут на репозиторий (Repo)."""
    pat = re.compile(r"\\(?:cite|nocite)(?:\[[^\]]*\])?\{[^}]*\bFull\b[^}]*\}|\\bibitem\{Full\}")
    hits = [n for d in (A1, A2) for n, t in _texts(d).items() if pat.search(t)]
    assert not hits, f"ссылка на удалённую полную версию: {hits}"
