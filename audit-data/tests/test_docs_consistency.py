"""Заголовочные числа документации совпадают с артефактами results/*.json.

Документация (README, RESULTS, статья) трижды расходилась с артефактами за
два дня ревью 08.08.2026: статусы таблицы результатов, лестница ℝ⁹ в README,
несинхронизированный сертификат 28812. Этот тест закрепляет: каждое
заголовочное число берётся из своего json-артефакта и обязано присутствовать
в производных документах; устаревшие паттерны — отсутствовать.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

README = (ROOT / "README.md").read_text()
RESULTS = (ROOT / "RESULTS.md").read_text()
INTRO = (ROOT / "paper" / "sections" / "intro.tex").read_text()
PRODUCT = (ROOT / "paper" / "sections" / "product.tex").read_text()
TILINGS = (ROOT / "paper" / "sections" / "tilings.tex").read_text()
SECTIONS = {
    p.name: p.read_text() for p in (ROOT / "paper" / "sections").glob("*.tex")
}


def test_no_known_typos_in_paper():
    for name, text in SECTIONS.items():
        assert "частасти" not in text, f"опечатка «частасти» в {name}"


def test_dim10_certificate_is_reflected_everywhere():
    rows = json.loads((RESULTS_DIR / "dim10_certificate.json").read_text())
    best = max(
        (r for r in rows if r["index"] == 28812 and r["admissible"]),
        key=lambda r: r["d_certified"],
    )
    # 1.043297... — в статье и хронике до шестого знака, в README до четвёртого
    d6 = f"{best['d_certified']:.6f}"          # 1.043297
    d4 = f"{best['d_certified']:.4f}"          # 1.0433
    assert "28812" in README and d4 in README
    assert "28812" in INTRO
    assert "28812" in PRODUCT and d6.replace(".", "{,}") in PRODUCT
    assert "28812" in RESULTS and d6 in RESULTS


def test_stale_r9_ladder_is_gone():
    # состояние до кусочного сертификата: 9604 → 1.0098 с 12005/14406 в лестнице
    assert "9604 → 1.0098" not in README
    assert "9604` → `1.0098" not in README


def test_semilattice_certificates_are_reflected():
    for n, doc_hits in ((8, (TILINGS, RESULTS, README)),
                        (15, (TILINGS, RESULTS, README))):
        cert = json.loads(
            (RESULTS_DIR / f"semilattice_cert_N{n}.json").read_text())
        assert cert["N"] == n and cert["tiling_exact"]
        d = float(cert["d"])
        d6, d4 = f"{d:.6f}", f"{d:.4f}"
        for doc in doc_hits:
            assert (d6 in doc or d6.replace(".", "{,}") in doc
                    or d4 in doc or d4.replace(".", "{,}") in doc), (
                f"N={n}: ни {d6}, ни {d4} не найдено в документе")


def test_intro_headline_matches_certificate():
    # таблица известных оценок: n=10 должен вести на 28812, а не на 45619
    assert r"\mathbf{28812}" in INTRO
    assert r"\mathbf{45619}" not in INTRO


def test_paper_artifact_paths_exist():
    """Каждый путь audit-data/..., на который ссылается статья, существует.

    Ловит класс ошибок «сертификат переехал, статья ссылается в пустоту»
    (замечания №2 и №23 к версии 5: пути hd-2026-07/ и cert46.json).
    """
    texts = dict(SECTIONS)
    texts["origin-and-ai.tex"] = (ROOT / "paper" / "origin-and-ai.tex").read_text()
    missing = []
    for name, text in texts.items():
        for m in re.finditer(r"\\path\{(audit-data/[^}]+)\}", text):
            rel = m.group(1).replace(r"\_", "_")
            if not (ROOT / rel).exists():
                missing.append(f"{name}: {rel}")
        for m in re.finditer(r"\\texttt\{(results/[^}]*\.json)\}", text):
            rel = m.group(1).replace(r"\_", "_")
            if not (ROOT / "audit-data" / rel).exists():
                missing.append(f"{name}: audit-data/{rel}")
    assert not missing, "битые ссылки на артефакты:\n  " + "\n  ".join(missing)
