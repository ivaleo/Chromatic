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


def test_intro_headline_carries_only_proven_bounds():
    """Заголовочные величины = лучшие ДОКАЗАННЫЕ оценки, а не численные.

    Замечание рецензента к версии 6: заголовок подавал 1029/7203/28812 как
    доказанные, тогда как шкала статусов объявляет [Ч] не имеющим
    доказательной силы. Тест закрепляет новую границу: в таблице известных
    оценок и в заголовке стоят 45/132/1323/9604/45619 (статусы [Т]/[С]), а
    численные значения присутствуют только с явной меткой.
    """
    MAIN = (ROOT / "paper" / "chi4-45.tex").read_text()
    proven = ("45", "132", "1323", "9604", "45619")
    numeric = ("1029", "7203", "28812")
    # таблица §1.2: полужирным — доказанные, численные без \mathbf
    for k in proven:
        assert rf"\mathbf{{{k}}}" in INTRO, f"{k} не выделено в таблице §1.2"
    for k in numeric:
        assert rf"\mathbf{{{k}}}" not in INTRO, (
            f"{k} выделено как доказанная оценка — статус [Ч]")
    # заголовок статьи: пять доказанных величин и ни одной численной
    title = MAIN[MAIN.index(r"\title{"):MAIN.index(r"\author{")]
    for k in proven:
        assert rf"\le{k}$" in title, f"{k} отсутствует в заголовке"
    for k in numeric:
        assert rf"\le{k}$" not in title, f"{k} (статус [Ч]) стоит в заголовке"
    # каждая численная оценка сопровождается меткой в тексте
    assert r"\stN" in INTRO and "повышенной пробы" in INTRO


def test_proven_r9_step_is_documented():
    """χ(R^9) <= 9604 доказано продуктовым исчислением: 6/7 + 1/9 = 61/63."""
    assert r"\label{cor:dim9}" in PRODUCT
    assert "61/63" in PRODUCT and r"\sqrt{63/61}" in PRODUCT
    assert "9604" in README and "9604" in RESULTS


def test_interval_certificates_match_paper_claims():
    """Теоремы 2 и 3: окна, счётчики векторов и интервалы в артефактах.

    Замечание №3 к версии 5: текст статьи говорил 38/72 вектора, артефакты
    содержали 36/70 (окно 4R вместо 2(1+ell)R) и у ℝ⁷ не было
    certified_interval. Тест закрепляет согласованность после перегенерации.
    """
    dim57 = " ".join((ROOT / "paper" / "sections" / "dim57.tex").read_text().split())
    for name, count, ell, extra in (
        ("metric_deform_a5_132_refined_certificate.json", 38, "101/100",
         [-2, 2, -2, 2, 2]),
        ("metric_deform_e7_1323_certificate.json", 72, "1007/1000",
         [2, -1, 0, 0, 0, 0, 1]),
    ):
        cert = json.loads((RESULTS_DIR / name).read_text())
        sv = cert["short_vector_certificate"]
        assert sv["exact_vector_count"] == count == sv["cpp_vector_count"]
        pairs = {tuple(p) for p in sv["beyond_4R_pairs"]}
        assert tuple(extra) in pairs and tuple(-x for x in extra) in pairs
        ci = cert["certified_interval"]
        assert ci["valid"] and ci["upper_endpoint"] == ell
        # каждый вектор окна обязан иметь KKT-сертификат в файле
        assert len(cert["separation"]["all_projection_certificates"]) == count
        assert f"ровно ${count}$ ненулевых вектор" in dim57


def test_dim3_certificate_matches_paper():
    """ℝ³/15: сертификат ℓ ≤ 102659/100000 и вырожденный Кулсон при α=1/3."""
    cert = json.loads((RESULTS_DIR / "dim3_k15_certificate.json").read_text())
    assert cert["coulson_bcc"]["width_squared"] == "1/1"
    assert cert["coulson_bcc"]["degenerate_interval"]
    opt = cert["certified_optimum"]
    assert opt["alpha"] == "3137/10000"
    assert opt["width_squared"] == "121967690/115730769"
    assert opt["certified_interval"]["upper_endpoint"] == "102659/100000"
    intervals = (ROOT / "paper" / "sections" / "intervals.tex").read_text()
    for token in ("121967690/115730769", "102659", "3137/10000",
                  "14\\alpha^3-3\\alpha^2-10\\alpha+3"):
        assert token in intervals, f"нет {token} в intervals.tex"
    assert "102659" in RESULTS and "102659" in README
    # прежняя точка 16/51 осталась в артефакте как история
    assert cert["earlier_rational_point"]["width_squared"] == "1586/1505"


def test_alpha_star_rational_bound_is_exact():
    """d(α*) > 102659/100000 доказывается в чистых дробях, без float.

    Замечание к версии 6: α* задан кубикой плюс рациональным изолирующим
    окном, ширина в нём точна, а граница получается из монотонности
    связывающей ветви и одного точного неравенства. Тест пересчитывает это
    независимо от кода кампании.
    """
    from fractions import Fraction as F
    star = json.loads(
        (RESULTS_DIR / "dim3_k15_certificate.json").read_text())["alpha_star"]
    lo, hi = (F(t) for t in star["isolating_interval"])
    p_ = lambda x: 14 * x**3 - 3 * x**2 - 10 * x + 3
    assert p_(lo) > 0 > p_(hi)                       # корень в окне есть
    assert 84 * lo - 6 > 0                           # p'' > 0 на окне
    assert 42 * hi**2 - 6 * hi - 10 < 0              # p' < 0 в правом конце
    r_a = lambda x: (4 * x**2 + 3 * x + 1) / ((x + 1) * (2 - x))
    ell = F(star["ell"])
    margin = r_a(lo) - ell**2
    assert margin > 0 and margin == F(star["rational_margin"])
    # то же число приведено в записке рецензента
    assert margin == F(38881620997802732729, 2215290558757910000000000)
    assert ell == F(102659, 100000)


def test_coulson_improved_colouring_is_alpha_4_13():
    """Улучшенная 15-раскраска Кулсона — точка α=4/13 семейства G(α).

    Замечание к версии 6: статья утверждала, что у Кулсона запрещено лишь
    единственное расстояние. На деле в конце его работы стоит интервал
    (sqrt 22, sqrt(389/17)), то есть (1, sqrt(389/374)) ~ (1, 1.0198563).
    Тест закрепляет: эта точка сертифицирована, её масштаб 13 воспроизводит
    опубликованные числа, и документы больше не говорят «вместо
    единственного расстояния».
    """
    from fractions import Fraction
    cert = json.loads((RESULTS_DIR / "dim3_k15_certificate.json").read_text())
    imp = cert["coulson_improved"]
    assert imp["alpha"] == "4/13"
    assert imp["width_squared"] == "389/374"
    assert Fraction(imp["diameter_squared"]) * 13 == 22
    assert Fraction(imp["minimum_distance_squared"]) * 13 == Fraction(389, 17)
    assert imp["certified_interval"]["valid"]
    # ширина Кулсона строго между вырожденной единицей и нашим сертификатом
    assert (Fraction(1) < Fraction(imp["width_squared"])
            < Fraction(cert["certified_optimum"]["width_squared"]))
    intervals = (ROOT / "paper" / "sections" / "intervals.tex").read_text()
    for token in ("4/13", "389/374", "389/17", r"\sqrt{22}"):
        assert token in intervals, f"нет {token} в intervals.tex"
    assert "вместо единственного" not in intervals
    assert "4/13" in README and "4/13" in RESULTS


def test_eisenstein_identity_is_a_theorem():
    """Тождество (10) доказано целиком: обе половины — теоремы, вопрос снят.

    До версии 7 доказана была лишь нижняя половина (prop:planar), равенство
    стояло в списке открытых вопросов. Верхнюю половину (prop:eisup, точка
    q = (2w + omega w)/3 в ячейке) прислала Н. Глушкова при рецензировании
    версии 6; вместе они дают thm:eis. Тест закрепляет, что все производные
    места переписаны и что численный контроль сходится.

    Атрибуция намеренно вынесена из текста статьи в журнал ревью до решения
    вопроса о соавторстве (см. test_eisenstein_proof_attribution_is_recorded).
    """
    extra = SECTIONS["extra.tex"]
    openq = SECTIONS["open.tex"]

    for label in (r"\label{lem:cellpoint}", r"\label{prop:eisup}",
                  r"\label{thm:eis}", r"\label{cor:hexcell}", r"\label{rem:a2tri}"):
        assert label in extra, f"нет {label} в extra.tex"
    assert r"\label{cor:Uexact}" in PRODUCT, "нет cor:Uexact в product.tex"

    # старые формулировки «доказана только половина» должны исчезнуть
    assert "во всех проверенных случаях" not in extra
    assert "открытым остаётся только вопрос о равенстве" not in extra
    assert "остаётся доказать\nравенство" not in openq
    assert "половина тождества" not in INTRO

    # численный контроль
    data = json.loads((RESULTS_DIR / "eisenstein_identity_checks.json").read_text())
    assert data["all_hold"] is True
    assert len(data["identity"]) >= 8 and len(data["alpha_ladder"]) >= 8
    assert any("random skew" in r["lattice"] for r in data["identity"]), \
        "нужны случайные косые решётки вне списка n = 2,4,6,8"
    for row in data["identity"]:
        assert abs(row["relative_error"]) < 1e-7, row["lattice"]
        assert row["dist_q_to_V0"] <= 1e-9, row["lattice"]
    for row in data["alpha_ladder"]:
        assert row["worst_gap"] <= 1e-7, row["lattice"]
    for row in data["a2_triangle"]:
        assert row["has_a2_triangle"] and row["holds"], row["lattice"]
    assert data["hurwitz_24cell_vertices"]["holds"] is True


def test_eisenstein_proof_attribution_is_recorded():
    """Авторство prop:eisup не должно потеряться, пока его нет в статье.

    По решению автора (21.08.2026) атрибуция Н. Глушковой временно убрана из
    текста статьи — вопрос о соавторстве решается перед подачей на arXiv.
    Значит, единственная запись живёт в журнале ревью и в хронике; тест
    следит, чтобы её не вычистили заодно, и чтобы статья тем временем не
    приписывала доказательство автору явным образом.
    """
    notes = (ROOT / "journal" / "REVIEW-notes-v6.md").read_text()
    assert "Глушков" in notes and "prop:eisup" in notes, \
        "запись об авторстве prop:eisup исчезла из journal/REVIEW-notes-v6.md"
    assert "соавторств" in notes, "в журнале нет пометки о нерешённом соавторстве"
    assert "Глушков" in RESULTS and "prop:eisup" in RESULTS, \
        "запись об авторстве prop:eisup исчезла из RESULTS.md"

    # в статье атрибуции нет — но и присвоения тоже
    for name in ("extra.tex", "intro.tex", "summary-en.tex"):
        text = SECTIONS[name]
        assert "Глушков" not in text and "Glushkova" not in text, \
            f"атрибуция вернулась в {name} — сверьтесь с решением по соавторству"
    paper = (ROOT / "paper" / "chi4-45.tex").read_text()
    assert r"\ref{prop:eisup}" not in paper, "атрибуция вернулась в благодарности"


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
