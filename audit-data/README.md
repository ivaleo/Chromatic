# audit-data/ — скрипты воспроизведения и сырые результаты

Рабочий архив вычислительной части проекта: генераторы данных, точные
сертификаты и json-выгрузки экспериментов. Файлы именованы по сессиям работы —
это исследовательская «кухня», а не отполированная библиотека. Чтобы
сориентироваться, ниже отмечено, **что является опорным** (на этом стоят статья
и её рисунки) и **что — черновое** (разведочные прогоны, оставлены для полноты).

Для больших размерностей (n ≥ 5) есть отдельный, курированный и отдельно
документированный набор: [`README-dim5-9.md`](README-dim5-9.md) — начинать
лучше с него.

## Где что лежит

| Каталог | Что |
|---|---|
| `chromatic_research/core/` | Модули, которыми пользуются несколько кампаний (33 модуля): методы, ядра АБПР (`abpr_kernels`), геометрия ячейки (`voronoi_geometry`), факторгруппы (`torus_quotient`). От `campaigns/` не зависят — это проверяет `tests/test_core_layering.py`. |
| `chromatic_research/campaigns/` | Отдельные кампании (142 модуля). |
| `chromatic_research/enumerators/` | Исходники C++-переборщиков (собираются локально). |
| `tests/` | Тесты; `make test` гоняет их вместе с остальными. |
| `results/` | Опорные данные (205 файлов): на них ссылаются статья, README и тесты. |
| `runs/` | Сырые выгрузки прогонов (542 файла, gzip); см. [`runs/MANIFEST.md`](runs/MANIFEST.md). |

## Запуск

Пакет ставится один раз: `make install` (или `pip install -e audit-data`).
Дальше любая кампания запускается из любого каталога:

```bash
python -m chromatic_research.campaigns.<имя> [аргументы]
```

Общие модули из `core/` импортируются, а не запускаются. Абсолютных путей в
коде нет: расположение артефактов даёт `chromatic_research.paths`.

## Опорные файлы (нужны статье — НЕ удалять)

**Точные сертификаты (рациональная арифметика):**

| Файл | Роль |
|---|---|
| `cert43_eisenstein.json` | Сертификат **главной теоремы**: χ(ℝ⁴) ≤ 43 (ℓ₀ = 1,00411, эйзенштейнова решётка). |
| `dim4_k43_verify.json` | Независимый точный пересчёт того же: diam², D², d² в дробях (ККТ-сертификаты, без объемлющего многогранника). |
| `cert43.json` | Сертификат первой найденной точки индекса 43 (ℓ₀ = 1,003714). |
| `cert45.json` | Сертификат конструкции k=45 (более широкий интервал, ℓ ≤ 1,015). |
| `cert46.json` | Сертификат конструкции k=46. |
| `cert48.json` | Сертификат более широкого интервала (k=48, ℓ≤1.0396). Генератор — `campaigns/cert_generic.py` (параметры по умолчанию). |
| `cert_generic.py` | Параметрический генератор/проверщик сертификатов. |
| `chromatic_research/campaigns/dim4_k43_verify.py` | Независимый точный верификатор раскрасок ℝ⁴ (ничего не берёт у voronoi4d/combigeo). |
| `results/r4_k43_eisenstein_rational.json` | Заголовочная решётка ℝ⁴/43: рациональная форма Грама и переход. |
| `results/dim4_below43_screen.json` | Экран **[Э]** k = 31…42: два симметричных семейства × исчерпывающий перебор всех подрешёток. |
| `results/dim4_eisenstein_scan.json` | Лестница эйзенштейновых норменных индексов 7…43 (порог пересекается ровно на 43). |
| `results/dim4_k43_optimum.json` | Оптимум семейства: четыре орбиты в связке, система трёх многочленов, 80 знаков. |
| `results/metric_deform_a5_132_refined_certificate.json` | Рациональный сертификат **χ(ℝ⁵) ≤ 132** (ℓ = 101/100); независимый аудит без Qhull — `results/metric_deform_a5_132_refined_independent_exact_audit.json`. |
| `results/dim7_1029_exact.json` | Точный рациональный сертификат **χ(ℝ⁷) ≤ 1029**: 254 фасеты, 30 368 вершин, полнота по 1-скелету. Генератор — `campaigns/dim7_1029_exact.py`. |
| `results/dim9_7203_exact.json` | Точный рациональный сертификат **χ(ℝ⁹) ≤ 7203**: 752 фасеты, 1 654 230 вершин (4590 непростых), полнота по 1-скелету. Генератор — `campaigns/dim9_7203_exact.py`. |
| `results/metric_deform_e7_1323_certificate.json` | Рациональный сертификат **χ(ℝ⁷) ≤ 1323**; генератор-проверщик — `chromatic_research/campaigns/verify_metric_candidate.py`. |
| `results/dim7_1029_layer_cert.json` | **Независимая перепроверка χ(ℝ⁷) ≤ 1029** точным слоёным сертификатом: работает в шестимерной базе, семимерную ячейку не строит. 167 кусков (103 доказано пустыми), max φ = 103310189182571717/59047277850000000 < 7/4. Генератор — `campaigns/dim7_1029_layer_cert.py`. |
| `results/shell_floor.json` | **Оболочечные полы** для A₃*, E₆*, E₈, K₁₂, Λ₂₄: на A₃* и E₈ пол совпал с рекордом (15 и 2401 неулучшаемы). Генератор — `campaigns/shell_floor.py`. |
| `results/dim6_e6star_floor.json` | Тот же пол для E₆* с **повекторными** свидетелями: 936 точных рациональных точек `x ∈ V₀`, закрывающих оболочку \|v\|² = 8 (запасной, более грубый путь к тому же 305). |
| `results/dim6_eisenstein_337_screen.json`, `results/dim6_eisenstein_337_calibration.json` | Исчерпывающий перебор всех 227 814 ℤ[ω]-подмодулей индекса 337 в ℝ⁶ и калибровка экрана по простым индексам (порог берётся лишь при 541 против 343 у подобной подрешётки). |
| `results/dim6_cyclotomic7_337.json` | Закрытое циклотомическое семейство ℤ[ζ₇] при k = 337: max d = 0.8278 по всему двупараметрическому семейству. |

**Зависимости рисунков статьи** (`../paper/figures.py` читает их напрямую):
`campaign_a.json`, `campaign_c.json`, `n2_4d_frontier.json`, `n5_cascade.json`,
`n4_push46.json`, `n6_push45.json`, `r5_push48.json`, `n8_cma44_ladder.json`,
`n7_push44.json`, `n10_push44.json`.

## Разведочные скрипты (по темам)

Не опорные для итоговых результатов; сохранены как воспроизводимый след работы.

- **Симметрийное сужение поиска в ℝ⁴ (23.08.2026):** `core/eisenstein4.py`
  (эрмитовы формы над ℤ[ω] и ℤ[i], подмодули, быстрый оценщик d),
  `campaigns/dim4_eisenstein_scan.py` (скан эйзенштейнова семейства),
  `campaigns/dim4_symmetry_scan.py` (гауссово и циклотомические семейства),
  `campaigns/dim4_below43_screen.py` (экран с исчерпывающим перебором
  подрешёток), `campaigns/dim4_below43_general.py` (поиск по всем формам с
  посевом из симметричной области), `campaigns/dim4_symmetry_atlas.py`
  (12 классов ℤ[S]-модулей ранга 4), `campaigns/dim4_k43_optimum.py`
  (алгебраическая модель оптимума: четыре орбиты в связке → три уравнения),
  `campaigns/dim4_tiling_below43.py` (мозаичная атака на k = 42: постадийная
  полировка решёточной точки — веса, затем узлы; улучшения ноль),
  `campaigns/dim4_glued_focus.py` (прицельный плотный скан одного класса
  симметрии по одному индексу).

- **Оболочечный пол и точный слоёный сертификат (24.08.2026):**
  `core/exact_dd.py` (точное перечисление вершин рационального многогранника,
  алгоритм двойного описания в дробях), `core/exact_layer_cert.py` (точная
  верхняя оценка радиуса покрытия ламинированной решётки: мажоранта φ, куски
  общего измельчения, CEGAR по неравенствам),
  `campaigns/dim7_1029_layer_cert.py` (независимая перепроверка 1029),
  `campaigns/shell_floor.py` (полы для пяти классических родителей),
  `campaigns/dim6_e6star_floor.py` (тот же пол для E₆* повекторными
  свидетелями), `campaigns/dim6_eisenstein_337.py` и
  `campaigns/dim6_eisenstein_calibrate.py` (исчерпывающий экран подмодулей и
  его калибровка), `campaigns/dim6_cyclotomic7_337.py` (циклотомическая ветвь).
  Тесты: `tests/test_exact_dd.py`, `tests/test_exact_layer_cert.py`,
  `tests/test_shell_floor.py`, `tests/test_dim6_e6star_floor.py`,
  `tests/test_dim6_eisenstein_337.py`, `tests/test_dim6_cyclotomic7.py`.

- **Верификация констант ABPR/Иванова:** `verify_a5s.py`, `verify_e6s.py`,
  `verify_e8.py`, `verify2.py` (+ `verify2_results.json`), `sweep.py`
  (+ `sweep_results.json`).
- **Кампании 4D/3D:** `campaign_a..d.py` (+ json), `campaign_b_witnesses.json`.
- **Спуск χ(ℝ⁴) 49→45:** `n4_push46.py`, `n6_push45.py`, `n7_push44.py`,
  `n5_cascade.py`, `n5p_cascade.py`, `joint.py`, `joint_optimize.py`,
  `r1_r3_refine.py`, `r2_cone48.py`, `r5_push48.py` (+ соответствующие json).
- **Кампания «ниже 45» (27.07.2026, барьер k=31..44):** `n8_cma44_ladder.py`
  (CMA-ES по формам Грама: атака k=44 восемью инстансами × 2200 точных оценок +
  лестница k=43..31; результат `n8_cma44_ladder.json` — все max d < 1, данные
  рис. спуска), `n9_bk4d.py` (+ json; диагностика best_killed: ровно 1
  неотделимый запрещённый вектор при k=36..44 на всех формах — сигнатура
  жёсткого препятствия, как в 5D/6D), `n10_push44.py` (+ json; NM-дожим k=44 от
  CMA-чемпиона по рецепту, пробившему 45).
- **3D-интервалы и Q-поиск:** `o1_widths4d.py`, `o2_r3.py`, `n1_r3_full.py`,
  `qsearch.py`, `q_frontier.py` (порт `../articles/Qpoisk.c`).
- **5D/6D — ранние черновики** (вытеснены `README-dim5-9.md`, оставлены для истории):
  `csp_campaign5d.py`, `csp_sweep5d.py`, `core/cyclic_csp.py`, `core/general_csp.py`,
  `minconf_csp.py`, `mc_attack5d.py`, `probe5d.py`, `smart_sub.py`, `dim6.py`,
  `r4_beat343.py`, `cma_form.py`, `n3_5d_probe.py`.
- **Экраны 8D/9D после результата 1323:** `results/prime_screen_e8_2400.json`,
  `prime_weighted_e8_2400.json`, `metric_deform_e8_2400.json`,
  `e8_neighbor_2400.json`, `a9_orbit_16875.json`, `a9_orbit_17150.json`;
  архивы `lazy_prime_a9_16875`, `lazy_prime_a9_16384`, `lazy_cxx_a9_17150` —
  в `runs/`. Нового рекорда не получено;
  полный разбор и следующий механизм — в
  `README-dim5-9.md` и `NEXT_MECHANISM.md`.

## Как воспроизвести

Все скрипты рассчитаны на общий `.venv` из корня репозитория (numpy/scipy/sympy;
для 5D/6D — собранный `combigeo`). См. корневой [`README.md`](../README.md).
Примеры:

```bash
python -m chromatic_research.campaigns.verify_main_results   # пять заголовочных утверждений (--full: полные верификаторы)
python -m chromatic_research.campaigns.cert_generic   # перепроверит сертификат k=48 (cert48.json)
make figures                                          # перестроит рисунки статьи
```
