# paper/ — статья об оценках χ(ℝ⁴) ≤ 45, χ(ℝ⁵) ≤ 132, χ(ℝ⁷) ≤ 1029 и χ(ℝ⁹) ≤ 7203

Основной итог исследования. Исходник — [`chi4-45.tex`](chi4-45.tex), собранный
PDF — `chi4-45.pdf`.

## Что внутри

| Файл | Что это |
|---|---|
| `chi4-45.tex` | Каркас статьи: преамбула, титул, `\input` секций, библиография. |
| `sections/*.tex` | Десять секций статьи (`intro`, `method`, `algo`, `main`, `dim57`, `intervals`, `extra`, `dim9-12`, `open`, `screens`). |
| `origin-and-ai.tex` | Раздел о происхождении результатов и роли ИИ. |
| `chi4-45.pdf` | Собранная версия (для чтения без TeX). |
| `figures.py` | Генератор всех иллюстраций из данных `../audit-data`. |
| `fig_*.pdf` | 5 готовых рисунков (method, interval, descent, eisenstein, staircase). |

## Требования к сборке

- **TeX Live** (или MacTeX / TeX Live ≥ 2021) с поддержкой кириллицы:
  преамбула использует `fontenc[T2A]` и `babel[russian]`.
- Для перегенерации рисунков — Python с `numpy` и `matplotlib`
  (см. корневой [`README.md`](../README.md) про общий `.venv`).

## Сборка

```bash
cd paper

# (опционально) перегенерировать рисунки из данных экспериментов:
python figures.py            # читает ../audit-data/*.json → fig_*.pdf

# собрать PDF:
latexmk -pdf chi4-45.tex     # рекомендуется (сам прогонит нужное число раз)
# либо вручную:
#   pdflatex chi4-45.tex && pdflatex chi4-45.tex
```

Библиография встроена в `.tex` (окружение `thebibliography`), отдельного
BibTeX-прогона не требуется.

`figures.py` находит данные и каталог вывода относительно собственного
расположения — работает из любого рабочего каталога, правки путей не нужны.

## Данные, на которых стоит статья

Точные сертификаты и результаты экспериментов — в [`../audit-data/`](../audit-data/README.md).
Ключевые:

- `cert45.json` — точный рациональный сертификат главной теоремы (χ(ℝ⁴) ≤ 45);
- `cert46.json`, `cert48.json` — сертификаты вторичных конструкций;
- `results/metric_deform_a5_132_refined_certificate.json` —
  рациональный сертификат новой оценки χ(ℝ⁵) ≤ 132 и интервала
  `[1,1.01]` (720 точных вершинных систем и 36 точных
  KKT-сертификатов расстояния);
- `results/metric_deform_a5_132_refined_independent_exact_audit.json`
  — независимый точный аудит той же конструкции без Qhull;
- `results/exact_prime_threshold_a5_131_escape_refine_round3.json` и
  `results/active_metric_a5_131_escape_frontier_independent_exact_d1000000.json`
  — полный дискретный фронтир 131 на лучшей сохранённой форме и независимая
  точная отрицательная диагностика (`0.990215962…`, не верхняя оценка);
- `results/metric_deform_e7_1323_certificate.json` — рациональный сертификат
  новой оценки χ(ℝ⁷) ≤ 1323 (39 296 точных вершинных систем и 70 точных
  KKT-сертификатов расстояния);
- `results/a9_orbit_16875.json`, `a9_orbit_17150.json` и
  `e8_neighbor_2400.json` — отрицательные вычислительные экраны ближайших
  индексов в размерностях 8–9 (не сертификаты невозможности);
- `results/portfolio_bridge_d6_342_340_seed6342001.json` и
  `trust_bridge_d6_{342,340}_r{1,2}_p{4,8}_seed*.json` — общая метрика двух
  ядер в ℝ⁶ и локальные projective-Hamming экраны; порог 1 не достигнут,
  оценка 343 не изменяется;
- `results/fixed7_d6_336_*.json`,
  `active_d6_336_44_alt_stage1.json`,
  `ltype_{cross,refine}*_d6_336_*.json` и
  `sdp_d6_336_*.json` — арифметическое продолжение \(343\to336\), точные
  переходы через L-type-стены и последовательный SDP/cutting-plane поиск;
  лучший полный численный фронтир равен `0.984244122033`, но это не
  допустимая раскраска и оценка 343 не изменяется;
- `results/highs_outer_d6_336_*.json`,
  `discrete_highs_d6_336_cycle2_archive.json`,
  `highs_kernel_{race,refine}_d6_336_*.json` — внешняя PSD-аппроксимация
  собственными векторами через HiGHS, 95 L-type-ветвей на трёх глубинах и
  турнир 169 HNF-различных ядер; `0.984244122202` численно неотличимо от
  прежнего барьера и не является новой оценкой;
- `results/highs_index_refine_d6_*.json` и
  `periodic_lift_d6_336_p7_all12.json` — собственные формы для более низких
  и нециклических индексов (лучший вторичный остров `333`: `0.961087668`) и
  не-косетный периодический подъём; все 72 первых межслойных назначения
  невозможны уже из-за пустой строки или столбца;
- `d6_torus_{column_generation,multilift,period_portfolio,index_sweep}.py`
  и `torus_{column,multilift,period_portfolio,index_sweep}_d6_*.json` —
  полный граф Кэли конечного тора, HiGHS set-cover/MIP и CP-SAT pricing:
  простые и составные подъёмы ядра 343, 16903 независимых периодов
  порядков 686--2401, 1798 гладких периодов классической формы и 1151
  период деформированной формы; требуемого независимого множества не найдено,
  но портфели не являются полным запретом всех периодов;
- `d6_torus_{prime_exhaustive,fourier_lp}.py`,
  `torus_prime_exact_d6_343_p{3,5,7}_bitset.json` и
  `torus_fourier_d6_343_*.json` — исчерпывающий точный экран всех 23941
  простых характеров при \(p=2,3,5,7\) и независимая Fourier-редукция
  к LP HiGHS: \(\vartheta=p\) на 56 профильных представителях, а
  \(\vartheta'=p\) следует вместе с точным \(\alpha=p\) и напрямую
  проверено для трёх самых разреженных графов; улучшения отношения 343 нет;
- `d6_{cyclic_hole_search,cyclic_highs_neighborhood,affine_metric_opt,affine_active_refine,affine_sdp_hybrid,cyclic_block_search}.py`
  и `cyclic_hole_d6_*.json`, `affine_*_d6_342_*.json`,
  `cyclic_block_d6_342_*.json` — 342-цветные аффинные пары и
  последовательные блоки циклического фактора; лучшие отношения
  `0.9212415444` и `0.9002398182`, полные MIP имеют статус `UNKNOWN`;
- `results/fixed49_d6_294_exact.json` и
  `fixed7_d6_nonmonotonic_*.json` — исчерпывающий экран ограниченного
  source-preserving семейства 294 и многорестартовый скан индексов
  329, 322, 315, 308, 301, 280;
- рисунки читают `campaign_a/c.json`, `n2_4d_frontier.json`, `n5_cascade.json`,
  `n4_push46.json`, `n6_push45.json`, `r5_push48.json`, а также данные кампании
  «ниже 45»: `n8_cma44_ladder.json`, `n7_push44.json`, `n10_push44.json`.
