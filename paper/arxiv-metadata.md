# Метаданные подачи на arXiv

Черновик полей формы подачи (утверждает автор перед отправкой).

## Title

```
New upper bounds for the chromatic numbers of Euclidean spaces:
chi(R^4) <= 45, chi(R^5) <= 132, chi(R^7) <= 1029, chi(R^9) <= 7203, chi(R^10) <= 28812
```

## Authors

```
Leonid L. Ivanov
```

## Abstract

(тот же текст, что в английском абстракте PDF; лимит arXiv — 1920 знаков)

```
A coloring of Euclidean space is proper for the forbidden distance segment
[1,l] if no two points of the same color realize a distance in [1,l]; the
minimum number of colors is denoted chi(R^n,[1,l]), and l=1 recovers the
classical chromatic number chi(R^n) of the Nelson--Hadwiger problem. We lower
the known upper bounds in five dimensions: chi(R^4) <= 45, chi(R^5) <= 132,
chi(R^7) <= 1029, chi(R^9) <= 7203, chi(R^10) <= 28812, against the
previously known 49, 140, 1372, 17253, and 3^10 = 59049. In particular, this
refutes the expectation of Arman, Bondarenko, Prymak, and Radchenko that 49
and 140 are optimal among lattice colorings of R^4 and R^5. Four independent
mechanisms drive the improvements: (i) lattices in general position, found by
optimizing the metric itself, give 45, 132, and 1323, with all key
inequalities verified in exact rational arithmetic; (ii) lamination ---
lifting a coloring of R^{n-1} in layers --- combined with a piecewise
certificate of the diameter yields the chains 17253 -> 9604 -> 7203 in R^9
and 1372 -> 1323 -> 1029 in R^7; (iii) a planar theorem
D((3+w)L) >= sqrt(7/3) lambda_1 makes the interval width for
chi(R^24) <= 7^12 rigorous; (iv) a product calculus of widths reduces
admissibility of an orthogonal product to the single inequality
sum_i 1/d_i^2 <= 1, giving chi(R^10) <= 2401*19 = 45619, improved to 28812
by layered lamination. Two rigorous index screens (Minkowski-volume and
inradius) trace the limits of the method; negative results are reported
alongside positive ones. Every claim carries an explicit status label
(theorem / exact rational certificate / numerical); all code, exact
certificates, and data are open.
```

## Categories

- **Primary:** math.MG (Metric Geometry)
- **Cross-list:** math.CO (Combinatorics)

Обоснование: содержание — решётки, ячейки Вороного, радиусы покрытия, ширины
запрещённых интервалов (метрическая геометрия); хроматическая постановка —
комбинаторика. Ключевой ориентир АБПР (arXiv:2112.13438) размещён так же.
Если endorsement для math.MG задержится — допустимо поменять местами
(primary math.CO, cross-list math.MG).

## MSC classes

```
52C10 (Primary); 05C15, 52C07, 11H31 (Secondary)
```

## Comments

```
52 pages, 9 figures, in Russian with an extended English summary.
Code, exact certificates and data: https://github.com/ivaleo/Chromatic
```

(число страниц сверить с финальным PDF перед отправкой)

## License

**arXiv non-exclusive license v1.0** (минимальная; сохраняет полную свободу
для последующей журнальной подачи — CC-лицензию можно выбрать позже, отозвать
нельзя).

## Чек-лист подачи

1. Аккаунт arXiv: leo.ivanov@gmail.com (зарегистрирован 17.08.2026,
   Unaffiliated, group math).
2. Endorsement: у аккаунта нет прежних arXiv-работ — при первой подаче arXiv,
   вероятно, попросит endorsement и выдаст код; код отправить любому
   arXiv-автору с правом эндорсинга в math.MG/math.CO.
3. Репозиторий GitHub должен быть публичным ДО отправки (ссылка в статье).
4. Тарболл: `chi4-45.tex`, `origin-and-ai.tex`, `sections/*.tex` (13 файлов),
   9 × `fig_*.pdf`. Без `.aux/.toc/.log/.out/.bbl` (библиография встроена).
   Контрольная сборка из тарболла в чистом каталоге перед загрузкой.
5. После загрузки сверить PDF, собранный arXiv, с локальным постранично.
6. Не-английская статья может уйти на модерацию (1–3 дня) — это штатно.
7. После анонса: вписать arXiv ID в README/RESULTS отдельным коммитом.
