# Метаданные подачи на arXiv

> **Примечание 14.09.2026.** По решению автора на arXiv первой идёт
> **короткая английская версия** `arxiv-en/bounds-en.tex` (8 страниц, все
> пять оценок с данными сертификатов, тождество, продуктовое правило, пол на
> E₈); её метаданные — в `arxiv-en/README.md`. Полная версия и статьи 1–2
> отложены до окончания вычитки. Аккаунт, endorsement и чек-лист ниже
> остаются в силе.

> **Примечание 03.09.2026.** Ниже — метаданные для подачи **полной версии**
> (`chi4-43.tex`, 69 страниц), которая с этой даты играет роль электронного
> дополнения. Для четырёх выделенных документов (`article1/bounds.tex`,
> `article2/widths.tex`) метаданные готовятся отдельно
> после решения авторов о целевых журналах; аннотации каждого документа уже
> лежат в его каркасе. Если на
> arXiv выкладывается только статья 1, её заголовок --- поле Title ниже
> (подзаголовок полной версии о механизмах в него не входит), а abstract берётся из `article1/bounds.tex`
> (он короче лимита 1920 знаков).


Черновик полей формы подачи (утверждает автор перед отправкой).

## Title

```
New upper bounds for the chromatic numbers of Euclidean spaces:
chi(R^4) <= 43, chi(R^5) <= 132, chi(R^7) <= 1029, chi(R^9) <= 7203, chi(R^10) <= 45619
```

## Authors

```
Leonid L. Ivanov, Nadezhda V. Glushkova
```

Порядок — по вкладу (решение от 22.08.2026), не алфавитный; см.
`CONTRIBUTIONS.md`.

**Не заполнено:** аффилиация второго автора — нужна для формы подачи
(в `chi4-43.tex` перед `\author` стоит TODO; e-mail вписан 05.09.2026).

## Abstract

(английский абстракт PDF; текущая фактическая длина — 2254 знака, то есть
перед подачей его необходимо сократить до лимита arXiv в 1920 знаков)

```
A coloring of Euclidean space is proper for the forbidden distance segment
[1,l] if no two points of the same color realize a distance in [1,l]; the
minimum number of colors is denoted chi(R^n,[1,l]), and l=1 recovers the
classical chromatic number chi(R^n) of the Nelson--Hadwiger problem. We lower
the known upper bounds in five dimensions: chi(R^4) <= 43, chi(R^5) <= 132,
chi(R^7) <= 1029, chi(R^9) <= 7203, chi(R^10) <= 45619, against the previously
known 49, 140, 1372, 17253, and 3^10 = 59049; each of the five is proven, by a
theorem or by an exact rational certificate. A piecewise diameter certificate
verified in floating point gives in addition chi(R^10) <= 28812; that one is
numerical, not proven, and stays out of the title. In particular, this refutes the expectation
of Arman, Bondarenko, Prymak, and Radchenko that 49 and 140 are optimal among
lattice colorings of R^4 and R^5. Four independent mechanisms drive the
improvements: (i) lattices in general position, found by optimizing the metric
itself, give 45, 132, and 1323, all verified in exact rational arithmetic;
symmetry-restricted search over lattices with a prescribed finite automorphism
cuts the space of quaternary forms to three parameters and yields the record 43
colors on an Eisenstein lattice;
(ii) lamination --- lifting a coloring of R^{n-1} in layers --- with an
exact enumeration of the 1654230 vertices of the nine-dimensional Voronoi cell
yields the chains 17253 -> 9604 -> 7203 in R^9 and 1372 -> 1323 -> 1029 in R^7,
the last step of each certified exactly;
(iii) the Eisenstein identity D((3+w)L) = sqrt(7/3) lambda_1, proved here for
every Eisenstein lattice, gives the exact width for chi(R^24) <= 7^12; (iv) a product calculus of widths reduces admissibility of an
orthogonal product to the single inequality sum_i 1/d_i^2 <= 1, giving
chi(R^10) <= 2401*19 = 45619, and chi(R^9) <= 2401*4 = 9604 as a shorter but
weaker route to dimension nine. Three rigorous index
screens (Minkowski-volume, inradius, and shell) trace the limits of the method;
negative results are reported too. Every claim carries a status label (theorem
/ exact certificate / numerical); no bound in the title depends on a numerical
one. All code, exact certificates, and data are open.
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
69 pages, 9 figures, in Russian with an extended English summary.
Code, exact certificates and data: https://github.com/ivaleo/Chromatic
```

(69 страниц подтверждены сборкой от 10.09.2026 — версия с точным
сертификатом chi(R^9) <= 7203 и предисловием о роли полной версии)

## License

**arXiv non-exclusive license v1.0** (минимальная; сохраняет полную свободу
для последующей журнальной подачи — CC-лицензию можно выбрать позже, отозвать
нельзя).

## Чек-лист подачи

0. **Второй автор.** E-mail Н. В. Глушковой — glushkova.nv@phystech.edu
   (вписан в `\author{}` всех рукописей 05.09.2026); осталось получить
   аффилиацию и вписать её в поле Authors формы. Подающий автор — Л.Л. Иванов;
   второму автору arXiv пришлёт уведомление об авторстве на указанный адрес.
1. Аккаунт arXiv: leo.ivanov@gmail.com (зарегистрирован 17.08.2026,
   Unaffiliated, group math).
   Default category аккаунта — math.CO; для подачи с primary math.MG менять
   ничего не нужно, категория выбирается в форме подачи.
2. Endorsement: у аккаунта нет прежних arXiv-работ — при первой подаче arXiv,
   вероятно, попросит endorsement и выдаст код; код отправить любому
   arXiv-автору с правом эндорсинга в math.MG/math.CO. Прежние работы автора
   (УМН 2006, Вестник РУДН 2011) на arXiv не размещались, поэтому
   «claim ownership» по ним недоступен и на endorsement не влияет; после
   первой публикации их можно связать с профилем только через arXiv Author ID
   (ORCID), если статьи там появятся.
3. Репозиторий GitHub должен быть публичным ДО отправки (ссылка в статье).
4. Тарболл собран: `chi4-43.tex`, `origin-and-ai.tex`, `split-preface.tex`,
   `sections/*.tex` (13 файлов в `sections/`, всего 16 файлов `.tex`),
   9 × `fig_*.pdf`, без вспомогательных файлов; контрольная сборка в чистом
   каталоге воспроизводит те же 69 страниц без ошибок и неопределённых ссылок.
5. После загрузки сверить PDF, собранный arXiv, с локальным постранично.
6. Не-английская статья может уйти на модерацию (1–3 дня) — это штатно.
7. После анонса: вписать arXiv ID в README/RESULTS отдельным коммитом.
