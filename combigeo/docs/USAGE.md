# combigeo — руководство по использованию

C++17 ядро для периодических раскрасок ℝⁿ (надёжно 2…4, размерность 5
практична, о 6 — см. часть C) плюс python-модуль через pybind11. Без внешних
зависимостей кроме pybind11 для обёртки.

Два способа применения: **C++ библиотека** (CMake) и **python-модуль**.

---

## Часть A. C++ библиотека

### A.1 Сборка

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
ctest --test-dir build            # все тесты (linalg, lll, lattice, polytope, gjk, sublattice, solver)
```

Цели: `combigeo_core` (статическая библиотека), `combigeo` (CLI), `test_*`.

### A.2 CLI

```bash
./build/combigeo D4 49          # решётка D4, 49 цветов
./build/combigeo BCC 2 16       # диапазон индексов 2..16
```

Пресеты решёток: `Z2 A2 Z3 FCC BCC Z4 D4` (других ввод CLI не принимает;
`A2` — гексагональная).
Вывод: диаметр, число обсчитанных подрешёток, расстояние D, нормированное
d = D/diam, матрица перехода (HNF в базисе пользователя) и базис подрешётки
`sub_basis` (объемлющие координаты).

### A.3 Использование из своего C++ кода

```cpp
#include "combigeo/solver.hpp"
#include "combigeo/polytope.hpp"
using namespace combigeo;

Mat basis = {{2,0,0,0},{1,1,0,0},{1,0,1,0},{1,0,0,1}};  // строки = базис

VoronoiCell cell = build_voronoi_cell(Lattice(basis).lll_reduced());
// cell.vertices, cell.facets, cell.f_vector ([24,96,96,24]), cell.diameter

SolveResult r = find_optimal(basis, 49);
// r.index, r.diameter, r.normalized, r.examined,
// r.best.transition (HNF в базисе пользователя), r.best.sub_basis
// (объемлющие координаты), r.best.min_distance, r.best.witness
```

Подключение библиотеки в своём CMake:

```cmake
add_subdirectory(combigeo)
target_link_libraries(your_target PRIVATE combigeo_core)
# заголовки в combigeo/include
```

### A.4 Обзор заголовков (`include/combigeo/`)

| Заголовок | Содержимое |
|---|---|
| `linalg.hpp` | `Vec`/`Mat`, `dot`/`det`/`solve_linear`/`gram_schmidt`, `validated_dim` |
| `lll.hpp` | `lll_reduce(basis, delta)` |
| `lattice.hpp` | `Lattice`: `det`, `lll_reduced`, `vectors_within`, `shortest_vector` |
| `polytope.hpp` | `build_voronoi_cell`, `VoronoiCell`, `Halfspace` |
| `gjk.hpp` | `distance_to_polytope`, `closest_point_on_simplex` |
| `sublattice.hpp` | `SublatticeIterator`, `ordered_factorizations`, `apply_hnf` |
| `solver.hpp` | `find_optimal`, `find_optimal_range`, `min_color_distance` |

Исключения (документированы в заголовках): `std::invalid_argument` (плохой базис,
несовпадение размерностей, `index < 1`), `std::overflow_error`
(`SublatticeIterator::count` при слишком большом индексе).

---

## Часть B. Python-модуль

### B.1 Установка

```bash
pip install .                   # компиляция + установка модуля combigeo (pybind11 подтянется)
```

Альтернатива через cmake:

```bash
cmake -S . -B build -DCOMBIGEO_BUILD_PYTHON=ON
ninja -C build combigeo_py      # модуль combigeo*.so в build/
```

### B.2 Быстрый старт

```python
import combigeo

D4 = [[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]]

cell = combigeo.voronoi_cell(D4)        # размерность 2..6 определяется по базису
cell.f_vector                           # [24, 96, 96, 24]
cell.diameter                           # 2.0
cell.vertices, cell.facets              # вершины и фасеты (Halfspace: normal, offset)

res = combigeo.find_optimal(D4, 49)     # SolveResult
res.normalized                          # 1.080123  (d = D/diam)
res.examined                            # 140050
res.best.transition                     # матрица перехода (HNF в базисе пользователя)
res.best.min_distance                   # сырое D
res.best.witness                        # вектор, дающий минимум
```

> **Важно:** поля `transition`, `sub_basis`, `min_distance`, `witness` лежат
> на `res.best` (объект `SublatticeResult`), а `index`, `diameter`,
> `normalized`, `examined` — на самом `res` (`SolveResult`).
> `transition` — HNF в базисе, переданном пользователем; однозначная
> идентификация подрешётки — `sub_basis` (объемлющие координаты, после LLL).

### B.3 Полный API модуля

```python
combigeo.voronoi_cell(basis, window=3)         -> VoronoiCell (window устарел и игнорируется)
combigeo.distance_to_cell(point, cell)         -> float (сырое расстояние, GJK)
combigeo.min_color_distance(basis, sub_basis)  -> float (сырое D)
combigeo.find_optimal(basis, index, progress_every=0, use_cache=True, threads=0) -> SolveResult
combigeo.find_optimal_range(basis, frm, to, progress_every=0, use_cache=True, threads=0) -> {index: SolveResult}
combigeo.lll_reduce(basis, delta=0.75)         -> базис
combigeo.shortest_vector(basis)                -> вектор
combigeo.count_sublattices(dim, index)         -> int
combigeo.sublattices(dim, index)               -> list[HNF]   (cap 2_000_000)
combigeo.__version__                           -> строка версии ("1.1.0")
```

Диапазонный запуск (kwargs `frm`, `to`, `progress_every`, `use_cache`) строит
ячейку один раз и держит общий кэш расстояний на весь диапазон индексов —
на ~26% быстрее, чем цикл однократных `find_optimal`.

`VoronoiCell`: `.dim`, `.vertices`, `.facets`, `.diameter`, `.f_vector`,
`.vertex_facets`, `.contains(point, tol=1e-9)`.

Валидация: при несовпадении размерностей, нечисловом (NaN/Inf) или вырожденном
базисе бросается исключение (а не падение) — защита работает и в release-сборке;
это касается и `lll_reduce` (рваный вложенный список → исключение). Вход
масштабно нормируется: корректность не зависит от масштаба базиса, все
результаты возвращаются в исходных единицах.

### B.4 Модуль bigdim (большие размерности n ≥ 5)

При n ≥ 5 перечисление вершин ячейки неосуществимо (число сочетаний фасет
C(m, n) растёт экспоненциально); `src/bigdim.cpp` заменяет его безвершинными
примитивами — опорные полупространства, проекция Дейкстры, CSP-переформулировка
поиска подрешётки. Эти функции входят в полный API модуля наравне с B.3.

```python
combigeo.relevant_facets(basis)                          -> list[(lattice_vector, offset)]
combigeo.forbidden_coords(basis, diam, ell=1.0)          -> list[list[int]]
combigeo.min_conflicts(F, e_list, n, max_steps=3000, restarts=20, seed=0)      -> (found, phi, index)
combigeo.min_conflicts_cost(F, e_list, n, max_steps=3000, restarts=20, seed=0) -> (found, best_killed, index)
combigeo.dist_to_halfspaces(point, facets)               -> float
```

- `relevant_facets(basis)` — опорные полупространства (фасеты) ячейки Вороного
  БЕЗ перечисления вершин; список пар `(lattice_vector, offset)`, где
  `lattice_vector` — вектор решётки v, `offset = |v|/2`.
- `forbidden_coords(basis, diam, ell=1.0)` — целочисленные координаты
  запрещённых векторов v (тех, у кого D(v) < ell·diam) в базисе решётки; список
  целых векторов `F` для CSP. `diam` — радиус покрытия (для известных решёток
  точный, иначе оценка сверху). Быстро, любая размерность.
- `min_conflicts(F, e_list, n, ...)` — локальный поиск валидной подрешётки как
  CSP: ищет гомоморфизм φ = (φ₁…φₘ): Λ → Z/e₁×…×Z/eₘ (модули из `e_list`) с
  φ(f)≠0 ∀f∈F. Возвращает `(found, phi, index)`: `phi` — m форм по n
  коэффициентов (`list[list[int]]`), `index` — размер образа = индекс подрешётки
  ker φ (число цветов). `found=False`, если не найдено.
- `min_conflicts_cost(F, e_list, n, ...)` — то же, но возвращает
  `(found, best_killed, index)`: `best_killed` — минимум числа неотделённых
  (не «убитых») f по всем рестартам (0 ⟺ found); гладкий сигнал для оптимизации
  по формам Грама.
- `dist_to_halfspaces(point, facets)` — расстояние от точки до H-многогранника
  (пересечения полупространств `facets` в формате `relevant_facets`) методом
  проекции Дейкстры; 0, если точка внутри.

Радиус покрытия (⇒ `diam`) без вершин считается в python поверх
`relevant_facets` — см. `../audit-data/chromatic_research/core/covrad.py`; все эксперименты
больших размерностей и их результаты — в `../audit-data/`.

---

## Часть C. Проверенные размерности

| Размерность | Статус | Эталон |
|---|---|---|
| 2 | проверена | квадрат Z² |
| 3 | проверена | куб, ромбододекаэдр (FCC), усечённый октаэдр (BCC) |
| 4 | проверена | тессеракт, 24-ячейка (D4), сверка с voronoi4d |
| 5 | практична | Z⁵ (f = [32, 80, 80, 40, 10]), D5, пермутоэдр A₅* [720, 1800, 1560, 540, 62] — секунды |
| 6 | генерические решётки вне досягаемости | переборное построение вершин C(m, n) растёт экспоненциально; (3+ω)E₆* посчитан обходным QP-путём (`../audit-data/`) |

## Часть D. Алгоритм (кратко)

- Ячейка Вороного строится по релевантным векторам классов смежности Λ/2Λ
  (теорема Вороного) — корректно в любой размерности; релевантные векторы
  находятся сертифицированным CVP-перебором по классам (прежнее окно `window`
  устарело и игнорируется), готовая ячейка проходит self-check — соотношение
  Эйлера и центральная симметрия вершин, при сбое — исключение.
- Расстояние точка→ячейка считается алгоритмом **GJK** по вершинам (без
  комбинаторики граней и ошибок ориентации нормалей).
- Подрешётки перебираются в эрмитовой нормальной форме (HNF), индекс = число цветов.
- Поиск оптимума: branch-and-bound с кэшем расстояний; перебор соседей полон
  (D(v) ≥ |v| − diam).

### Параллельность (1.1.0)

`find_optimal(..., threads=0)` и `find_optimal_range(..., threads=0)` перебирают
подрешётки в несколько потоков (страйд по HNF-перебору): `threads=0` — авто
(число ядер минус два), `threads=1` — однопоточно. Результат (максимум d) от
числа потоков не зависит; при нескольких оптимумах с равным d выбранная
подрешётка может отличаться от запуска к запуску. Кэш расстояний — по одному
на поток. Типичный выигрыш на 8 потоках — 2–3× (генерические ячейки), поверх
этого масштабируются мультистартовые кампании на пуле процессов.
