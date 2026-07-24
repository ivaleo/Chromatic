# chromatic — единый фасад над бэкендами voronoi4d и combigeo

Над-проект, дающий один python-API для поиска периодических раскрасок ℝⁿ
поверх двух независимых движков:

- **`voronoi4d`** — эталонная python-реализация (размерность 4);
- **`combigeo`** — быстрое C++ ядро через pybind11 (размерности 2…6).

Бэкенд выбирается **явно** по имени — без автоопределения и скрытых фолбэков.

## Установка

`chromatic` не тянет бэкенды как жёсткие зависимости — установите нужные:

```bash
pip install -e .                       # сам фасад
pip install -e ../voronoi              # бэкенд voronoi4d (numpy/scipy/sympy)
pip install ../combigeo                # бэкенд combigeo (нужен компилятор C++)
```

Бэкенды `voronoi4d` и `combigeo` — не пакеты с PyPI, а **соседние проекты в
монорепозитории Chromatic**. При клонировании монорепо они лежат рядом с этим
каталогом, поэтому и ставятся по относительному пути:

- `voronoi4d` — каталог `../voronoi`: `pip install -e ../voronoi`;
- `combigeo` — каталог `../combigeo`: `pip install ../combigeo` (нужен компилятор
  C++ и pybind11 для сборки ядра).

Проверка:

```python
import chromatic
chromatic.available_backends()         # ['combigeo', 'voronoi4d']
```

## Быстрый старт

```python
import chromatic

D4 = [[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]]

backend = chromatic.get_backend("combigeo")   # явный выбор

cell = backend.build_cell(D4)
print(cell.f_vector)        # [24, 96, 96, 24] — 24-ячейка
print(cell.diameter)        # 2.0

res = backend.find_optimal(D4, 49)
print(res.normalized, res.feasible)   # 1.080123  True
print(res.transition)                 # матрица перехода (HNF)
```

## Кросс-валидация (только размерность 4)

```python
report = chromatic.compare_backends(D4, range(2, 50))
print(report.agree)            # True — оба движка согласны
for d in report.discrepancies:
    print(d)                   # расхождения, если есть
```

combigeo и voronoi4d используют независимые геометрические алгоритмы (GJK против
каскада проекций), поэтому совпадение — сильная взаимная проверка корректности.

## API

Полное описание — в [docs/USAGE.md](docs/USAGE.md). Кратко:

- `get_backend(name)` → `Backend` (`"voronoi4d"` или `"combigeo"`);
- `available_backends()` → список доступных;
- `Backend.build_cell(basis)` → `Cell`;
- `Backend.cell_distance(point, cell)` → сырое расстояние;
- `Backend.min_color_distance(basis, sub_basis)` → сырое D;
- `Backend.find_optimal(basis, index)` → `OptimalResult`;
- `Backend.find_optimal_range(basis, indices)` → `{index: OptimalResult}`;
- `Backend.lll_reduce(basis, delta=0.75)`, `Backend.shortest_vector(basis)` (combigeo);
- `compare_backends(basis, indices, tol)` → `ComparisonReport`.

Единая терминология: `num_colors` = det(M) = индекс; `diameter` = diam(V₀);
`min_distance` = сырое D; `normalized` = d = D/diam (пригодно при d ≥ 1, поле
`feasible`); `transition` = M (HNF); `witness` = вектор, дающий минимум.

## Тесты

```bash
pip install -e '.[dev]'   # кавычки обязательны в zsh: без них — «no matches found»
python -m pytest
```

Про поведение при отсутствии бэкендов: тесты, которым нужен конкретный бэкенд,
**пропускаются** (`skip`), если он не установлен. Но есть один тест-страж
`test_at_least_one_backend_available` (в `tests/test_compare.py`), который
намеренно **ПАДАЕТ** (`FAILED`), если не установлено *ни одного* бэкенда — чтобы
`pytest` не «зеленел» на пустой установке, где проверять нечего. Так что красный
`FAILED` на голом окружении — это ожидаемо, а не поломка.
