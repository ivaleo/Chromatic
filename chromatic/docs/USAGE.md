# chromatic — руководство по использованию

Фасад даёт единый интерфейс над двумя бэкендами. Здесь — полное описание API
с примерами. Терминология и соглашения едины (см. конец документа).

## 1. Установка и выбор бэкенда

```python
import chromatic

chromatic.available_backends()        # какие бэкенды реально установлены
# ['combigeo', 'voronoi4d']

backend = chromatic.get_backend("combigeo")   # явный выбор по имени
```

`get_backend(name)`:
- `name` — строго `"voronoi4d"` или `"combigeo"`;
- бросает `ValueError`, если имя неизвестно;
- бросает `ImportError`, если бэкенд известен, но его пакет не установлен.

Автоопределения нет намеренно: поведение всегда воспроизводимо.

| Бэкенд | Размерности | Когда выбирать |
|---|---|---|
| `combigeo` | 2…6 (надёжно 2…4) | скорость, любые размерности, f-вектор ячейки |
| `voronoi4d` | только 4 | эталон, сверка, чистый python без компиляции |

## 2. Построение ячейки Вороного

```python
D4 = [[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]]
cell = backend.build_cell(D4)

cell.dim          # 4
cell.diameter     # 2.0  — diam(V0) = 2*max|вершина|
cell.vertices     # список вершин (list[list[float]])
cell.facets       # список Facet(normal, offset): x·normal <= offset
cell.f_vector     # [24, 96, 96, 24] у combigeo; None у voronoi4d
```

`Cell` — единое представление; внутренний объект бэкенда лежит в `cell.handle`
(не часть стабильного API).

## 3. Расстояние от точки до ячейки

```python
backend.cell_distance([1.0, 1.0, 0.0, 0.0], cell)   # сырое геометрическое расстояние
```

> Примечание о voronoi4d: у его `dist_to_s` есть ранний выход (параметр
> `early_stop`), но фасад вызывает его с `early_stop=0`, поэтому значения
> точны у обоих бэкендов (для точки внутри ячейки — ровно 0.0).

## 4. Расстояние между одноцветными ячейками

```python
sub = [[4,0,0,0],[2,2,0,0],[2,0,2,0],[2,0,0,2]]     # подрешётка 2*D4
backend.min_color_distance(D4, sub)                  # сырое D = sqrt(2)
```

## 5. Поиск оптимальной подрешётки

```python
res = backend.find_optimal(D4, 49)        # 49 цветов
```

`OptimalResult`:

| Поле | Смысл |
|---|---|
| `num_colors` | число цветов k = det(M) = индекс |
| `diameter` | diam(V₀) |
| `min_distance` | сырое расстояние D между одноцветными ячейками |
| `normalized` | d = D / diam(V₀) |
| `feasible` | свойство: `normalized >= 1` (раскраска пригодна) |
| `transition` | матрица перехода M (HNF в базисе, переданном пользователем), `list[list[int]]` |
| `sub_basis` | базис подрешётки в объемлющих координатах (после LLL); оба бэкенда |
| `witness` | вектор, на котором достигается минимум |
| `examined` | сколько подрешёток обсчитано (combigeo; voronoi4d None) |

Диапазон индексов:

```python
results = backend.find_optimal_range(D4, range(2, 17))   # {index: OptimalResult}
best = max(results.values(), key=lambda r: r.normalized)
```

## 6. Вспомогательные операции

```python
backend.lll_reduce(D4)                 # LLL-приведение базиса
backend.shortest_vector(D4)            # кратчайший вектор (только combigeo)
```

`shortest_vector` у voronoi4d бросает `NotImplementedError`.

## 7. Кросс-валидация бэкендов (размерность 4)

```python
report = chromatic.compare_backends(D4, range(2, 50), tol=1e-6)

report.agree                  # True/False
report.discrepancies          # список Discrepancy(num_colors, field, voronoi4d, combigeo, diff)
report.results_combigeo       # {index: OptimalResult}
report.results_voronoi4d
```

Сверяются диаметр и нормированное d. Оба движка считают без раннего выхода,
поэтому значения сравниваются в пределах `tol` при любом d; отдельно
фиксируется разный вердикт пригодности вне tol-окрестности границы d = 1.

## 8. Замечания о бэкендах (1.1.0)

- `transition` у **обоих** бэкендов — HNF в базисе, переданном пользователем
  (внутренние LLL-рамки наружу не выдаются); матрицы двух бэкендов можно
  сравнивать между собой напрямую.
- `sub_basis` теперь возвращают оба бэкенда (у voronoi4d появился в 1.1.0):
  базис подрешётки в объемлющих координатах после LLL — однозначная
  идентификация подрешётки.
- Нормированное d точно у обоих бэкендов при любом значении: voronoi4d в
  фасаде считает без раннего выхода (`early_stop=0`).
- fpylll-путь voronoi4d: целочисленное масштабирование с `precision=12` по
  умолчанию, `delta` передаётся в бэкенд; при непредставимости базиса —
  фолбэк на чистый python-LLL.
- Поля `transition`/`witness` (и `sub_basis`) могут быть `None` только в
  вырожденном случае «ни одна подрешётка не оценена».

## 9. Соглашения и терминология

| Термин фасада | voronoi4d | combigeo |
|---|---|---|
| `num_colors` / `index` | определитель `det` | `index` |
| `diameter` | `max_len` | `diameter` |
| `min_distance` (сырое D) | `det_dist · max_len` | `best.min_distance` |
| `normalized` (d) | `det_dist` (уже нормирован) | `normalized` |
| `transition` | `det_mat` | `best.transition` |
| `witness` | `det_center` | `best.witness` |

Нормировка единая: **d = D / diam(V₀)**. voronoi4d сворачивает ×2 и /diam внутри
`dist_to_s`, combigeo держит их раздельно — фасад приводит к одному виду.

## 10. Расширение

Свой бэкенд: унаследуйте `chromatic.Backend`, реализуйте абстрактные методы,
пометьте класс `@chromatic.register_backend`. Он появится в `available_backends()`,
если его метод `available()` вернёт True.
