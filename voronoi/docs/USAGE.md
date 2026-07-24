# voronoi4d — руководство по использованию

Эталонная python-реализация поиска периодических раскрасок **ℝ⁴**
(только размерность 4). Чистый python: numpy, scipy, sympy; опционально fpylll
для быстрого LLL.

## 1. Установка

```bash
pip install -e .              # базовый набор (numpy, scipy, sympy)
pip install -e ".[fast]"      # + fpylll (быстрый LLL на C)
pip install -e ".[plot,dev]"  # + matplotlib, pytest
```

Проверка: `python -c "import voronoi4d; print(voronoi4d.__version__)"`.

## 2. Базовый сценарий

```python
import numpy as np
from voronoi4d import VoronoiPolyhedra, find_optimal, lll_reduce

# решётка задаётся базисом (строки — векторы), обязательно 4×4
grid = np.array([
    [2, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
], dtype=float)

grid = lll_reduce(grid)               # привести базис (рекомендуется)

vor4 = VoronoiPolyhedra(grid)
vor4.build()                          # построить разбиение и центральную ячейку

det_dist, det_center, det_mat = find_optimal(
    range(49, 50),    # диапазон определителей (= числа цветов)
    2,                # limits: УСТАРЕЛ, игнорируется
    grid,
    vor4,
    vor4.max_len,
    threshold=1.0,    # минимальное нормированное расстояние d (фильтр)
)

print(det_dist)       # {49: 1.080123}  — d = D/diam для лучшей подрешётки
print(det_mat[49])    # матрица перехода M (HNF)
print(det_center[49]) # вектор-свидетель (центр ближайшей одноцветной ячейки)
```

## 3. Класс VoronoiPolyhedra

`VoronoiPolyhedra(grid)` — подкласс `scipy.spatial.Voronoi`. После `build()`:

| Атрибут | Смысл |
|---|---|
| `central` | вершины центральной ячейки V₀ (numpy) |
| `max_len` | диаметр diam(V₀) = 2·max\|вершина\| |
| `polyhedrons` | 3-мерные грани: объекты `Face3D` с `.normal`, `.bias`, `.center`, `.faces` |
| `delaunay` | триангуляция Делоне центральной ячейки |
| `vertex_to_faces` | для каждой вершины — индексы содержащих её граней |

`build(verbose=False)` подавляет диагностические распечатки.

Иерархия граней: `Face3D` → `Face2D` → `Edge2D`, у каждой нормаль (наружу),
смещение и центр.

## 4. Расстояния

```python
from voronoi4d import dist_to_s

s = 0.5 * np.array([1.0, 1.0, 0.0, 0.0])   # середина до соседнего центра
d = dist_to_s(vor4, s, vor4.max_len)        # нормированное d = 2·dist(s,V0)/max_len
```

> **Ранний выход.** При нормированном расстоянии < 1 `dist_to_s` возвращает
> текущий (не глобальный) минимум — верхнюю оценку. Для алгоритма достаточно
> факта d < 1, поэтому досчёт не делается; точное значение гарантируется
> только для d ≥ 1.

## 5. find_optimal

```python
find_optimal(det_range, limits, grid, vor4, max_len,
             precision=12, threshold=1.0,
             output_file="results.txt", verbose=True)
```

- `det_range` — `range`/список определителей (чисел цветов k);
- `limits` — УСТАРЕЛ и игнорируется (оставлен для совместимости сигнатуры);
- `precision` — точность целочисленного масштабирования для LLL (fpylll-путь);
- `threshold` — матрицы с d < threshold пропускаются (1.0 = только пригодные);
- `output_file` — результаты дописываются сюда (`save_result`);
- `verbose` — печать прогресса.

Возвращает три словаря с ключом-определителем: `det_dist` (d), `det_center`
(вектор-свидетель), `det_mat` (матрица перехода).

## 6. Генерация решёток

```python
from voronoi4d import generate_grids, generate_integer_grids, min_max_det

grids = generate_grids(norm_limit=2.0, norm_factor=1.5, cos_limit=0.3, det_limit=2.0)
ints  = generate_integer_grids(coeff_range=(-1, 2))   # полный перебор целочисленных
lo, hi = min_max_det(grids)
```

## 7. LLL

```python
from voronoi4d import lll_reduce, lll_reduce_python, HAS_FPYLLL

lll_reduce(basis, delta=0.75, precision=12)   # fpylll если есть, иначе чистый python
lll_reduce_python(basis, delta=0.75)         # всегда чистый python
```

## 8. Тесты и ноутбуки

```bash
python -m pytest                 # тесты пакета
```

- `notebooks/voronoi_main.ipynb` — основной сценарий;
- `notebooks/voronoi_info.ipynb` — устройство структур данных;
- `docs/article.tex` — теоретическое обоснование метода.

## 9. Ограничения

- Только размерность 4 (зашита в `product(..., repeat=4)`, `COEFF_RANGE`).
- `find_optimal` всегда пишет в `output_file` — при программном использовании
  передавайте временный путь, если файл не нужен.
- Для размерностей 2/3/5/6 или большей скорости используйте **combigeo**.
