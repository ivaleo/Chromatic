# План переработки кода и материалов Chromatic

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Цель:** привести монорепо Chromatic к состоянию, в котором чужой человек клонирует
его на чужую машину, одной командой прогоняет все тесты, находит нужный скрипт и
данные за минуту и не спотыкается о мёртвый код, дубли и 187 МБ безымянных выгрузок.

**Подход:** три библиотеки (`voronoi`, `combigeo`, `chromatic`) уже в порядке — их
трогаем точечно (мёртвый API, сигнатуры, горячий путь). Основная работа в
`audit-data/`: он превращается из плоской свалки на 152 скрипта и 662 json в
устанавливаемый пакет `chromatic_research` (ядро + кампании), отдельный каталог
тестов и разделённые `results/` (опорные) и `runs/` (сырые, сжатые). Материалы
расслаиваются на постоянные (`docs/`), датированные (`journal/`) и статью.

**Стек:** Python ≥ 3.10, numpy/scipy/sympy, pybind11 + C++17 (combigeo), pytest,
setuptools, LaTeX (pdflatex + latexmk), GNU make.

## Глобальные ограничения

- **Проверочный шлюз после каждой задачи:** `199 passed` за ~9 с. Команда:
  `.venv/bin/python -m pytest voronoi/tests chromatic/tests combigeo/tests/python audit-data -q`
  (после задачи 1 — просто `make test`). Замерено 2026-08-05: 199 passed, 8.66 s.
- **Ни один числовой результат не должен измениться.** Все константы
  (`d = 1.016339`, `χ(ℝ⁴) ≤ 45`, `χ(ℝ⁵) ≤ 132`, `χ(ℝ⁷) ≤ 1323`) и содержимое
  сертификатов `cert45/46/48.json` — неприкосновенны. Задачи, меняющие вычисления,
  обязаны иметь характеризационный тест ДО правки.
- **Никакой перезаписи истории git.** `.git` = 52 МБ, это терпимо; `filter-repo`
  и `lfs migrate` в план не входят.
- **Язык:** проза, README и статья — русский; код, имена, докстринги новых
  модулей — английский (так уже написаны 85 модулей `audit-data`).
- **Версия пакетов:** `1.1.0` во всех трёх `pyproject.toml`. Не поднимать в этом
  плане, кроме задачи 2, где чинится рассинхрон установленных метаданных.
- **Стиль коммитов:** `<область>: <что сделано>` на русском, например
  `voronoi4d: убрать мёртвый API совместимости`. Коммит после каждой задачи.

---

## Целевая структура

```
Chromatic/
├── Makefile                    ← НОВЫЙ: test / lint / paper / figures / clean / install
├── pyproject.toml              ← НОВЫЙ: только [tool.pytest] и [tool.ruff], без [project]
├── .gitignore                  ← единственный (подпроектные удалены)
├── README.md                   ← только ориентация + таблица результатов
├── RESULTS.md                  ← НОВЫЙ: журнал кампаний (переехал из README)
├── docs/                       ← НОВЫЙ: постоянные документы
│   └── superpowers/plans/      ← этот план
├── journal/                    ← НОВЫЙ: всё датированное (бывш. archive/ + RESEARCH_* + AUDIT-* + PLAN-*)
├── voronoi/                    ← структура без изменений
├── combigeo/                   ← структура без изменений
├── chromatic/                  ← структура без изменений
├── audit-data/
│   ├── pyproject.toml          ← НОВЫЙ: пакет chromatic-research
│   ├── README.md               ← индекс
│   ├── README-dim5-9.md        ← бывш. hd-2026-07/README.md
│   ├── chromatic_research/     ← НОВЫЙ пакет
│   │   ├── __init__.py
│   │   ├── paths.py            ← results_path / runs_path / load_json
│   │   ├── forms.py            ← norm_gram / pack / unpack (13 копий → 1)
│   │   ├── core/               ← 15 модулей с in-degree ≥ 5
│   │   └── campaigns/          ← 103 модуля-кампании
│   ├── tests/                  ← 34 теста
│   ├── results/                ← 125 опорных json (35 МБ)
│   └── runs/                   ← 537 сырых прогонов, *.json.gz (~14 МБ)
├── paper/
│   ├── chi4-45.tex             ← преамбула + \input
│   ├── sections/*.tex          ← 9 секций
│   └── figures.py
└── articles/                   ← без изменений (исходные материалы Иванова)
```

Каталоги `tmp/`, `output/`, `archive/`, `audit-data/hd-2026-07/` исчезают.

---

# Фаза 0. Инфраструктура проверки

Без неё остальные фазы нечем проверять. Две задачи, обе быстрые.

## Задача 1: единая точка запуска (pyproject + Makefile + один .gitignore)

**Файлы:**
- Создать: `pyproject.toml`
- Создать: `Makefile`
- Изменить: `.gitignore`
- Удалить: `voronoi/.gitignore`, `chromatic/.gitignore`, `combigeo/.gitignore`

**Интерфейс (используется всеми последующими задачами):**
- `make test` — 199 тестов
- `make lint` — ruff по всему репо
- `make figures` / `make paper` — пересборка рисунков и PDF
- `make install` — установка всех пакетов в `.venv` в правильном порядке

- [x] **Шаг 1: зафиксировать базовую линию**

```bash
cd /Users/mac/Documents/_My_code/Chromatic
.venv/bin/python -m pytest voronoi/tests chromatic/tests combigeo/tests/python audit-data -q 2>&1 | tail -2
```

Ожидается: `199 passed`. Если число другое — остановиться и разобраться, план
опирается на это значение.

- [x] **Шаг 2: создать корневой `pyproject.toml`**

Секции `[project]` и `[build-system]` НЕ добавляем: корень — не пакет, и
`pip install .` в нём не должен работать.

```toml
# Корень монорепо пакетом не является: здесь только настройки инструментов.
# Установка пакетов — см. Makefile (make install) и README.

[tool.pytest.ini_options]
testpaths = [
    "voronoi/tests",
    "chromatic/tests",
    "combigeo/tests/python",
    "audit-data",
]
pythonpath = ["voronoi/src", "chromatic/src"]

[tool.ruff]
line-length = 100
target-version = "py310"
exclude = [".venv", "audit-data/runs", "paper", "journal"]

[tool.ruff.lint]
# E501 (длина строки) проверяется, F401 (неиспользуемые импорты) — да;
# исследовательский код не обязан быть идеальным, поэтому набор правил узкий.
select = ["E4", "E7", "E9", "F"]
```

- [x] **Шаг 3: проверить, что конфиг не сломал сбор тестов**

```bash
.venv/bin/python -m pytest -q 2>&1 | tail -3
```

Ожидается: `199 passed`. Если pytest вдруг подхватил другой rootdir (например,
`voronoi/pyproject.toml`), запустить с явным `-c pyproject.toml`.

- [x] **Шаг 4: создать `Makefile`**

```makefile
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install test lint figures paper clean

help:
	@echo "install  — поставить voronoi4d, combigeo, chromatic и chromatic-research в .venv"
	@echo "test     — все тесты монорепо (ожидается 199 passed)"
	@echo "lint     — ruff по коду"
	@echo "figures  — пересобрать рисунки статьи из данных"
	@echo "paper    — собрать paper/chi4-45.pdf"
	@echo "clean    — убрать артефакты сборки (кроме .venv)"

install:
	$(PIP) install -q pytest ruff
	$(PIP) install -e voronoi
	$(PIP) install ./combigeo
	$(PIP) install -e 'chromatic[dev]'
	$(PIP) install -e 'audit-data[solvers]'

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

figures:
	$(PY) paper/figures.py

paper:
	cd paper && latexmk -pdf chi4-45.tex

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache */.pytest_cache combigeo/build
	cd paper && latexmk -c chi4-45.tex || true
```

Цель `install` включает `audit-data` — он станет пакетом в задаче 7. До неё
строка будет падать, поэтому добавляем её сразу, но саму цель до фазы 3 не
запускаем (в этом плане `make install` первый раз вызывается в задаче 7).

- [x] **Шаг 5: свести `.gitignore` в один корневой**

Заменить содержимое корневого `.gitignore` целиком:

```gitignore
# окружение и локальные настройки
.venv/
venv/
.claude/
.DS_Store

# python
__pycache__/
*.pyc
*.egg-info/
build*/
dist/
.pytest_cache/
.ipynb_checkpoints/

# C++ / clangd
*.o
*.so
*.dylib
.cache/

# логи
*.log

# LaTeX
paper/*.aux
paper/*.fdb_latexmk
paper/*.fls
paper/*.log
paper/*.out
paper/*.synctex.gz
paper/*.toc

# результаты локальных прогонов voronoi4d
results.txt

# воспроизводимые, но дорогие кэши запрещённых множеств
audit-data/**/.search-cache/
audit-data/**/*_forbidden*.npz

# локально скомпилированные переборщики
audit-data/**/threshold_enum
audit-data/**/threshold_enum_rebuilt
audit-data/**/threshold_mask_enum
```

```bash
git rm --cached -q voronoi/.gitignore chromatic/.gitignore combigeo/.gitignore
rm voronoi/.gitignore chromatic/.gitignore combigeo/.gitignore
```

- [x] **Шаг 6: проверить, что ничего лишнего не всплыло и не пропало**

```bash
git status --porcelain -uall | head -20      # должно быть пусто, кроме правок выше
git ls-files | wc -l                          # было 945; станет 945 - 3 + 3 = 945
```

- [x] **Шаг 7: добавить CI**

Репозиторий живёт на GitHub (`Homepage` во всех трёх `pyproject.toml`), поэтому
шлюз «199 тестов» имеет смысл вынести в Actions. Создать `.github/workflows/ci.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Установить пакеты монорепо
        run: |
          python -m pip install --upgrade pip
          pip install pytest ruff
          pip install -e voronoi
          pip install ./combigeo          # нужен компилятор C++17; в ubuntu-latest есть
          pip install -e 'chromatic[dev]'

      - name: Тесты
        run: pytest -q

      - name: Линтер
        run: ruff check .
```

Строку `pip install -e 'audit-data[solvers]'` добавить в этот workflow **после
задачи 8** (пакета `chromatic-research` пока не существует), а `testpaths` в
корневом `pyproject.toml` до тех пор оставляют `audit-data` — в CI тесты
`audit-data` до фазы 3 не пройдут из-за абсолютных путей. Поэтому на этом шаге в
workflow временно ограничиваем прогон:

```yaml
      - name: Тесты
        run: pytest -q voronoi/tests chromatic/tests combigeo/tests/python
```

и возвращаем полный `pytest -q` в задаче 11, шаг 12.

- [x] **Шаг 8: коммит**

```bash
git add pyproject.toml Makefile .gitignore .github/workflows/ci.yml
git commit -m "инфраструктура: единая точка запуска тестов, Makefile, один .gitignore, CI"
```

---

## Задача 2: починить рассинхрон версий пакетов

`voronoi4d.__version__` возвращает `1.0.0`, хотя `voronoi/pyproject.toml` объявляет
`1.1.0`, а корневой README утверждает «Все три пакета — версия 1.1.0». Причина —
устаревшие метаданные editable-установки. Тест ниже не даёт этому повториться.

**Файлы:**
- Создать: `chromatic/tests/test_versions.py`

- [x] **Шаг 1: воспроизвести дефект**

```bash
.venv/bin/python -c "import voronoi4d; print(voronoi4d.__version__)"
```

Ожидается: `1.0.0` (расходится с `pyproject.toml`).

- [x] **Шаг 2: написать падающий тест**

Создать `chromatic/tests/test_versions.py`:

```python
"""Версии установленных пакетов не должны расходиться с pyproject.toml."""

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = [
    ("voronoi4d", ROOT / "voronoi" / "pyproject.toml"),
    ("combigeo", ROOT / "combigeo" / "pyproject.toml"),
    ("chromatic", ROOT / "chromatic" / "pyproject.toml"),
]


@pytest.mark.parametrize("dist_name,pyproject", PACKAGES)
def test_installed_version_matches_pyproject(dist_name, pyproject):
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert version(dist_name) == declared, (
        f"{dist_name}: установлено {version(dist_name)}, объявлено {declared}. "
        f"Переустановите: pip install -e {pyproject.parent.name}"
    )
```

- [x] **Шаг 3: убедиться, что тест падает**

```bash
.venv/bin/python -m pytest chromatic/tests/test_versions.py -q
```

Ожидается: `1 failed` — `voronoi4d: установлено 1.0.0, объявлено 1.1.0`.

- [x] **Шаг 4: переустановить пакет**

```bash
.venv/bin/pip install -e voronoi --no-deps --force-reinstall -q
```

- [x] **Шаг 5: тест зелёный**

```bash
.venv/bin/python -m pytest chromatic/tests/test_versions.py -q   # 3 passed
make test                                                        # 202 passed
```

- [x] **Шаг 6: коммит**

```bash
git add chromatic/tests/test_versions.py
git commit -m "тесты: сверка версий установленных пакетов с pyproject"
```

---

# Фаза 1. Чистка репозитория

## Задача 3: убрать `tmp/` и `output/`

`tmp/pdfs/` — 27 PNG-страниц статьи под контролем версий, не упоминаются нигде
(проверено grep по `*.md`, `*.py`, `*.tex`). `output/pdf/chi4-45.pdf` — устаревший
дубль `paper/chi4-45.pdf` (31.07 против 05.08, разные md5), опасен тем, что его
можно прочитать вместо актуального.

**Файлы:**
- Удалить: `tmp/` (27 файлов), `output/` (1 файл)
- Изменить: `README.md` (если есть ссылки), `.gitignore`

- [x] **Шаг 1: убедиться, что на них правда никто не ссылается**

```bash
grep -rn "tmp/pdfs\|output/pdf" --include='*.md' --include='*.py' --include='*.tex' . | grep -v .venv
```

Ожидается: пусто. Если что-то найдено — сначала починить ссылку, потом удалять.

- [x] **Шаг 2: сверить, что удаляемый PDF действительно устарел**

```bash
md5 -q output/pdf/chi4-45.pdf paper/chi4-45.pdf
ls -l output/pdf/chi4-45.pdf paper/chi4-45.pdf
```

Ожидается: разные хеши, `paper/` новее. Если наоборот — остановиться: значит,
`output/` содержит что-то уникальное.

- [x] **Шаг 3: удалить**

```bash
git rm -r -q tmp output
```

- [x] **Шаг 4: закрыть путь назад**

Добавить в конец `.gitignore`:

```gitignore
# рабочие каталоги сессий: не под контролем версий
/tmp/
/output/
```

- [x] **Шаг 5: проверка и коммит**

```bash
make test                                     # 202 passed
git ls-files | wc -l                          # на 28 меньше
git add -A && git commit -m "репозиторий: убрать tmp/ и устаревший дубль output/pdf"
```

---

# Фаза 2. voronoi4d: ясность и скорость

Пакет маленький (1455 строк) и с хорошими тестами — здесь безопасно наводить порядок.

## Задача 4: удалить мёртвый API совместимости

Три экспортируемые сущности не используются нигде, кроме собственных тестов, но
занимают место в публичном интерфейсе и требуют объяснений:

| Что | Где | Состояние |
|---|---|---|
| `pad_lists_with_ones` | `factorization.py:45` | с 1.1.0 тождественная функция |
| `lattice_points_no_central_symmetry` | `search.py:28` | помечена «УСТАРЕЛА», вытеснена `lattice_points_within` |
| `find_faces_from_nearest_vertices` | `distances.py:35` | помечена «УСТАРЕЛА», эвристика без доказательства |

**Файлы:**
- Изменить: `voronoi/src/voronoi4d/factorization.py:45-54` (удалить функцию)
- Изменить: `voronoi/src/voronoi4d/search.py:21,28-65,119` (удалить функцию и её вызов)
- Изменить: `voronoi/src/voronoi4d/distances.py:11,35-62` (удалить функцию)
- Изменить: `voronoi/src/voronoi4d/__init__.py` (убрать 3 импорта и 3 строки `__all__`)
- Изменить: `voronoi/tests/test_factorization.py:8,46-58` (удалить 2 теста)
- Изменить: `voronoi/tests/test_distances.py:5-22` (удалить импорт и 1 тест)
- Изменить: `voronoi/README.md:65`, `voronoi/docs/USAGE.md` (упоминания)

- [x] **Шаг 1: подтвердить, что внешних потребителей нет**

```bash
grep -rn "pad_lists_with_ones\|lattice_points_no_central_symmetry\|find_faces_from_nearest_vertices" \
  --include='*.py' --include='*.ipynb' --include='*.md' . | grep -v .venv | grep -v 'src/voronoi4d'
```

Ожидается: только `voronoi/tests/test_factorization.py` и `voronoi/tests/test_distances.py`.

- [x] **Шаг 2: удалить `pad_lists_with_ones`**

Из `voronoi/src/voronoi4d/factorization.py` удалить строки 45-54 целиком
(функцию `pad_lists_with_ones` вместе с докстрингом).

В `voronoi/src/voronoi4d/search.py` заменить строку 21:

```python
from .factorization import compute_factorizations, pad_lists_with_ones
```

на:

```python
from .factorization import compute_factorizations
```

и строку 119:

```python
        list_diag_el = pad_lists_with_ones(compute_factorizations(det))
```

на:

```python
        list_diag_el = compute_factorizations(det)
```

- [x] **Шаг 3: удалить `lattice_points_no_central_symmetry`**

Из `voronoi/src/voronoi4d/search.py` удалить строки 28-65 (функцию с докстрингом)
и разделитель-комментарий над ней. Затем удалить ставшие ненужными импорты в
шапке файла — после удаления функции `product` ещё нужен (используется в цикле
HNF на строке 139), а `distance` из scipy больше не нужен:

```python
from scipy.spatial import distance
```

— эту строку (`search.py:17`) удалить.

- [x] **Шаг 4: удалить `find_faces_from_nearest_vertices`**

Из `voronoi/src/voronoi4d/distances.py` удалить строки 35-62 вместе с
разделителями-комментариями и константой `TOL_NEAREST` (строка 15) — она
используется только этой функцией. Проверить:

```bash
grep -n "TOL_NEAREST" voronoi/src/voronoi4d/*.py    # должно остаться пусто
```

- [x] **Шаг 5: почистить `__init__.py`**

В `voronoi/src/voronoi4d/__init__.py` удалить:
- из блока `from .distances import (...)` — строку `find_faces_from_nearest_vertices,`
- из блока `from .factorization import (...)` — строку `pad_lists_with_ones,`
- строку `from .search import find_optimal, lattice_points_no_central_symmetry`
  заменить на `from .search import find_optimal`
- из `__all__` — строки `"find_faces_from_nearest_vertices",`,
  `"lattice_points_no_central_symmetry",`, `"pad_lists_with_ones",`

- [x] **Шаг 6: удалить осиротевшие тесты**

В `voronoi/tests/test_factorization.py` удалить строки 46-58 (тесты
`test_pad_lists_with_ones` и `test_pad_lists_with_ones_does_not_mutate_input`) и
убрать `pad_lists_with_ones` из импорта на строке 8.

В `voronoi/tests/test_distances.py` удалить тест
`test_lattice_points_no_central_symmetry` (строки 12-22) и убрать
`lattice_points_no_central_symmetry` из импорта в шапке.

- [x] **Шаг 7: обновить документацию**

В `voronoi/README.md` и `voronoi/docs/USAGE.md` убрать абзацы про удалённые
функции. Найти их:

```bash
grep -n "pad_lists_with_ones\|lattice_points_no_central\|find_faces_from_nearest" \
  voronoi/README.md voronoi/docs/USAGE.md
```

- [x] **Шаг 8: проверка**

```bash
.venv/bin/python -c "import voronoi4d; print(len(voronoi4d.__all__))"   # было 27, стало 24
make test                                                               # 199 passed (3 теста удалены)
```

- [x] **Шаг 9: коммит**

```bash
git add voronoi chromatic
git commit -m "voronoi4d: удалить мёртвый API совместимости (3 функции)"
```

---

## Задача 5: привести в порядок сигнатуру `find_optimal`

Сейчас: `find_optimal(det_range, limits, grid, vor4, max_len, precision=12, threshold=1.0, output_file="results.txt", verbose=True)`.
Параметр `limits` принимается и тут же выбрасывается (`search.py:88` — `del limits`),
все вызывающие обязаны передавать `None` позиционно. `verbose=True` печатает в
stdout из вычислительного ядра — в библиотеке это неуместно.

**Файлы:**
- Изменить: `voronoi/src/voronoi4d/search.py:71-208`
- Изменить: `chromatic/src/chromatic/_voronoi4d.py:110-113`
- Изменить: `voronoi/notebooks/voronoi_main.ipynb` (2 вызова)
- Изменить: `voronoi/docs/USAGE.md:38,84,90`, `voronoi/README.md:131`
- Создать: `voronoi/tests/test_search_progress.py`

**Интерфейс (используется задачей 6 и фасадом):**
- Новая сигнатура:
  `find_optimal(det_range, grid, vor4, max_len, *, precision=12, threshold=1.0, output_file="results.txt", progress=None)`
- `progress: Callable[[str], None] | None` — если задан, вызывается со строками
  прогресса; по умолчанию молчит.

- [x] **Шаг 1: написать падающий тест на новую сигнатуру**

Создать `voronoi/tests/test_search_progress.py`:

```python
"""Новая сигнатура find_optimal: без limits, с callback вместо print."""

import numpy as np

from voronoi4d import VoronoiPolyhedra, find_optimal


def test_find_optimal_accepts_keyword_only_options(tmp_path):
    grid = np.eye(4)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)

    det_dist, _, _ = find_optimal(
        range(2, 3), grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
    )

    assert 2 in det_dist


def test_progress_callback_receives_lines(tmp_path):
    grid = np.eye(4)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)

    lines = []
    find_optimal(
        range(2, 3), grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
        progress=lines.append,
    )

    assert lines, "progress должен получить хотя бы одну строку"
    assert any("det" in line for line in lines)


def test_no_progress_means_silence(tmp_path, capsys):
    grid = np.eye(4)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)

    find_optimal(
        range(2, 3), grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
    )

    assert capsys.readouterr().out == ""
```

- [x] **Шаг 2: убедиться, что тест падает**

```bash
.venv/bin/python -m pytest voronoi/tests/test_search_progress.py -q
```

Ожидается: `3 failed` — старая сигнатура требует `limits` вторым позиционным
аргументом.

- [x] **Шаг 3: поменять сигнатуру и вывод**

В `voronoi/src/voronoi4d/search.py` заменить объявление функции (строки 71-72) на:

```python
def find_optimal(det_range, grid, vor4, max_len, *, precision=12, threshold=1.0,
                 output_file="results.txt", progress=None):
```

Обновить докстринг: убрать описание `limits` и `verbose`, добавить

```
    :param progress: необязательный callback вида f(str) для сообщений о ходе
                     перебора; по умолчанию функция молчит.
```

Удалить строку 88 (`del limits  # устарел: ...`) вместе с комментарием над ней.

Сразу после проверок согласованности аргументов добавить:

```python
    def report(message):
        """Сообщение о ходе перебора — только если вызывающий об этом попросил."""
        if progress is not None:
            progress(message)
```

- [x] **Шаг 4: заменить все `print` на `report`**

В теле `find_optimal` заменить блоки печати. Строки 112-116:

```python
    for det in det_range:
        if verbose:
            print("\r                                                          ")
            print("---------------------------------")
            print("det:", det)
```

на:

```python
    for det in det_range:
        report(f"det: {det}")
```

Строки 134-136:

```python
            if verbose:
                print("\r                                                          ")
                print("▶ diag factors:", *diag_el, "   iters:", num_iterations)
```

на:

```python
            report(f"диагональ {diag_el}: {num_iterations} итераций")
```

Строки 150-152:

```python
                if verbose and iteration % 500 == 0:
                    print("\r[", int(10000 * iteration / num_iterations) / 100, "% ]",
                          "   iter:", iteration, end="")
```

на:

```python
                if iteration % 500 == 0:
                    report(f"  {100 * iteration / num_iterations:.2f}% ({iteration})")
```

Строки 189-190:

```python
                if verbose:
                    print("\r", mat, "                      \n", min_dist_mat, min_center)
```

на:

```python
                report(f"  кандидат d={min_dist_mat:.6f} при {min_center}")
```

- [x] **Шаг 5: обновить фасад**

В `chromatic/src/chromatic/_voronoi4d.py` заменить строки 110-113:

```python
            det_dist, det_center, det_mat = _find_optimal(
                range(index, index + 1), None, reduced, vor, vor.max_len,
                threshold=0.0, output_file=tmp.name, verbose=False,
            )
```

на:

```python
            det_dist, det_center, det_mat = _find_optimal(
                range(index, index + 1), reduced, vor, vor.max_len,
                threshold=0.0, output_file=tmp.name,
            )
```

- [x] **Шаг 6: обновить notebook**

В `voronoi/notebooks/voronoi_main.ipynb` два места. Найти:

```bash
grep -n "limits" voronoi/notebooks/voronoi_main.ipynb
```

Заменить вызов `find_optimal(range(49, 50), None, grid, vor4, vor4.max_len, threshold=1.0,)`
на `find_optimal(range(49, 50), grid, vor4, vor4.max_len, threshold=1.0, progress=print)`
и убрать строку-комментарий про устаревший `limits`.

- [x] **Шаг 7: обновить документацию**

В `voronoi/docs/USAGE.md` (строки 38, 84, 90) и `voronoi/README.md` (строки 65, 131)
убрать `limits` из примеров и описания, добавить строку про `progress`:

```markdown
- `progress` — необязательный callback `f(str)` для сообщений о ходе перебора
  (например, `progress=print`); по умолчанию функция ничего не печатает.
```

- [x] **Шаг 8: тесты зелёные**

```bash
.venv/bin/python -m pytest voronoi/tests/test_search_progress.py -q   # 3 passed
make test                                                             # 202 passed
```

- [x] **Шаг 9: коммит**

```bash
git add voronoi chromatic
git commit -m "voronoi4d: убрать мёртвый параметр limits, verbose -> progress callback"
```

---

## Задача 6: упростить внутренности `find_optimal`

Три параллельных словаря (`mat_dist`, `mat_center`, `list_mats`), синхронизируемые
вручную счётчиком `index`, — источник ошибок и лишнего чтения.

**Файлы:**
- Изменить: `voronoi/src/voronoi4d/search.py:97-208`

- [x] **Шаг 1: убедиться, что поведение зафиксировано тестами**

```bash
.venv/bin/python -m pytest voronoi/tests/test_search.py -q
```

Ожидается: зелено. Это характеризационная страховка для рефакторинга.

- [x] **Шаг 2: добавить запись-кандидата**

В шапку `voronoi/src/voronoi4d/search.py` добавить импорт:

```python
from dataclasses import dataclass
```

И перед `find_optimal` — тип записи:

```python
@dataclass
class _Candidate:
    """Подрешётка-кандидат: матрица перехода, минимальное d и точка минимума."""

    matrix: "np.ndarray"
    distance: float
    center: "np.ndarray"
```

- [x] **Шаг 3: заменить три словаря на список**

Строки 122-125:

```python
        mat_dist = {}
        mat_center = {}
        list_mats = []
        index = 0
```

на:

```python
        candidates = []
```

Строки 192-196:

```python
                # сохраняем значения для mat (копию: mat мутируется в цикле!)
                list_mats.append(mat.copy())
                mat_dist[index] = min_dist_mat
                mat_center[index] = min_center
                index += 1
```

на:

```python
                # копия: mat мутируется в цикле перебора наддиагональных элементов
                candidates.append(_Candidate(mat.copy(), min_dist_mat, min_center.copy()))
```

Строки 198-206:

```python
        if mat_dist:
            best_index, best_dist = max(mat_dist.items(), key=lambda item: item[1])
            det_dist[det] = best_dist
            det_center[det] = mat_center[best_index]
            det_mat[det] = list_mats[best_index]

            save_result(grid, det, list_mats[best_index], mat_center[best_index], best_dist,
                        output_file=output_file)
```

на:

```python
        if candidates:
            best = max(candidates, key=lambda c: c.distance)
            det_dist[det] = best.distance
            det_center[det] = best.center
            det_mat[det] = best.matrix

            save_result(grid, det, best.matrix, best.center, best.distance,
                        output_file=output_file)
```

Внимание: во внутреннем цикле есть локальная переменная `candidates` (строка 173,
результат `lattice_points_within`). Переименовать её в `neighbours`, иначе имена
столкнутся:

```python
                    neighbours = sorted(lattice_points_within(sub_grid_lll, bound),
                                        key=lambda v: float(v @ v))
                    for center in neighbours:
```

- [x] **Шаг 4: проверка**

```bash
make test                                                # 202 passed
grep -n "mat_dist\|list_mats\|index += 1" voronoi/src/voronoi4d/search.py   # пусто
```

- [x] **Шаг 5: коммит**

```bash
git add voronoi/src/voronoi4d/search.py
git commit -m "voronoi4d: один список записей вместо трёх параллельных словарей"
```

---

## Задача 7: ускорить `dist_to_s` (замерено ×1.4)

Профиль на D₄ (200 точек, 1.322 с): `scipy.spatial.distance.euclidean` — 0.769 с,
58% времени, 263 364 вызова. Почти всё это внутренняя валидация scipy
(`asarray_chkfinite`, `_validate_vector`, `minkowski`, `linalg.norm`), которая на
4-мерных векторах дороже самого вычисления. Замерено: замена на `math.sqrt(d @ d)`
даёт 0.921 → 0.663 с (×1.39) при побитово совпадающих результатах.

**Файлы:**
- Создать: `voronoi/tests/test_distances_characterization.py`
- Изменить: `voronoi/src/voronoi4d/distances.py:7-15,68-176`

- [x] **Шаг 1: написать характеризационный тест (он должен пройти ДО правки)**

Создать `voronoi/tests/test_distances_characterization.py`:

```python
"""Опорные значения dist_to_s: страховка для оптимизаций горячего пути.

Значения получены на текущей реализации 2026-08-05 и не должны меняться
ни при каких оптимизациях — только при изменении самого алгоритма.
"""

import numpy as np
import pytest

from voronoi4d import VoronoiPolyhedra, dist_to_s

D4 = np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float)

# 12 точек из np.random.default_rng(20260805).normal(size=4) * 1.2
EXPECTED = [
    1.877943432871477, 0.15194681369983248, 2.8776962779871913,
    1.8344149411138804, 0.3182553077904282, 1.1951097888868285,
    2.419226762485762, 0.921579230714697, 0.9523475752796856,
    1.3212828842435729, 1.5721696031153471, 2.5254306489745346,
]

# точки с известным геометрическим смыслом
FIXED = [
    ((1.0, 0.0, 0.0, 0.0), 0.0),                  # внутри ячейки
    ((0.5, 0.5, 0.0, 0.0), 0.0),                  # внутри ячейки
    ((1.0, 1.0, 0.0, 0.0), 0.7071067811865475),   # на грани
    ((0.75, 0.25, 0.5, 0.0), 0.17677669529663687),
    ((2.0, 0.0, 0.0, 0.0), 1.0),                  # ровно порог d = 1
]


@pytest.fixture(scope="module")
def cell():
    vor = VoronoiPolyhedra(D4)
    vor.build(verbose=False)
    return vor


def test_random_points_match_reference(cell):
    rng = np.random.default_rng(20260805)
    got = [float(dist_to_s(cell, rng.normal(size=4) * 1.2, cell.max_len, early_stop=0.0))
           for _ in range(12)]
    assert got == pytest.approx(EXPECTED, abs=1e-12)


@pytest.mark.parametrize("point,expected", FIXED)
def test_fixed_points_match_reference(cell, point, expected):
    got = dist_to_s(cell, np.array(point), cell.max_len, early_stop=0.0)
    assert got == pytest.approx(expected, abs=1e-12)


def test_result_is_plain_float(cell):
    got = dist_to_s(cell, np.array([1.0, 1.0, 0.0, 0.0]), cell.max_len, early_stop=0.0)
    assert type(got) is float, "функция должна возвращать float, а не np.float64"
```

- [x] **Шаг 2: запустить — 17 из 18 проходят**

```bash
.venv/bin/python -m pytest voronoi/tests/test_distances_characterization.py -q
```

Ожидается: `1 failed, 6 passed` — падает только `test_result_is_plain_float`
(сейчас возвращается `np.float64`). Это и есть цель правки. Если падает
что-то ещё — остановиться: значит, опорные значения сняты не с той сборки.

- [x] **Шаг 3: заменить scipy-расстояние на прямое**

В `voronoi/src/voronoi4d/distances.py` удалить импорт scipy (строка 11):

```python
from scipy.spatial import distance
```

и добавить после импорта numpy:

```python
def _dist(a, b):
    """Евклидово расстояние между 4-мерными точками.

    scipy.spatial.distance.euclidean на векторах такой длины тратит на
    валидацию входа больше, чем на само вычисление: в профиле dist_to_s это
    58% времени при 263 тыс. вызовов на 200 точек.
    """
    delta = a - b
    return math.sqrt(delta @ delta)
```

Заменить шесть вызовов в `dist_to_s`:

| Строка | Было | Стало |
|---|---|---|
| 132 | `dist = distance.euclidean(s, coord1)` | `dist = _dist(s, coord1)` |
| 149 | `dist = distance.euclidean(s, coord2)` | `dist = _dist(s, coord2)` |
| 157 | `d3 = distance.euclidean(coord2, edge.vertex1)` | `d3 = _dist(coord2, edge.vertex1)` |
| 158 | `d4 = distance.euclidean(coord2, edge.vertex2)` | `d4 = _dist(coord2, edge.vertex2)` |
| 161 | `dist = distance.euclidean(s, edge.vertex1)` | `dist = _dist(s, edge.vertex1)` |
| 164 | `dist = distance.euclidean(s, edge.vertex2)` | `dist = _dist(s, edge.vertex2)` |

- [x] **Шаг 4: убрать глобальный флаг `CHECK_DIST`**

`CHECK_DIST` не упоминается нигде за пределами своего модуля (проверено). Удалить
строку 13 и заменить проверки на параметр. Сигнатуру `dist_to_s` заменить на:

```python
def dist_to_s(vor4, s, max_len, early_stop=1.0, check=True):
```

Добавить в докстринг:

```
    :param check: сверять расстояния по теореме Пифагора (диагностика; выключение
                  экономит около 1% времени).
```

Три вхождения `if CHECK_DIST:` (строки 134, 151, 167) заменить на `if check:`.

- [x] **Шаг 5: вернуть обычный float**

Две строки возврата в конце `dist_to_s` (173-176):

```python
        if early_stop > 0.0 and min_dist_to_pol * 2 / max_len < early_stop:
            return min_dist_to_pol * 2 / max_len

    return min_dist_to_pol * 2 / max_len
```

заменить на:

```python
        if early_stop > 0.0 and min_dist_to_pol * 2 / max_len < early_stop:
            return float(min_dist_to_pol * 2 / max_len)

    return float(min_dist_to_pol * 2 / max_len)
```

- [x] **Шаг 6: убрать замыкание `update_min_distance`**

Оно читает свободную переменную `dist`, которую тело цикла присваивает побочным
эффектом в шести местах. Удалить строки 103-107:

```python
    def update_min_distance():
        nonlocal min_dist_to_pol

        if dist < min_dist_to_pol:
            min_dist_to_pol = dist
```

и заменить все шесть вызовов `update_min_distance()` на прямое:

```python
                min_dist_to_pol = min(min_dist_to_pol, dist)
```

В ветке на строках 119-121 это выглядит так:

```python
        if simplex != -1:  # проекция принадлежит центральному многограннику
            dist = abs(d0)
            min_dist_to_pol = min(min_dist_to_pol, dist)
            continue
```

- [x] **Шаг 7: все тесты зелёные, включая тип возврата**

```bash
.venv/bin/python -m pytest voronoi/tests/test_distances_characterization.py -q   # 7 passed
make test                                                                        # 209 passed
```

- [x] **Шаг 8: подтвердить ускорение замером**

```bash
.venv/bin/python - <<'PY'
import time, numpy as np
from voronoi4d import VoronoiPolyhedra, dist_to_s
D4 = np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float)
v = VoronoiPolyhedra(D4); v.build(verbose=False)
rng = np.random.default_rng(0); pts = [rng.normal(size=4)*1.2 for _ in range(200)]
t = time.perf_counter()
for p in pts: dist_to_s(v, p, v.max_len, early_stop=0.0)
print(f"{time.perf_counter()-t:.3f} с  (базовая линия 2026-08-05: 0.921 с)")
PY
```

Ожидается: около 0.66 с. Если больше 0.80 — оптимизация не применилась, проверить,
что заменены все шесть вызовов.

- [x] **Шаг 9: коммит**

```bash
git add voronoi/src/voronoi4d/distances.py voronoi/tests/test_distances_characterization.py
git commit -m "voronoi4d: ускорить dist_to_s в 1.4 раза, CHECK_DIST -> параметр"
```

---

> **Фазы 1 и 2 выполнены 05.08.2026** (коммиты `654b15d`, `e6269c8`, `e6d1afe`,
> `3f942a9`, `3aafe3a`). Отклонения:
> - Задача 4: линтер поймал, что после удаления `find_faces_from_nearest_vertices`
>   в `distances.py` осиротел импорт `numpy` — убран.
> - Задача 5: кроме перечисленных мест, старую сигнатуру звали ещё два теста
>   (`test_search.py:38,62`) — обновлены.
> - Задача 7: ускорение вышло **×1.59**, а не ×1.39: снятие замыкания
>   `update_min_distance` (109 тыс. вызовов) дало прибавку сверх замены scipy.
>   0.921 → 0.579 с; без проверки Пифагора 0.563 с.
>
> Итог фаз 1-2: `make test` → **209 passed**, `make lint` → **All checks passed**.

# Фаза 3. audit-data как устанавливаемый пакет

Ядро проблемы воспроизводимости: 47 строк `sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/...")`
в 28 файлах и абсолютные пути вывода ещё в двух десятках. На чужой машине
`audit-data` не работает вообще.

Измеренный граф импортов (152 файла, 34 из них тесты): 15 модулей с входящей
степенью ≥ 5 (взаимных импортов между ними нет — циклов не будет), 35 модулей
с 1-4 импортами, 68 листьев.

## Задача 8: скелет пакета `chromatic_research`

**Файлы:**
- Создать: `audit-data/pyproject.toml`
- Создать: `audit-data/chromatic_research/__init__.py`
- Создать: `audit-data/chromatic_research/paths.py`
- Создать: `audit-data/tests/test_paths.py`

**Интерфейс (используется всеми задачами фазы 3-5):**
- `chromatic_research.paths.RESULTS_DIR: Path` — `audit-data/results`
- `chromatic_research.paths.RUNS_DIR: Path` — `audit-data/runs`
- `chromatic_research.paths.results_path(name: str) -> Path` — путь для записи опорного результата
- `chromatic_research.paths.load_json(name: str) -> dict` — читает из `results/`, при
  отсутствии — `runs/<name>` или `runs/<name>.gz`; иначе `FileNotFoundError`

- [ ] **Шаг 1: создать `audit-data/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "chromatic-research"
version = "1.1.0"
description = "Вычислительные кампании и общие модули исследования верхних оценок χ(ℝⁿ)"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Leonid Ivanov", email = "leo.ivadev@gmail.com" }]
dependencies = ["numpy", "scipy", "sympy"]

[project.optional-dependencies]
# combigeo и voronoi4d — локальные пакеты монорепо, ставятся отдельно (см. Makefile).
solvers = ["ortools", "cma", "cvxpy"]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["."]
include = ["chromatic_research*"]
```

- [ ] **Шаг 2: создать `audit-data/chromatic_research/__init__.py`**

```python
"""Computational campaigns behind the χ(ℝⁿ) upper bounds.

`core` holds modules shared by several campaigns; `campaigns` holds the
individual runs.  Every artifact path goes through :mod:`chromatic_research.paths`
so that nothing depends on a checkout location.
"""

from . import paths

__all__ = ["paths"]
```

- [ ] **Шаг 3: создать `audit-data/chromatic_research/paths.py`**

```python
"""Artifact locations, resolved relative to the checkout — never absolute.

`results/` holds artifacts referenced by the paper, the READMEs or the tests.
`runs/` holds raw campaign output, gzipped; `load_json` transparently reads both.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

# chromatic_research/paths.py -> chromatic_research -> audit-data
AUDIT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = AUDIT_DIR / "results"
RUNS_DIR = AUDIT_DIR / "runs"


def results_path(name: str) -> Path:
    """Path for writing a supporting artifact; creates the directory if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def runs_path(name: str) -> Path:
    """Path for writing a raw campaign artifact; creates the directory if needed."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / name


def find_artifact(name: str) -> Path:
    """Locate an artifact by bare file name in results/ or runs/ (also gzipped)."""
    for candidate in (RESULTS_DIR / name,
                      RUNS_DIR / name,
                      RUNS_DIR / (name + ".gz")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"артефакт {name!r} не найден ни в {RESULTS_DIR}, ни в {RUNS_DIR}"
    )


def load_json(name: str) -> dict:
    """Read an artifact by bare file name, transparently handling gzip."""
    path = find_artifact(name)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Шаг 4: тест на пути**

Создать `audit-data/tests/test_paths.py`:

```python
"""Пути к артефактам не должны зависеть от каталога запуска и от машины."""

import json

from chromatic_research import paths


def test_dirs_are_inside_the_checkout():
    assert paths.RESULTS_DIR.parent == paths.AUDIT_DIR
    assert paths.RUNS_DIR.parent == paths.AUDIT_DIR
    assert paths.AUDIT_DIR.name == "audit-data"


def test_load_json_reads_plain_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    (tmp_path / "sample.json").write_text(json.dumps({"d": 1.5}))
    assert paths.load_json("sample.json") == {"d": 1.5}


def test_load_json_reads_gzipped_run(tmp_path, monkeypatch):
    import gzip

    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    paths.RUNS_DIR.mkdir(parents=True)
    with gzip.open(paths.RUNS_DIR / "raw.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"d": 0.99}, handle)
    assert paths.load_json("raw.json") == {"d": 0.99}


def test_missing_artifact_names_both_directories():
    try:
        paths.load_json("нет-такого.json")
    except FileNotFoundError as error:
        assert "results" in str(error) and "runs" in str(error)
    else:
        raise AssertionError("ожидался FileNotFoundError")
```

- [ ] **Шаг 5: установить пакет и прогнать тест**

```bash
.venv/bin/pip install -e audit-data -q
.venv/bin/python -m pytest audit-data/tests/test_paths.py -q     # 4 passed
make test                                                        # 213 passed
```

- [ ] **Шаг 6: коммит**

```bash
git add audit-data/pyproject.toml audit-data/chromatic_research audit-data/tests
git commit -m "audit-data: пакет chromatic_research и модуль путей к артефактам"
```

---

## Задача 9: main-guard и относительные пути вывода в 22 старых скриптах

Скрипты первого поколения (`campaign_*`, `n*`, `r*`, `o*`, `verify*`, `cert*`)
выполняют кампанию на уровне модуля — импорт такого файла запускает счёт — и
пишут результат по абсолютному пути вида
`open("/Users/mac/Documents/_My_code/Chromatic/audit-data/campaign_c.json", "w")`.
Пока они не станут импортируемыми без побочных эффектов, задача 10 (smoke-тест)
невозможна.

**Файлы (22 штуки):** `audit-data/`: `a5s_full140.py`, `campaign_a.py`,
`campaign_b.py`, `campaign_c.py`, `campaign_d.py`, `cert48.py`, `cert_generic.py`,
`n1_r3_full.py`, `n2_4d_frontier.py`, `n3_5d_probe.py`, `n4_push46.py`,
`n5_cascade.py`, `probe5d.py`, `r1_r3_refine.py`, `r2_cone48.py`, `r4_beat343.py`,
`r5_push48.py`, `sweep.py`, `verify2.py`, `verify_a5s.py`, `verify_e6s.py`,
`verify_e8.py`; `audit-data/hd-2026-07/`: `bench_mc.py`, `permutohedral_cover.py`

- [ ] **Шаг 1: получить точный список и убедиться, что он совпал**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
for f in $(find . -name '*.py' -not -name 'test_*' -not -path '*__pycache__*'); do
  grep -q '__main__' "$f" || echo "$f"
done | sort
```

Ожидается ровно 24 строки (22 + 2 из `hd-2026-07`).

- [ ] **Шаг 2: разобрать один файл как образец**

Для `audit-data/campaign_c.py`: весь код верхнего уровня, кроме импортов и
констант, обернуть в `def main():`, а запись результата перевести на `results_path`.

Было (строка 43):

```python
json.dump(out, open("/Users/mac/Documents/_My_code/Chromatic/audit-data/campaign_c.json", "w"), indent=1)
```

Стало:

```python
from chromatic_research.paths import results_path

...

def main():
    ...
    results_path("campaign_c.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Шаг 3: найти все абсолютные пути вывода**

```bash
grep -rn '"/Users/mac' --include='*.py' audit-data | grep -v sys.path
```

Каждое вхождение заменить на `results_path("<имя>.json")`. Файлы с
`f"...{переменная}.json"` (например `cma_form.py`, `dim6.py`, `csp_campaign5d.py`)
переводятся так же: `results_path(f"cma_d{dim}_k{k}.json")`.

- [ ] **Шаг 4: проверить, что импорт больше ничего не запускает**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
for f in $(find . -name '*.py' -not -name 'test_*' -not -path '*__pycache__*'); do
  grep -q '__main__' "$f" || echo "БЕЗ ГВАРДА: $f"
done
```

Ожидается: пусто.

- [ ] **Шаг 5: убедиться, что записи ушли из старого места**

```bash
grep -rn '"/Users/mac' --include='*.py' audit-data | grep -v sys.path
```

Ожидается: пусто (останутся только `sys.path`-хаки — их снимает задача 12).

- [ ] **Шаг 6: проверка и коммит**

```bash
make test                                    # 213 passed
git add audit-data
git commit -m "audit-data: main-guard и относительные пути вывода в скриптах первого поколения"
```

---

## Задача 10: smoke-тест импорта всех модулей

Тестами покрыто около 30 модулей из 118 — при массовом переписывании импортов
(задача 11) сломанный импорт в непокрытой кампании иначе не всплывёт.

**Файлы:**
- Создать: `audit-data/tests/test_all_modules_import.py`

- [ ] **Шаг 1: написать тест**

```python
"""Каждый модуль исследования должен импортироваться без побочных эффектов.

Это страховка для массовых перемещений: тестами покрыта лишь четверть модулей,
а сломанный импорт в непокрытой кампании иначе обнаружится только при запуске.
"""

import importlib
import pkgutil

import pytest

import chromatic_research


def _module_names():
    return sorted(
        info.name
        for info in pkgutil.walk_packages(chromatic_research.__path__,
                                          prefix="chromatic_research.")
        if not info.ispkg
    )


@pytest.mark.parametrize("name", _module_names())
def test_module_imports(name):
    importlib.import_module(name)
```

- [ ] **Шаг 2: запустить (пока модулей мало — пройдёт быстро)**

```bash
.venv/bin/python -m pytest audit-data/tests/test_all_modules_import.py -q
```

Ожидается: `1 passed` (пока в пакете только `paths`). После задачи 11 здесь
станет 118 тестов.

- [ ] **Шаг 3: коммит**

```bash
git add audit-data/tests/test_all_modules_import.py
git commit -m "тесты: smoke-проверка импортируемости модулей исследования"
```

---

## Задача 11: перенести 118 модулей в пакет и переписать импорты

Разделение по измеренной входящей степени: 15 модулей ядра (≥ 5 импортов,
взаимных зависимостей между ними нет) — в `core/`, остальные 103 — в `campaigns/`.

**Состав `chromatic_research/core/` (15 модулей, все из `hd-2026-07`, кроме двух):**
`prime_radon` (49 импортов), `prime_row_opt` (25), `determinant_repair` (13),
`metric_deform` (13), `active_metric_refine` (12), `lattices` (12), `covrad` (11),
`block_row_metric_opt` (11), `e7_abpr` (10), `lazy_prime_campaign` (10),
`campaign_hd` (7), `d6_cyclic_hole_search` (6), `d6_sdp_hybrid` (5),
а также `general_csp` (7) и `cyclic_csp` (6) — эти два лежат в `audit-data/`.

**Состав `chromatic_research/campaigns/`:** все остальные 103 не-тестовых модуля
из `audit-data/` и `audit-data/hd-2026-07/`.

**Файлы:**
- Создать: `audit-data/chromatic_research/core/__init__.py`, `audit-data/chromatic_research/campaigns/__init__.py`
- Переместить: 118 `.py` (git mv), 34 теста в `audit-data/tests/`
- Изменить: импорты во всех перемещённых файлах и тестах

- [ ] **Шаг 1: подготовить каталоги**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
mkdir -p chromatic_research/core chromatic_research/campaigns tests
printf '"""Modules shared by several campaigns."""\n' > chromatic_research/core/__init__.py
printf '"""Individual computational campaigns."""\n' > chromatic_research/campaigns/__init__.py
```

- [ ] **Шаг 2: перенести ядро**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
for m in prime_radon prime_row_opt determinant_repair metric_deform \
         active_metric_refine lattices covrad block_row_metric_opt e7_abpr \
         lazy_prime_campaign campaign_hd d6_cyclic_hole_search d6_sdp_hybrid; do
  git mv "hd-2026-07/$m.py" "chromatic_research/core/$m.py"
done
git mv general_csp.py chromatic_research/core/general_csp.py
git mv cyclic_csp.py chromatic_research/core/cyclic_csp.py
```

- [ ] **Шаг 3: перенести тесты**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
git mv hd-2026-07/test_*.py tests/
```

- [ ] **Шаг 4: перенести остальные модули в campaigns**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
for f in hd-2026-07/*.py *.py; do
  [ -e "$f" ] || continue
  git mv "$f" "chromatic_research/campaigns/$(basename "$f")"
done
```

Проверить, что в `audit-data/` и `audit-data/hd-2026-07/` не осталось `.py`:

```bash
ls *.py hd-2026-07/*.py 2>/dev/null    # ожидается: No such file
```

- [ ] **Шаг 5: переписать импорты скриптом**

Создать временный `/tmp/rewrite_imports.py` (после прогона удалить):

```python
"""Переписывает импорты между модулями audit-data на пакетные."""

import re
from pathlib import Path

PKG = Path("/Users/mac/Documents/_My_code/Chromatic/audit-data")
CORE = {p.stem for p in (PKG / "chromatic_research" / "core").glob("*.py")
        if p.stem != "__init__"}
CAMP = {p.stem for p in (PKG / "chromatic_research" / "campaigns").glob("*.py")
        if p.stem != "__init__"}
WHERE = {name: "core" for name in CORE} | {name: "campaigns" for name in CAMP}

targets = list((PKG / "chromatic_research").rglob("*.py")) + list((PKG / "tests").glob("*.py"))

for path in targets:
    text = original = path.read_text()
    for name, sub in sorted(WHERE.items(), key=lambda kv: -len(kv[0])):
        # from X import a, b   ->  from chromatic_research.<sub>.X import a, b
        text = re.sub(rf"^(\s*)from {name} import ",
                      rf"\1from chromatic_research.{sub}.{name} import ",
                      text, flags=re.M)
        # import X as Y        ->  from chromatic_research.<sub> import X as Y
        text = re.sub(rf"^(\s*)import {name} as ",
                      rf"\1from chromatic_research.{sub} import {name} as ",
                      text, flags=re.M)
        # import X             ->  from chromatic_research.<sub> import X
        text = re.sub(rf"^(\s*)import {name}\s*$",
                      rf"\1from chromatic_research.{sub} import {name}",
                      text, flags=re.M)
    if text != original:
        path.write_text(text)
        print("переписан:", path.relative_to(PKG))
```

```bash
.venv/bin/python /tmp/rewrite_imports.py
```

Скрипт не ловит групповую форму `import a, b` — проверить её отдельно и
переписать руками:

```bash
grep -rnE "^\s*import [a-z_0-9]+, " audit-data/chromatic_research audit-data/tests
```

- [ ] **Шаг 6: убрать `sys.path`-хаки**

```bash
cd /Users/mac/Documents/_My_code/Chromatic/audit-data
grep -rln "sys.path" chromatic_research tests
```

В каждом найденном файле удалить блок вида

```python
sys.path.insert(0, "/Users/mac/Documents/_My_code/Chromatic/audit-data/hd-2026-07")
```

и вариант из `prime_radon.py`:

```python
HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
for path in (HERE, AUDIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
```

После удаления убрать ставший ненужным `import sys`, если он больше не используется
(проверять `grep -n "sys\." <файл>`).

Контроль:

```bash
grep -rn "/Users/mac" --include='*.py' . | wc -l    # ожидается 0
```

- [ ] **Шаг 7: тесты читают json через `paths`**

21 тест читает артефакты как `(HERE / name).read_text()`, где `HERE` — каталог
теста. После переезда каталог другой. В каждом тесте заменить

```python
HERE = Path(__file__).resolve().parent


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text())
```

на

```python
from chromatic_research.paths import load_json


def _load(name: str) -> dict:
    return load_json(name)
```

Найти все такие места:

```bash
grep -rn "Path(__file__)" audit-data/tests/*.py
```

- [ ] **Шаг 8: smoke-тест ловит всё, что сломалось**

```bash
.venv/bin/python -m pytest audit-data/tests/test_all_modules_import.py -q
```

Ожидается: `119 passed` (1 `paths` + 15 модулей `core` + 103 `campaigns`). Каждый
упавший модуль чинить точечно — как правило это импорт, который скрипт не выносил
на верхний уровень.

- [ ] **Шаг 9: полный прогон**

```bash
make test                                    # 332 passed (213 + 119 smoke)
```

- [ ] **Шаг 10: обновить документацию запуска**

В `audit-data/README.md` и `audit-data/hd-2026-07/README.md` заменить примеры
вида `python cert_generic.py` и
`.venv/bin/python audit-data/hd-2026-07/verify_metric_candidate.py ...` на

```bash
.venv/bin/python -m chromatic_research.campaigns.cert_generic
.venv/bin/python -m chromatic_research.campaigns.verify_metric_candidate ...
```

Добавить в начало `audit-data/README.md`:

```markdown
## Запуск

Пакет ставится один раз: `make install` (или `pip install -e audit-data`).
Дальше любая кампания запускается из любого каталога:

    python -m chromatic_research.campaigns.<имя> [аргументы]

Общие модули — в `chromatic_research/core/`, они импортируются, а не запускаются.
```

- [ ] **Шаг 11: сузить `testpaths` и включить audit-data в CI**

В корневом `pyproject.toml` заменить `"audit-data"` на `"audit-data/tests"` —
теперь тесты лежат в отдельном каталоге, и pytest не должен обходить
`chromatic_research/` и `runs/`:

```toml
testpaths = [
    "voronoi/tests",
    "chromatic/tests",
    "combigeo/tests/python",
    "audit-data/tests",
]
```

В `.github/workflows/ci.yml` добавить в шаг установки строку

```yaml
          pip install -e 'audit-data[solvers]'
```

и вернуть полный прогон:

```yaml
      - name: Тесты
        run: pytest -q
```

- [ ] **Шаг 12: коммит**

```bash
make test                                    # 332 passed
git add -A audit-data pyproject.toml .github
git commit -m "audit-data: 118 модулей в пакет chromatic_research, ноль абсолютных путей"
rm /tmp/rewrite_imports.py
```

---

## Задача 12: тест-сторож против абсолютных путей

Чтобы проблема не вернулась со следующей кампанией.

**Файлы:**
- Создать: `audit-data/tests/test_no_absolute_paths.py`

- [ ] **Шаг 1: написать тест**

```python
"""В репозитории не должно быть путей, привязанных к машине автора."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "journal"}
FORBIDDEN = ("/Users/", "/home/", "C:\\")


def test_no_machine_specific_paths_in_sources():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        if path == SELF or any(part in SKIP_DIRS for part in path.parts):
            continue
        text = path.read_text(errors="ignore")
        for marker in FORBIDDEN:
            if marker in text:
                offenders.append(f"{path.relative_to(ROOT)}: {marker}")

    assert not offenders, (
        "машинно-зависимые пути (используйте chromatic_research.paths):\n  "
        + "\n  ".join(offenders)
    )
```

Тест исключает сам себя (иначе строки из `FORBIDDEN` считались бы нарушением) и
каталог `journal/` — там исторические отчёты, которые правке не подлежат.

- [ ] **Шаг 2: запустить**

```bash
.venv/bin/python -m pytest audit-data/tests/test_no_absolute_paths.py -q
```

Ожидается: всё зелено. Если что-то падает — это остаток задачи 11, починить.

- [ ] **Шаг 3: коммит**

```bash
git add audit-data/tests/test_no_absolute_paths.py
git commit -m "тесты: запрет машинно-зависимых путей в исходниках"
```

---

# Фаза 4. Дедупликация

## Задача 13: общий модуль `forms.py`

`unpack`/`pack` (холецкая параметризация нормированной формы Грама) скопированы в
13 файлов, причём с разъезжающимися допусками: `1e-12` в `n6_push45.py:14`,
`1e-10` в `o2_r3.py:10`. `norm_gram` — 10 копий в трёх вариантах (жёстко зашитое
`** 0.25` для n = 4 против общего `** (1/n)`).

**Файлы:**
- Создать: `audit-data/chromatic_research/forms.py`
- Создать: `audit-data/tests/test_forms.py`
- Изменить: 13 модулей в `chromatic_research/campaigns/` (список в шаге 4)

**Интерфейс:**
- `norm_gram(basis: np.ndarray) -> np.ndarray` — Gram-матрица базиса, нормированная на det = 1
- `pack(form: np.ndarray) -> np.ndarray` — нижний треугольник холецкого разложения
- `unpack(vector: np.ndarray, dim: int, tol: float = 1e-12) -> np.ndarray | None` — обратно; `None` для вырожденной формы

- [ ] **Шаг 1: написать тест до реализации**

Создать `audit-data/tests/test_forms.py`:

```python
"""Параметризация форм Грама: одна реализация вместо тринадцати копий."""

import numpy as np
import pytest

from chromatic_research.forms import norm_gram, pack, unpack


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_norm_gram_has_unit_determinant(dim):
    rng = np.random.default_rng(dim)
    basis = rng.normal(size=(dim, dim)) + dim * np.eye(dim)
    assert np.linalg.det(norm_gram(basis)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_pack_unpack_roundtrip(dim):
    rng = np.random.default_rng(100 + dim)
    basis = rng.normal(size=(dim, dim)) + dim * np.eye(dim)
    form = norm_gram(basis)
    assert unpack(pack(form), dim) == pytest.approx(form, abs=1e-12)


def test_pack_length_is_triangular_number():
    form = norm_gram(np.eye(4))
    assert pack(form).shape == (10,)     # 4*5/2


def test_unpack_rejects_degenerate_form():
    assert unpack(np.zeros(10), 4) is None


def test_d4_gram_matches_known_value():
    d4 = np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float)
    form = norm_gram(d4)
    # det(D4 basis) = 2, det(Gram) = 4, нормировка делит на 4**(1/4) на каждую ось
    assert np.linalg.det(form) == pytest.approx(1.0, rel=1e-12)
    assert form == pytest.approx(form.T, abs=1e-15)
```

- [ ] **Шаг 2: убедиться, что тест падает**

```bash
.venv/bin/python -m pytest audit-data/tests/test_forms.py -q
```

Ожидается: `ModuleNotFoundError: chromatic_research.forms`.

- [ ] **Шаг 3: реализовать `audit-data/chromatic_research/forms.py`**

```python
"""Cholesky parametrization of normalized Gram forms.

Optimizers search over the lower triangle of a Cholesky factor; the resulting
form is normalized to unit determinant so that the objective is scale-free.
This replaces thirteen near-identical copies that had drifted apart in both
tolerance (1e-10 vs 1e-12) and dimension handling (hard-coded ``** 0.25``).
"""

from __future__ import annotations

import numpy as np


def norm_gram(basis: np.ndarray) -> np.ndarray:
    """Gram matrix of `basis`, scaled to determinant 1."""
    basis = np.asarray(basis, dtype=float)
    gram = basis @ basis.T
    dim = gram.shape[0]
    return gram / abs(np.linalg.det(gram)) ** (1.0 / dim)


def pack(form: np.ndarray) -> np.ndarray:
    """Lower triangle of the Cholesky factor of `form`, as a flat vector."""
    form = np.asarray(form, dtype=float)
    dim = form.shape[0]
    return np.linalg.cholesky(form)[np.tril_indices(dim)]


def unpack(vector: np.ndarray, dim: int, tol: float = 1e-12) -> np.ndarray | None:
    """Inverse of :func:`pack`; ``None`` when the resulting form is degenerate."""
    factor = np.zeros((dim, dim))
    factor[np.tril_indices(dim)] = np.asarray(vector, dtype=float)
    form = factor @ factor.T
    determinant = abs(np.linalg.det(form))
    if determinant <= tol:
        return None
    return form / determinant ** (1.0 / dim)
```

- [ ] **Шаг 4: тест зелёный**

```bash
.venv/bin/python -m pytest audit-data/tests/test_forms.py -q     # 11 passed
```

- [ ] **Шаг 5: заменить копии**

Найти их:

```bash
grep -rln "^def unpack\|^def norm_gram\|^def pack" audit-data/chromatic_research/campaigns/
```

Ожидается 13-15 файлов, среди них `n1_r3_full.py`, `n2_4d_frontier.py`,
`n4_push46.py`, `n6_push45.py`, `n7_push44.py`, `n8_cma44_ladder.py`,
`n10_push44.py`, `o1_widths4d.py`, `o2_r3.py`, `r5_push48.py`, `cma_form.py`,
`csp_campaign5d.py`, `csp_sweep5d.py`, `mc_attack5d.py`.

В каждом удалить локальные определения и добавить в шапку:

```python
from chromatic_research.forms import norm_gram, pack, unpack
```

**Осторожно с двумя различиями поведения:**
1. локальные `unpack` берут размерность из замыкания (`np.zeros((4, 4))`) — новая
   требует явного `dim`. Вызовы `unpack(x)` заменить на `unpack(x, 4)` (или на ту
   размерность, что стоит в файле).
2. локальные `norm_gram(M, n)` в `csp_sweep5d.py`, `mc_attack5d.py`,
   `csp_campaign5d.py` принимают размерность вторым аргументом — она равна
   `M.shape[0]`, поэтому вызовы становятся `norm_gram(M)`.

- [ ] **Шаг 6: проверить, что численные результаты не поехали**

Прогнать самую дешёвую из затронутых кампаний и сверить с сохранённым результатом:

```bash
.venv/bin/python -m pytest audit-data/tests -q       # все тесты audit-data
make test                                            # полный прогон
```

Дополнительно — точечная сверка на опорном значении:

```bash
.venv/bin/python -c "
import numpy as np
from chromatic_research.forms import norm_gram, pack, unpack
d4 = np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float)
form = norm_gram(d4)
assert np.allclose(unpack(pack(form), 4), form, atol=1e-13)
print('форма D4 воспроизводится:', np.linalg.det(form))
"
```

- [ ] **Шаг 7: коммит**

```bash
git add audit-data
git commit -m "audit-data: единая параметризация форм Грама вместо 13 копий"
```

---

## Задача 14: убрать единственное настоящее дублирование решётки

**Уточнение к аудиту.** Первоначальная оценка «решётки задаются заново в четырёх
местах» при проверке не подтвердилась. Фактически:

- `symlat.An_star_ambient(n)` строит A*ₙ в **объемлющих** координатах ℝⁿ⁺¹ и без
  нормировки — это другое представление, а не копия. Оставить как есть.
- `permutohedral_cover` содержит собственный **точный** радиус покрытия
  перестановочного многогранника (`covering_radius`, `radii_for_orders`), тогда
  как `core/covrad.covering_radius` — общий численный по опорным полупространствам.
  Разные алгоритмы с разными гарантиями. Оставить оба.
- `n3_5d_probe.py:13-18` действительно строит A₅* заново. Проверено численно:
  совпадает с `lattices.Astar(5)` — одинаковые `diam = 1.668064710953` и
  `λ₁ = 1.092004686004` (12 знаков). Это и есть предмет задачи.

**Файлы:**
- Изменить: `audit-data/chromatic_research/campaigns/n3_5d_probe.py:13-18`
- Создать: `audit-data/tests/test_lattices.py`

- [ ] **Шаг 1: тест, фиксирующий инварианты решёток**

Создать `audit-data/tests/test_lattices.py`:

```python
"""Инварианты стандартных решёток из core.lattices."""

import numpy as np
import pytest

from chromatic_research.core import lattices


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 6])
def test_a_star_has_unit_covolume(dim):
    assert abs(np.linalg.det(lattices.Astar(dim))) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("dim", [4, 5, 6, 8])
def test_d_lattice_has_unit_covolume(dim):
    assert abs(np.linalg.det(lattices.D(dim))) == pytest.approx(1.0, rel=1e-12)


def test_a5_star_invariants_are_stable():
    """Опорные значения A5* — страховка для задачи о дублировании конструкции."""
    import combigeo

    basis = lattices.Astar(5)
    cell = combigeo.voronoi_cell(basis.tolist())
    shortest = combigeo.shortest_vector(basis.tolist())

    assert cell.diameter == pytest.approx(1.668064710953, abs=1e-11)
    assert float(np.linalg.norm(shortest)) == pytest.approx(1.092004686004, abs=1e-11)
```

- [ ] **Шаг 2: запустить — должен пройти сразу**

```bash
.venv/bin/python -m pytest audit-data/tests/test_lattices.py -q    # 10 passed
```

Это характеризационный тест: он фиксирует значения ДО правки.

- [ ] **Шаг 3: заменить локальную конструкцию A₅***

В `audit-data/chromatic_research/campaigns/n3_5d_probe.py` строки 13-18:

```python
n, K = 5, 139
M = np.ones((n + 1, n), float)
for j in range(n):
    M[j, j] = -n
A5S = np.linalg.cholesky(M.T @ M)
A5S /= abs(np.linalg.det(A5S)) ** (1 / n)
```

заменить на:

```python
from chromatic_research.core import lattices

n, K = 5, 139
A5S = lattices.Astar(n)
```

- [ ] **Шаг 4: убедиться, что решётка та же**

```bash
.venv/bin/python - <<'PY'
import numpy as np, combigeo
from chromatic_research.campaigns.n3_5d_probe import A5S
cell = combigeo.voronoi_cell(A5S.tolist())
print("diam =", cell.diameter, "(ожидается 1.668064710953)")
assert abs(cell.diameter - 1.668064710953) < 1e-11
PY
```

Импорт модуля не должен запускать кампанию — гарантировано задачей 9.

- [ ] **Шаг 5: проверка и коммит**

```bash
make test
git add audit-data
git commit -m "audit-data: A5* берётся из core.lattices вместо локальной копии"
```

---

# Фаза 5. Данные

## Задача 15: разделить `results/` и `runs/`, сжать сырые прогоны

Измерено: из 662 json в `audit-data` на 125 (35.4 МБ) ссылаются доки, статья, код
или тесты; на 537 (187.4 МБ) — не ссылается ничто. Сырые выгрузки сжимаются в 13.2
раза (176 МБ → 13 МБ на выборке из 20 крупнейших). Чекаут сократится с 227 МБ
примерно до 50 МБ.

`.git` при этом подрастёт (сжатые blob-ы git уже не ужмёт) примерно на 15 МБ —
это осознанная плата за лёгкий рабочий каталог; история не переписывается.

**Файлы:**
- Создать: `audit-data/results/` (125 файлов), `audit-data/runs/` (537 `.json.gz`)
- Создать: `audit-data/runs/MANIFEST.md`
- Изменить: `paper/figures.py:20`
- Изменить: `audit-data/README.md`, `audit-data/README-dim5-9.md`

- [ ] **Шаг 1: получить точные списки**

Создать `/tmp/classify_json.py`:

```python
"""Классификация json: на какие ссылается код/доки (включая f-строки)."""
import re
import sys
from pathlib import Path

ROOT = Path("/Users/mac/Documents/_My_code/Chromatic")
SCAN_EXT = {".py", ".md", ".tex", ".sh"}
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".cache"}

artifacts = sorted(p for p in ROOT.glob("audit-data/**/*.json")
                   if not any(s in p.parts for s in SKIP_DIRS))

STR_CHUNK = re.compile(r"[A-Za-z0-9_./{}\[\]()+-]*\.json")
patterns, literals = [], set()
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix not in SCAN_EXT:
        continue
    if any(s in path.parts for s in SKIP_DIRS):
        continue
    for chunk in STR_CHUNK.findall(path.read_text(errors="ignore")):
        name = chunk.rsplit("/", 1)[-1]
        if "{" in name:
            body = re.sub(r"\\\{[^}]*\\\}", "[^/]+", re.escape(name))
            patterns.append(re.compile("^" + body + "$"))
        else:
            literals.add(name)

referenced, orphan = [], []
for art in artifacts:
    (referenced if art.name in literals or any(rx.match(art.name) for rx in patterns)
     else orphan).append(art)

Path(sys.argv[1]).write_text("\n".join(str(p.relative_to(ROOT)) for p in referenced) + "\n")
Path(sys.argv[2]).write_text("\n".join(str(p.relative_to(ROOT)) for p in orphan) + "\n")
mb = lambda ps: sum(p.stat().st_size for p in ps) / 2**20
print(f"опорных {len(referenced)} ({mb(referenced):.1f} МБ), "
      f"сырых {len(orphan)} ({mb(orphan):.1f} МБ)")
```

```bash
.venv/bin/python /tmp/classify_json.py /tmp/referenced.txt /tmp/orphans.txt
```

Ожидается примерно: `опорных 125 (35.4 МБ), сырых 537 (187.4 МБ)`. Числа могут
слегка отличаться после фазы 3 — это нормально, важно, что порядок тот же.

- [ ] **Шаг 2: перенести опорные в `results/`**

```bash
cd /Users/mac/Documents/_My_code/Chromatic
mkdir -p audit-data/results
while read -r f; do
  [ -n "$f" ] && git mv "$f" "audit-data/results/$(basename "$f")"
done < /tmp/referenced.txt
ls audit-data/results | wc -l     # ожидается 125
```

- [ ] **Шаг 3: перенести и сжать сырые**

```bash
cd /Users/mac/Documents/_My_code/Chromatic
mkdir -p audit-data/runs
while read -r f; do
  [ -n "$f" ] || continue
  base=$(basename "$f")
  gzip -9 -c "$f" > "audit-data/runs/$base.gz"
  git rm -q "$f"
done < /tmp/orphans.txt
git add audit-data/runs
du -sh audit-data/runs        # ожидается около 14 МБ
```

- [ ] **Шаг 4: проверить, что данные читаются обратно**

```bash
.venv/bin/python - <<'PY'
import gzip, json, random
from pathlib import Path
runs = sorted(Path("audit-data/runs").glob("*.json.gz"))
print("файлов:", len(runs))
for path in random.Random(0).sample(runs, 10):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        json.load(handle)
print("10 случайных файлов распакованы и разобраны")
PY
```

- [ ] **Шаг 5: обновить `paper/figures.py`**

Строка 20:

```python
DATA = str(_HERE.parent / "audit-data")   # ../audit-data
```

заменить на:

```python
DATA = str(_HERE.parent / "audit-data" / "results")   # ../audit-data/results
```

Проверить:

```bash
make figures
git status --porcelain paper/*.pdf      # рисунки пересобрались
```

Затем сверить, что PDF рисунков не изменились по содержанию (они пересобираются
из тех же данных — допустимо различие только в метке времени внутри PDF).

- [ ] **Шаг 6: написать `audit-data/runs/MANIFEST.md`**

```markdown
# runs/ — сырые выгрузки кампаний

Здесь лежат результаты прогонов, на которые не ссылаются ни статья, ни README,
ни тесты: разведочные запуски, промежуточные стадии, перебор параметров.
Они сохранены как воспроизводимый след работы, а не как опорные данные.

- формат: тот же json, что и раньше, сжатый gzip (`*.json.gz`, около ×13);
- чтение из кода: `chromatic_research.paths.load_json("<имя>.json")` —
  распаковывает прозрачно, имя указывается без `.gz`;
- чтение из оболочки: `gzcat audit-data/runs/<имя>.json.gz | jq .`;
- опорные данные (те, на которые ссылаются статья, доки, тесты) — в `../results/`.

Если файл отсюда понадобился статье или тесту — перенесите его в `results/`
разжатым: `gzcat runs/X.json.gz > results/X.json && git rm runs/X.json.gz`.
```

- [ ] **Шаг 7: обновить README каталога**

В `audit-data/README.md` заменить упоминания путей вида
`hd-2026-07/metric_deform_e7_1323_certificate.json` на
`results/metric_deform_e7_1323_certificate.json` и добавить абзац:

```markdown
## Где что лежит

| Каталог | Что |
|---|---|
| `chromatic_research/core/` | Модули, которыми пользуются несколько кампаний. |
| `chromatic_research/campaigns/` | Отдельные кампании; запуск `python -m chromatic_research.campaigns.<имя>`. |
| `tests/` | Тесты (`make test` гоняет их вместе с остальными). |
| `results/` | Опорные данные: на них ссылаются статья, README и тесты. |
| `runs/` | Сырые выгрузки прогонов, gzip; см. `runs/MANIFEST.md`. |
```

- [ ] **Шаг 8: полная проверка**

```bash
make test                     # все тесты, включая 21 читающий json
du -sh audit-data             # было 228 МБ, ожидается около 50 МБ
```

- [ ] **Шаг 9: коммит**

```bash
git add -A audit-data paper/figures.py
git commit -m "данные: results/ и runs/ раздельно, сырые прогоны сжаты (227 МБ -> 50 МБ)"
rm /tmp/classify_json.py /tmp/referenced.txt /tmp/orphans.txt
```

---

## Задача 16: тест-сторож против гниения ссылок в документации

**Файлы:**
- Создать: `audit-data/tests/test_docs_references_exist.py`

- [ ] **Шаг 1: написать тест**

```python
"""Каждый json, упомянутый в README, должен существовать."""

import re

from chromatic_research.paths import AUDIT_DIR, find_artifact

DOCS = sorted(AUDIT_DIR.glob("*.md")) + [
    AUDIT_DIR.parent / "README.md",
    AUDIT_DIR.parent / "RESULTS.md",
]
NAME = re.compile(r"`([A-Za-z0-9_./-]+\.json)`")


def test_every_mentioned_artifact_exists():
    missing = []
    for doc in DOCS:
        if not doc.exists():
            continue
        for match in NAME.finditer(doc.read_text(encoding="utf-8")):
            name = match.group(1).rsplit("/", 1)[-1]
            try:
                find_artifact(name)
            except FileNotFoundError:
                missing.append(f"{doc.name}: {name}")

    assert not missing, "битые ссылки на данные:\n  " + "\n  ".join(sorted(set(missing)))
```

Регулярка ловит имена в обратных кавычках — так они записаны во всех README
проекта. Имена с фигурными скобками (`..._{certificate,audit}.json` в
`README-dim5-9.md`) под неё не подпадают и проверяются глазами.

- [ ] **Шаг 2: запустить и починить найденное**

```bash
.venv/bin/python -m pytest audit-data/tests/test_docs_references_exist.py -q
```

Каждое падение — реальная битая ссылка в README. Варианты: файл переименован
(поправить README), файл удалён (убрать упоминание), файл в `runs/` (это норма,
`find_artifact` его найдёт).

- [ ] **Шаг 3: коммит**

```bash
git add audit-data/tests/test_docs_references_exist.py audit-data/*.md
git commit -m "тесты: сторож против битых ссылок на данные в README"
```

---

# Фаза 6. Материалы

## Задача 17: разгрузить корневой README

Сейчас 198 строк смешивают точку входа и журнал кампаний: строки 45-59 и 155-194 —
плотная хроника с числами вроде `0.984244122202` и `296 765 305`, которая нужна
рецензенту, но не человеку, впервые открывшему репозиторий.

**Файлы:**
- Изменить: `README.md`
- Создать: `RESULTS.md`

- [ ] **Шаг 1: создать `RESULTS.md`**

Перенести в него без изменения текста:
- абзац про кампании из `README.md:45-59` (determinant-aware exact-repair и далее);
- блок про размерности 6, 8, 9 из `README.md:155-194`.

Начать файл шапкой:

```markdown
# Хроника вычислительных кампаний

Подробный отчёт о том, что и как проверялось: границы, достигнутые численно,
отрицательные экраны и их честный статус. Итоговые доказанные оценки — в
[README.md](README.md), полное изложение — в статье
[`paper/chi4-45.tex`](paper/chi4-45.tex).

Все результаты, не снабжённые точным сертификатом, — вычислительные наблюдения,
а не доказательства невозможности.
```

- [ ] **Шаг 2: сократить `README.md`**

На месте вырезанных блоков оставить по одной ссылке:

```markdown
Подробная хроника кампаний, численные фронтиры и отрицательные экраны —
в [RESULTS.md](RESULTS.md).
```

Раздел «Статья и результаты» сократить до пяти пунктов (χ(ℝ⁴) ≤ 45,
χ(ℝ⁵) ≤ 132, χ(ℝ⁷) ≤ 1323, кампания «ниже 45», точные ширины АБПР) без
детализации методов.

- [ ] **Шаг 3: обновить структурную схему в README**

Заменить блок со схемой каталогов на актуальный (см. «Целевая структура» этого
плана), добавив `audit-data/results`, `audit-data/runs`, `journal/`, `RESULTS.md`.

- [ ] **Шаг 4: проверить ссылки**

```bash
grep -oE '\[[^]]+\]\(([^)]+)\)' README.md | sed -E 's/.*\((.*)\)/\1/' | \
  grep -v '^http' | while read -r link; do
    [ -e "${link%%#*}" ] || echo "битая ссылка: $link"
  done
```

Ожидается: пусто.

- [ ] **Шаг 5: коммит**

```bash
git add README.md RESULTS.md
git commit -m "материалы: README — точка входа, хроника кампаний в RESULTS.md"
```

---

## Задача 18: расслоить документацию на постоянную и датированную

5956 строк markdown в 19 файлах с перекрытием; датированные отчёты лежат вперемешку
с справочными.

**Файлы:**
- Создать: `journal/`
- Переместить: `archive/AUDIT-2026-07-21.md`, `archive/PLAN-2026-07-21.md`,
  `archive/RESULTS-2026-07-21.md`, `paper/AUDIT-2026-08-05.md`,
  `paper/PLAN-podacha-2026-08-05.md`, `audit-data/hd-2026-07/RESEARCH_2026-07-29.md`,
  `audit-data/hd-2026-07/RESEARCH_2026-07-30.md` → `journal/`
- Переместить: `audit-data/hd-2026-07/README.md` → `audit-data/README-dim5-9.md`
- Переместить: `audit-data/hd-2026-07/NEXT_MECHANISM.md` → `audit-data/NEXT_MECHANISM.md`
- Создать: `journal/README.md`
- Удалить: пустой `archive/`, пустой `audit-data/hd-2026-07/`

- [ ] **Шаг 1: перенести**

```bash
cd /Users/mac/Documents/_My_code/Chromatic
mkdir -p journal
git mv archive/AUDIT-2026-07-21.md journal/
git mv archive/PLAN-2026-07-21.md journal/
git mv archive/RESULTS-2026-07-21.md journal/
git mv paper/AUDIT-2026-08-05.md journal/
git mv paper/PLAN-podacha-2026-08-05.md journal/
git mv audit-data/hd-2026-07/RESEARCH_2026-07-29.md journal/
git mv audit-data/hd-2026-07/RESEARCH_2026-07-30.md journal/
git mv audit-data/hd-2026-07/README.md audit-data/README-dim5-9.md
git mv audit-data/hd-2026-07/NEXT_MECHANISM.md audit-data/NEXT_MECHANISM.md
rmdir archive audit-data/hd-2026-07 2>/dev/null || ls archive audit-data/hd-2026-07
```

Если `rmdir` не сработал — посмотреть, что осталось, и разобрать вручную.

- [ ] **Шаг 2: создать `journal/README.md`**

```markdown
# journal/ — датированные документы

Отчёты, планы и аудиты, привязанные к моменту времени. Они **не обновляются**:
каждый описывает состояние дел на свою дату и остаётся как след процесса.

| Файл | Дата | Что это |
|---|---|---|
| `AUDIT-2026-07-21.md` | 21.07.2026 | Аудит кода трёх пакетов (предшествует результату χ(ℝ⁴) ≤ 45). |
| `PLAN-2026-07-21.md` | 21.07.2026 | План доработок до версии 1.1.0. |
| `RESULTS-2026-07-21.md` | 22.07.2026 | Промежуточный отчёт (называет рекордом 46/48). |
| `RESEARCH_2026-07-29.md` | 29.07.2026 | Кампании больших размерностей. |
| `RESEARCH_2026-07-30.md` | 30.07.2026 | 342-цветная ветвь ℝ⁶, честные статусы экранов. |
| `AUDIT-2026-08-05.md` | 05.08.2026 | Аудит статьи перед подачей. |
| `PLAN-podacha-2026-08-05.md` | 05.08.2026 | План подачи статьи. |

Актуальное состояние — в корневых [README.md](../README.md) и
[RESULTS.md](../RESULTS.md), справочник по кампаниям n ≥ 5 —
в [audit-data/README-dim5-9.md](../audit-data/README-dim5-9.md).
```

- [ ] **Шаг 3: починить ссылки на переехавшие файлы**

```bash
grep -rn "archive/\|hd-2026-07/\|paper/AUDIT-2026-08-05\|paper/PLAN-podacha" \
  --include='*.md' --include='*.tex' --include='*.py' . | grep -v .venv | grep -v journal/README
```

Каждую найденную ссылку поправить на новый путь.

- [ ] **Шаг 4: проверка и коммит**

```bash
make test
git add -A
git commit -m "материалы: датированные отчёты в journal/, справочники — рядом с данными"
```

---

## Задача 19: разбить статью на секции

`paper/chi4-45.tex` — 1771 строка и 134 КБ в одном файле (9 секций, 19 подсекций).
Механизм уже используется: строка 1336 — `\input{origin-and-ai}`.

**Файлы:**
- Создать: `paper/sections/intro.tex`, `method.tex`, `algo.tex`, `main.tex`,
  `dim57.tex`, `intervals.tex`, `extra.tex`, `open.tex`, `screens.tex`
- Изменить: `paper/chi4-45.tex` (остаются преамбула, титул, `\input`, библиография)
- Изменить: `paper/README.md`

- [ ] **Шаг 1: зафиксировать эталон PDF**

```bash
cd paper && latexmk -pdf chi4-45.tex >/dev/null 2>&1
.venv/bin/python -c "
import hashlib, subprocess
# сравнивать будем не байты (в PDF есть метка времени), а извлечённый текст
text = subprocess.run(['pdftotext', 'chi4-45.pdf', '-'], capture_output=True, text=True).stdout
open('/tmp/paper-before.txt', 'w').write(text)
print('страниц текста:', text.count(chr(12)) + 1, 'символов:', len(text))
"
```

Если `pdftotext` недоступен — поставить (`brew install poppler`) или сравнивать
по числу страниц из `pdfinfo`.

- [ ] **Шаг 2: нарезать по границам секций**

Границы (номера строк на 2026-08-05):

| Файл | Строки | Секция |
|---|---|---|
| `sections/intro.tex` | 85-209 | Введение |
| `sections/method.tex` | 210-366 | Метод: раскраска по решёткам Вороного |
| `sections/algo.tex` | 367-571 | Алгоритм и его реализация |
| `sections/main.tex` | 572-702 | Размерность 4: χ(ℝ⁴) ≤ 45 |
| `sections/dim57.tex` | 703-984 | Размерности 5 и 7 |
| `sections/intervals.tex` | 985-1160 | Интервальные версии |
| `sections/extra.tex` | 1161-1240 | Полная картина: точные ширины АБПР |
| `sections/open.tex` | 1241-1339 | Границы метода и открытые вопросы |
| `sections/screens.tex` | 1340-конец секций | Отрицательные экраны в ℝ⁵-ℝ⁹ |

Перед нарезкой перепроверить номера — они могли сдвинуться:

```bash
grep -n '^\\section' paper/chi4-45.tex
```

Существующий `\input{origin-and-ai}` на строке 1336 попадает внутрь `open.tex` —
перенести его в `chi4-45.tex`, в общий список `\input`, чтобы все включения были
в одном месте.

- [ ] **Шаг 3: собрать `chi4-45.tex` из включений**

После преамбулы и `\maketitle` оставить:

```latex
\input{sections/intro}
\input{sections/method}
\input{sections/algo}
\input{sections/main}
\input{sections/dim57}
\input{sections/intervals}
\input{sections/extra}
\input{sections/open}
\input{origin-and-ai}
\input{sections/screens}
```

Порядок обязан совпадать с исходным: `origin-and-ai` шёл внутри секции
«Границы метода» — если он был подразделом, поставить его строго туда же
относительно текста.

- [ ] **Шаг 4: пересобрать и сверить текст**

```bash
cd paper && latexmk -pdf chi4-45.tex >/dev/null 2>&1
pdftotext chi4-45.pdf - > /tmp/paper-after.txt
diff /tmp/paper-before.txt /tmp/paper-after.txt && echo "текст статьи не изменился"
```

Ожидается: `diff` молчит. Любое расхождение — ошибка нарезки, чинить до нуля
различий.

- [ ] **Шаг 5: обновить `paper/README.md`**

В таблицу «Что внутри» добавить строку:

```markdown
| `sections/*.tex` | Девять секций статьи; `chi4-45.tex` содержит преамбулу и `\input`. |
```

- [ ] **Шаг 6: коммит**

```bash
git add paper
git commit -m "статья: разбить chi4-45.tex на секции (текст PDF не изменился)"
```

---

# Фаза 7. Необязательное продолжение

## Задача 20: векторизовать `dist_to_s`

После задачи 7 около 70% времени — сам питоновский тройной цикл: на ячейке D₄ это
24 гиперграни × 192 2D-грани × 576 рёбер, примерно 1317 вычислений расстояния на
точку. Уровни каскада вычисляются numpy-массивами целиком: `scipy.spatial.Delaunay.find_simplex`
принимает массив точек, то есть 576 отдельных вызовов заменяются одним.

**Обоснование корректности:** текущий код обрывает спуск, когда проекция на грань
попала внутрь многогранника. Все кандидаты, которые при этом не рассматриваются, —
точки той же грани или её границы, то есть не ближе найденной. Значит, вычисление
всех уровней без обрыва даёт тот же минимум. Ранний выход (`early_stop`) по
контракту возвращает верхнюю оценку, поэтому его ослабление решений
`find_optimal` не меняет.

**Файлы:**
- Изменить: `voronoi/src/voronoi4d/polyhedra.py` (сборка плоских массивов в `build`)
- Изменить: `voronoi/src/voronoi4d/distances.py` (векторизованный `dist_to_s`)
- Создать: `voronoi/tests/test_distances_vectorized.py`

- [ ] **Шаг 1: тест эквивалентности против текущей реализации**

```python
"""Векторизованный dist_to_s обязан совпадать с каскадным на всех решётках."""

import numpy as np
import pytest

from voronoi4d import VoronoiPolyhedra, dist_to_s
from voronoi4d.distances import dist_to_s_cascade   # старая реализация, оставлена для сверки

LATTICES = {
    "Z4": np.eye(4),
    "D4": np.array([[2, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], float),
    "skew": np.array([[1.3, 0.2, 0, 0], [0.1, 1.1, 0.3, 0],
                      [0, 0.2, 1.2, 0.1], [0.1, 0, 0.2, 0.9]], float),
}


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_vectorized_matches_cascade(name):
    vor = VoronoiPolyhedra(LATTICES[name])
    vor.build(verbose=False)
    rng = np.random.default_rng(7)

    for _ in range(50):
        point = rng.normal(size=4) * 1.5
        fast = dist_to_s(vor, point, vor.max_len, early_stop=0.0)
        slow = dist_to_s_cascade(vor, point, vor.max_len, early_stop=0.0)
        assert fast == pytest.approx(slow, abs=1e-12)
```

- [ ] **Шаг 2: сохранить текущую реализацию под именем `dist_to_s_cascade`**

Переименовать существующую функцию в `dist_to_s_cascade` (оставить как эталон
для сверки, экспортировать из модуля, но НЕ из `__init__`), а `dist_to_s` писать
заново.

- [ ] **Шаг 3: собрать плоские массивы при построении ячейки**

В `VoronoiPolyhedra.build` после `create_polyhedrons()` добавить:

```python
        self._pack_face_arrays()
```

и метод:

```python
    def _pack_face_arrays(self):
        """Плоские массивы граней всех уровней — для векторизованного dist_to_s."""
        self.face3_normal = np.array([p.normal for p in self.polyhedrons])
        self.face3_center = np.array([p.center for p in self.polyhedrons])

        face2, parent2 = [], []
        for i, pol in enumerate(self.polyhedrons):
            for face in pol.faces:
                face2.append(face)
                parent2.append(i)
        self.face2_normal = np.array([f.normal for f in face2])
        self.face2_center = np.array([f.center for f in face2])
        self.face2_parent = np.array(parent2)

        edges, parent1 = [], []
        for j, face in enumerate(face2):
            for edge in face.edges:
                edges.append(edge)
                parent1.append(j)
        self.edge_normal = np.array([e.normal for e in edges])
        self.edge_center = np.array([e.center for e in edges])
        self.edge_vertex1 = np.array([e.vertex1 for e in edges])
        self.edge_vertex2 = np.array([e.vertex2 for e in edges])
        self.edge_parent = np.array(parent1)
```

- [ ] **Шаг 4: написать векторизованный `dist_to_s`**

```python
def dist_to_s(vor4, s, max_len, early_stop=1.0, check=True):
    """Нормированное расстояние от точки s до центрального многогранника V0.

    Векторизованный вариант каскада проекций: все грани каждого уровня
    обрабатываются одним numpy-выражением, принадлежность проверяется одним
    вызовом Delaunay.find_simplex на массив точек.  Эталон — dist_to_s_cascade.

    Отличие от каскада: значение всегда точное.  Каскадный вариант при
    early_stop возвращал верхнюю оценку — точное значение её не хуже, поэтому
    решения find_optimal не меняются, а сам параметр здесь не нужен.
    """
    del check       # сверка по теореме Пифагора избыточна: расстояния прямые
    del early_stop  # всегда считаем точно, см. докстринг
    s = np.asarray(s, dtype=float)

    inside = vor4.face3_normal @ s - np.einsum(
        "ij,ij->i", vor4.face3_normal, vor4.face3_center)
    if np.all(inside <= TOL_SIMPLEX):
        return 0.0

    # уровень 3: проекция на каждую гипергрань
    d0 = inside
    coord0 = s - d0[:, None] * vor4.face3_normal
    hit0 = vor4.delaunay.find_simplex(coord0, tol=TOL_SIMPLEX) != -1
    best = np.abs(d0[hit0]).min() if hit0.any() else np.inf

    # уровень 2: проекция на каждую 2-мерную грань
    base1 = coord0[vor4.face2_parent]
    d1 = np.einsum("ij,ij->i", vor4.face2_normal, base1 - vor4.face2_center)
    coord1 = base1 - d1[:, None] * vor4.face2_normal
    hit1 = vor4.delaunay.find_simplex(coord1, tol=TOL_SIMPLEX) != -1
    if hit1.any():
        best = min(best, float(np.linalg.norm(coord1[hit1] - s, axis=1).min()))

    # уровень 1: проекция на каждое ребро, иначе — ближайшая его вершина
    base2 = coord1[vor4.edge_parent]
    d2 = np.einsum("ij,ij->i", vor4.edge_normal, base2 - vor4.edge_center)
    coord2 = base2 - d2[:, None] * vor4.edge_normal
    hit2 = vor4.delaunay.find_simplex(coord2, tol=TOL_SIMPLEX) != -1
    if hit2.any():
        best = min(best, float(np.linalg.norm(coord2[hit2] - s, axis=1).min()))

    outside = ~hit2
    if outside.any():
        to_v1 = np.linalg.norm(coord2[outside] - vor4.edge_vertex1[outside], axis=1)
        to_v2 = np.linalg.norm(coord2[outside] - vor4.edge_vertex2[outside], axis=1)
        nearest = np.where((to_v1 < to_v2)[:, None],
                           vor4.edge_vertex1[outside], vor4.edge_vertex2[outside])
        best = min(best, float(np.linalg.norm(nearest - s, axis=1).min()))

    return float(best * 2 / max_len)
```

- [ ] **Шаг 5: тест эквивалентности зелёный**

```bash
.venv/bin/python -m pytest voronoi/tests/test_distances_vectorized.py -q      # 3 passed
.venv/bin/python -m pytest voronoi/tests/test_distances_characterization.py -q  # 18 passed
make test
```

Если эквивалентность нарушается — НЕ подгонять допуск. Разобраться, какая ветвь
каскада не воспроизведена, и починить логику.

- [ ] **Шаг 6: замерить**

```bash
.venv/bin/python - <<'PY'
import time, numpy as np
from voronoi4d import VoronoiPolyhedra, dist_to_s
from voronoi4d.distances import dist_to_s_cascade
D4 = np.array([[2,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1]], float)
v = VoronoiPolyhedra(D4); v.build(verbose=False)
rng = np.random.default_rng(0); pts = [rng.normal(size=4)*1.2 for _ in range(200)]
for name, fn in (("каскад", dist_to_s_cascade), ("векторный", dist_to_s)):
    t = time.perf_counter()
    for p in pts: fn(v, p, v.max_len, early_stop=0.0)
    print(f"{name:10s}: {time.perf_counter()-t:.3f} с")
PY
```

Если векторный вариант не быстрее каскадного минимум вдвое — оставить каскад
(векторизация не окупилась) и зафиксировать замер в коммите как отрицательный
результат.

- [ ] **Шаг 7: коммит**

```bash
git add voronoi
git commit -m "voronoi4d: векторизованный dist_to_s с эталонной сверкой против каскада"
```

---

## Проверено и отклонено: перебор в `shortest_vector`

В аудите отмечено, что `enumeration.shortest_vector` (`enumeration.py:77`)
перечисляет **все** векторы в шаре радиуса `min |bᵢ|` и лишь потом берёт минимум,
вместо перечисления Шнорра–Эйхнера со сжимающимся радиусом. Замер на 200
подрешётках индекса 45 (LLL-приведённых, как в `find_optimal`):

```
shortest_vector x200: 0.009 с  (0.05 мс/вызов)
перечисляется векторов: медиана 1, максимум 6 (нужен 1)
```

После LLL стартовый радиус уже практически точен, перебирать почти нечего.
Оптимизировать нечего — **задача не ставится**. Записано здесь, чтобы к этому
не возвращались.

---

# Порядок и оценка

| Фаза | Задачи | Риск | Что даёт |
|---|---|---|---|
| 0. Инфраструктура | 1-2 | нет | `make test` = 199 и CI → шлюз для всего остального |
| 1. Чистка | 3 | нет | −28 файлов, нет устаревшего дубля PDF |
| 2. voronoi4d | 4-7 | низкий | −3 мёртвые функции, чистая сигнатура, ×1.4 на горячем пути |
| 3. Пакет | 8-12 | средний | 0 абсолютных путей, запуск с любой машины |
| 4. Дедупликация | 13-14 | средний | −13 копий параметризации форм, A₅* из одного места |
| 5. Данные | 15-16 | средний | чекаут 227 → 50 МБ, разделены опорные и сырые |
| 6. Материалы | 17-19 | низкий | README как точка входа, статья по секциям |
| 7. Векторизация | 20 | высокий | ещё кратное ускорение (может не окупиться) |

Фазы 2, 6 и 7 не зависят от 3-5 и могут идти параллельно или отдельной сессией.
Внутри фазы 3 порядок обязателен: 8 → 9 → 10 → 11 → 12.

**Точки, где стоит остановиться и подумать:**
- задача 11 (перенос 118 модулей) — самый крупный необратимый шаг; делать в
  отдельной ветке и смотреть `git log --stat` перед слиянием;
- задача 15 (сжатие данных) — если предпочтительнее хранить сырые прогоны вне
  репозитория (архив в релизе GitHub), это решается здесь, а не позже;
- задача 20 — единственная, где допустим отрицательный итог: если замер не
  показал двукратного выигрыша, откатываем и оставляем каскад.
