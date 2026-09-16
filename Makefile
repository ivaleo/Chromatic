PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install test lint figures paper clean

help:
	@echo "install  — поставить voronoi4d, combigeo, chromatic и chromatic-research в .venv"
	@echo "test     — все тесты монорепо"
	@echo "lint     — ruff по коду"
	@echo "figures  — пересобрать рисунки статьи из данных"
	@echo "paper    — собрать все PDF: статьи 1–2, сообщение для Докладов, английскую версию"
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
	cd paper/article1 && latexmk -pdf bounds.tex
	cd paper/article2 && latexmk -pdf widths.tex
	cd paper/note-dan && latexmk -pdf dan.tex
	cd paper/arxiv-en && latexmk -pdf bounds-en.tex

clean:
	find . -path ./.venv -prune -o \( -name __pycache__ -type d -prune -exec rm -rf {} + \)
	rm -rf .pytest_cache */.pytest_cache .ruff_cache combigeo/build combigeo/.cache
	find . -path ./.venv -prune -o \( -name '*.egg-info' -type d -prune -exec rm -rf {} + \)
	rm -f combigeo/src/*.so
	find . -path ./.venv -prune -o -name .DS_Store -type f -exec rm -f {} +
	cd paper/article1 && latexmk -c bounds.tex || true
	cd paper/article2 && latexmk -c widths.tex || true
	cd paper/note-dan && latexmk -c dan.tex || true
	cd paper/arxiv-en && latexmk -c bounds-en.tex || true
