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
