"""Абстракция бэкенда и явный выбор реализации.

Фасад chromatic не делает автоопределения: бэкенд задаётся ЯВНО по имени
(``"voronoi4d"`` или ``"combigeo"``). Это убирает скрытую магию и делает
поведение воспроизводимым.

- ``voronoi4d`` — эталонная python-реализация, только размерность 4;
- ``combigeo`` — быстрое C++ ядро (pybind11), размерности 2…6.
"""

from __future__ import annotations

import abc
from typing import Iterable, List, Sequence

from .model import Cell, OptimalResult


class Backend(abc.ABC):
    """Единый интерфейс над конкретной реализацией алгоритма."""

    name: str = ""
    supported_dims: Sequence[int] = ()

    @staticmethod
    @abc.abstractmethod
    def available() -> bool:
        """Доступен ли бэкенд (установлен ли соответствующий пакет)."""

    def check_dim(self, dim: int) -> None:
        """Проверяет, что размерность поддерживается бэкендом."""
        if dim not in self.supported_dims:
            raise ValueError(
                f"бэкенд {self.name!r} не поддерживает размерность {dim} "
                f"(поддерживаются: {list(self.supported_dims)})"
            )

    @abc.abstractmethod
    def build_cell(self, basis: Sequence[Sequence[float]]) -> Cell:
        """Строит ячейку Вороного центральной точки решётки."""

    @abc.abstractmethod
    def cell_distance(self, point: Sequence[float], cell: Cell) -> float:
        """Сырое геометрическое расстояние от точки до ячейки."""

    @abc.abstractmethod
    def min_color_distance(self, basis: Sequence[Sequence[float]],
                           sub_basis: Sequence[Sequence[float]]) -> float:
        """Сырое расстояние D между одноцветными ячейками для подрешётки."""

    @abc.abstractmethod
    def find_optimal(self, basis: Sequence[Sequence[float]], index: int) -> OptimalResult:
        """Оптимальная подрешётка заданного индекса (числа цветов)."""

    def find_optimal_range(self, basis: Sequence[Sequence[float]],
                           indices: Iterable[int]) -> "dict[int, OptimalResult]":
        """Поиск по набору индексов; по умолчанию — цикл find_optimal."""
        return {int(k): self.find_optimal(basis, int(k)) for k in indices}

    @abc.abstractmethod
    def lll_reduce(self, basis: Sequence[Sequence[float]],
                   delta: float = 0.75) -> List[List[float]]:
        """LLL-приведение базиса."""

    def shortest_vector(self, basis: Sequence[Sequence[float]]) -> List[float]:
        """Кратчайший ненулевой вектор решётки (если бэкенд поддерживает)."""
        raise NotImplementedError(
            f"бэкенд {self.name!r} не предоставляет shortest_vector"
        )


# --- реестр и явный выбор -------------------------------------------------------

_REGISTRY: "dict[str, type[Backend]]" = {}


def register_backend(cls: "type[Backend]") -> "type[Backend]":
    """Регистрирует класс бэкенда по его имени (декоратор)."""
    _REGISTRY[cls.name] = cls
    return cls


def available_backends() -> List[str]:
    """Имена бэкендов, которые реально доступны (пакеты установлены)."""
    _ensure_loaded()
    return [name for name, cls in sorted(_REGISTRY.items()) if cls.available()]


def get_backend(name: str) -> Backend:
    """Возвращает экземпляр бэкенда по ИМЕНИ (явный выбор, без автоопределения).

    :param name: ``"voronoi4d"`` или ``"combigeo"``.
    :raises ValueError: имя неизвестно.
    :raises ImportError: бэкенд известен, но его пакет не установлен.
    """
    _ensure_loaded()
    if name not in _REGISTRY:
        raise ValueError(
            f"неизвестный бэкенд {name!r}; доступные имена: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    if not cls.available():
        raise ImportError(
            f"бэкенд {name!r} не установлен. См. инструкцию по установке в README "
            f"(voronoi4d: pip install -e ../voronoi; combigeo: pip install ../combigeo)."
        )
    return cls()


_loaded = False


def _ensure_loaded() -> None:
    """Лениво импортирует модули бэкендов (регистрация при импорте)."""
    global _loaded
    if _loaded:
        return
    from . import _voronoi4d, _combigeo  # noqa: F401  (регистрация через декоратор)
    _loaded = True
