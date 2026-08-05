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
