"""Версии установленных пакетов не должны расходиться с pyproject.toml."""

from importlib.metadata import version
from pathlib import Path

import pytest

# tomllib появился в 3.11, а монорепо объявляет floor 3.10
tomllib = pytest.importorskip("tomllib", reason="tomllib доступен с Python 3.11")

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
