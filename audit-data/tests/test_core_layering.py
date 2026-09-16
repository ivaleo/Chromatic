"""Модули core/ не зависят от кампаний: общие части лежат в core, кампании импортируют их оттуда."""

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "chromatic_research" / "core"


def test_core_does_not_import_campaigns():
    offenders = []
    for path in sorted(CORE.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            offenders += [f"{path.name}: {m}" for m in modules
                          if m.startswith("chromatic_research.campaigns")]
    assert not offenders, "core импортирует кампании:\n  " + "\n  ".join(offenders)
