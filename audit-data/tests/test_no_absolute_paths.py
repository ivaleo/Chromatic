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
