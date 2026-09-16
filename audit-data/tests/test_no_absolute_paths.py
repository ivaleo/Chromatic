"""В репозитории не должно быть путей, привязанных к машине автора.

Проверяются все отслеживаемые текстовые файлы — код, документация, данные
(в том числе сжатые `runs/*.json.gz`); журнал `journal/` — исторический и не
проверяется.
"""

import gzip
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "journal"}
FORBIDDEN = ("/Users/", "/home/", "C:\\", "/private/tmp/")
SUFFIXES = {".py", ".json", ".gz", ".md", ".txt", ".tex", ".sh", ".toml", ".yml", ".cfg", ".ipynb"}


def _tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        return [ROOT / name for name in out]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in ROOT.rglob("*") if p.is_file()]


def test_no_machine_specific_paths():
    offenders = []
    for path in sorted(_tracked_files()):
        rel = path.relative_to(ROOT)
        if (path == SELF or path.suffix not in SUFFIXES or not path.is_file()
                or any(part in SKIP_DIRS for part in rel.parts)):
            continue
        data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        offenders += [f"{rel}: {marker}" for marker in FORBIDDEN if marker in text]

    assert not offenders, (
        "машинно-зависимые пути (в коде — chromatic_research.paths.portable/results_path):\n  "
        + "\n  ".join(offenders)
    )
