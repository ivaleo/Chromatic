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
