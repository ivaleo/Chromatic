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
