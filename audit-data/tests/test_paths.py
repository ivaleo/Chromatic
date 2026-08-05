"""Пути к артефактам не должны зависеть от каталога запуска и от машины."""

import gzip
import json

from chromatic_research import paths


def test_dirs_are_inside_the_checkout():
    assert paths.RESULTS_DIR.parent == paths.AUDIT_DIR
    assert paths.RUNS_DIR.parent == paths.AUDIT_DIR
    assert paths.AUDIT_DIR.name == "audit-data"


def test_load_json_reads_plain_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    (tmp_path / "sample.json").write_text(json.dumps({"d": 1.5}))
    assert paths.load_json("sample.json") == {"d": 1.5}


def test_load_json_reads_gzipped_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    paths.RUNS_DIR.mkdir(parents=True)
    with gzip.open(paths.RUNS_DIR / "raw.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"d": 0.99}, handle)
    assert paths.load_json("raw.json") == {"d": 0.99}


def test_missing_artifact_names_both_directories():
    try:
        paths.load_json("нет-такого.json")
    except FileNotFoundError as error:
        assert "results" in str(error) and "runs" in str(error)
    else:
        raise AssertionError("ожидался FileNotFoundError")
