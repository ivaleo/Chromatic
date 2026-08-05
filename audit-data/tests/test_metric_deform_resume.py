from pathlib import Path

from chromatic_research.core.metric_deform import checkpoint_base_metric, resolve_saved_path


def test_checkpoint_base_metric_supports_old_and_new_layouts():
    assert checkpoint_base_metric({"base_metric": "new.json"}) == "new.json"
    assert (
        checkpoint_base_metric(
            {"optimizer": {"base_metric": "legacy.json"}}
        )
        == "legacy.json"
    )
    assert checkpoint_base_metric({"optimizer": {}}) is None


def test_resolve_saved_path_falls_back_to_checkpoint_sibling(tmp_path: Path):
    checkpoint = tmp_path / "resume.json"
    checkpoint.write_text("{}")
    sibling = tmp_path / "base.json"
    sibling.write_text("{}")

    resolved = resolve_saved_path(
        checkpoint, Path("old/location") / sibling.name
    )

    assert resolved == sibling
