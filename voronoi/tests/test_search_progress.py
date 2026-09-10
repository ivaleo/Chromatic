"""Новая сигнатура find_optimal: без limits, с callback вместо print."""

from voronoi4d import find_optimal

# решётка Z^4 берётся из общей фикстуры vor (conftest.py)


def test_find_optimal_accepts_keyword_only_options(vor, tmp_path):
    det_dist, _, _ = find_optimal(
        range(2, 3), vor.grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
    )

    assert 2 in det_dist


def test_progress_callback_receives_lines(vor, tmp_path):
    lines = []
    find_optimal(
        range(2, 3), vor.grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
        progress=lines.append,
    )

    assert lines, "progress должен получить хотя бы одну строку"
    assert any("det" in line for line in lines)


def test_no_progress_means_silence(vor, tmp_path, capsys):
    find_optimal(
        range(2, 3), vor.grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
    )

    assert capsys.readouterr().out == ""
