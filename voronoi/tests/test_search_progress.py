"""Новая сигнатура find_optimal: без limits, с callback вместо print."""

import numpy as np

from voronoi4d import VoronoiPolyhedra, find_optimal


def test_find_optimal_accepts_keyword_only_options(tmp_path):
    grid = np.eye(4)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)

    det_dist, _, _ = find_optimal(
        range(2, 3), grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
    )

    assert 2 in det_dist


def test_progress_callback_receives_lines(tmp_path):
    grid = np.eye(4)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)

    lines = []
    find_optimal(
        range(2, 3), grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
        progress=lines.append,
    )

    assert lines, "progress должен получить хотя бы одну строку"
    assert any("det" in line for line in lines)


def test_no_progress_means_silence(tmp_path, capsys):
    grid = np.eye(4)
    vor = VoronoiPolyhedra(grid)
    vor.build(verbose=False)

    find_optimal(
        range(2, 3), grid, vor, vor.max_len,
        threshold=0.0, output_file=str(tmp_path / "r.txt"),
    )

    assert capsys.readouterr().out == ""
