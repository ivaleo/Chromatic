import numpy as np

from block_row_metric_opt import metric_source_rows


def test_metric_source_rows_accepts_exact_matching_image() -> None:
    metric = {
        "source_record": {
            "moduli": [2, 3],
            "rows": [[1, 0], [0, 1]],
        }
    }
    rows = metric_source_rows(metric, [2, 3], 2)
    assert rows is not None
    assert [row.tolist() for row in rows] == [[1, 0], [0, 1]]


def test_metric_source_rows_rejects_mismatch_and_non_surjection() -> None:
    metric = {
        "source_record": {
            "moduli": [2, 3],
            "rows": [[0, 0], [0, 1]],
        }
    }
    assert metric_source_rows(metric, [2, 3], 2) is None
    assert metric_source_rows(metric, [2, 2], 2) is None


def test_metric_source_rows_returns_independent_arrays() -> None:
    raw = [[1, 0], [0, 1]]
    metric = {"source_record": {"moduli": [2, 3], "rows": raw}}
    rows = metric_source_rows(metric, [2, 3], 2)
    assert rows is not None
    rows[0][0] = 0
    assert np.asarray(raw)[0, 0] == 1
