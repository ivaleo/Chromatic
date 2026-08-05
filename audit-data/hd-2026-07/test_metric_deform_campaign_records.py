from metric_deform import campaign_records, select_record


def test_cpsat_candidates_are_metric_campaign_records() -> None:
    payload = {
        "candidates": [
            {
                "moduli": [7, 4, 4, 3],
                "rows": [[1, 0], [0, 1], [1, 1], [1, 2]],
                "image_index": 336,
                "beta": 1.0,
                "killed": 3,
                "minimum_conflict_ratio": 0.91,
            },
            {
                "moduli": [7, 4, 4, 3],
                "rows": [[1, 1], [0, 1], [1, 0], [2, 1]],
                "image_index": 336,
                "beta": 1.0,
                "killed": 2,
                "minimum_conflict_ratio": 0.94,
            },
        ]
    }
    records = campaign_records(payload)
    assert len(records) == 2
    selected = select_record(
        payload, [7, 4, 4, 3], 1.0, rank=0
    )
    assert selected["minimum_conflict_ratio"] == 0.94
