import json

import numpy as np
import pytest

from chromatic_research.campaigns.portfolio_metric_opt import _optimization_frame, _ranked


def _metric_payload(n, parameters=None, basis=None, base_metric=None):
    expected = n * (n + 1) // 2 - 1
    payload = {
        "best": {
            "parameters": (
                list(parameters)
                if parameters is not None
                else [0.0] * expected
            ),
            "basis": (
                np.asarray(basis, dtype=float).tolist()
                if basis is not None
                else np.eye(n).tolist()
            ),
        }
    }
    if base_metric is not None:
        payload["base_metric"] = str(base_metric)
    return payload


def test_ranked_descending_order_statistic():
    assert _ranked([0.2, 0.9, 0.5], 1) == 0.9
    assert _ranked([0.2, 0.9, 0.5], 2) == 0.5
    assert _ranked([0.2, 0.9, 0.5], 3) == 0.2


def test_local_base_starts_zero_centered(tmp_path):
    n = 3
    basis = np.diag([1.2, 0.9, 1.0 / 1.08])
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_metric_payload(n, basis=basis)))
    source = _metric_payload(n, parameters=np.arange(5))

    basis0, initial = _optimization_frame(
        lattice="unused",
        n=n,
        source_payload=source,
        base_metric=base,
        resume=None,
    )

    assert np.allclose(basis0, basis)
    assert np.array_equal(initial, np.zeros(5))


def test_local_resume_must_use_same_base(tmp_path):
    n = 3
    base = tmp_path / "base.json"
    other = tmp_path / "other.json"
    base.write_text(json.dumps(_metric_payload(n)))
    other.write_text(json.dumps(_metric_payload(n)))
    resume = tmp_path / "resume.json"
    resume.write_text(
        json.dumps(
            _metric_payload(
                n,
                parameters=[0.1, 0.2, 0.3, 0.4, 0.5],
                base_metric=other,
            )
        )
    )

    with pytest.raises(ValueError, match="different local base metric"):
        _optimization_frame(
            lattice="unused",
            n=n,
            source_payload=_metric_payload(n),
            base_metric=base,
            resume=resume,
        )


def test_local_resume_reuses_compatible_parameters(tmp_path):
    n = 3
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_metric_payload(n)))
    parameters = [0.1, 0.2, 0.3, 0.4, 0.5]
    resume = tmp_path / "resume.json"
    resume.write_text(
        json.dumps(
            _metric_payload(
                n,
                parameters=parameters,
                base_metric=base,
            )
        )
    )

    _, initial = _optimization_frame(
        lattice="unused",
        n=n,
        source_payload=_metric_payload(n),
        base_metric=base,
        resume=resume,
    )

    assert np.allclose(initial, parameters)
