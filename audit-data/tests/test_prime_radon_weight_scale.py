import numpy as np

from chromatic_research.core.prime_radon import PrimarySearch, weighted_close, weighted_improves


class _DeterministicSecondStart(PrimarySearch):
    def random_rows(self) -> list[np.ndarray]:
        return [np.asarray([0, 1], dtype=np.int64)]


def test_weighted_comparison_resolves_subnormal_scale_improvement() -> None:
    assert weighted_improves(1e-30, 2e-30)
    assert not weighted_improves(2e-30, 1e-30)
    assert weighted_close(1e-30, 1e-30 * (1.0 + 1e-12))


def test_weighted_restart_is_not_frozen_by_absolute_tolerance() -> None:
    forbidden = np.asarray(
        [[0, 1], [1, 0], [1, 1]], dtype=np.int64
    )
    weights = np.asarray([2e-30, 1e-30, 3e-30])
    search = _DeterministicSecondStart(forbidden, [2], seed=1)
    result = search.run_weighted(
        weights,
        restarts=1,
        max_sweeps=0,
        top=1,
        progress_every=0,
        initial_rows=[[1, 0]],
    )
    assert result.weighted_loss == 1e-30
    assert result.rows[0].tolist() == [0, 1]


def test_weighted_archive_retains_distinct_exact_kernels() -> None:
    forbidden = np.asarray(
        [[0, 1], [1, 0], [1, 1]], dtype=np.int64
    )
    weights = np.asarray([2e-30, 1e-30, 3e-30])
    search = _DeterministicSecondStart(forbidden, [2], seed=1)
    archive = search.run_weighted_archive(
        weights,
        archive_size=2,
        restarts=1,
        max_sweeps=0,
        top=1,
        progress_every=0,
        initial_rows=[[1, 0]],
    )
    assert len(archive) == 2
    assert {tuple(result.rows[0]) for result in archive} == {
        (1, 0),
        (0, 1),
    }
