"""Sanity-тесты движка кластерных раскрасок (fable_pseudolattice_20260807).

Запуск:  .venv/bin/python -m pytest audit-data/fable_pseudolattice_20260807/test_sanity.py -q
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import cluster_coloring as cc


def test_cosets_identity_scaled():
    H = 2 * np.eye(3, dtype=int)
    cs = cc.cosets_of_hnf(H)
    assert len(cs) == 8


def test_cosets_hnf_15():
    H = np.array([[1, 0, 12], [0, 1, 10], [0, 0, 15]])
    cs = cc.cosets_of_hnf(H)
    assert len(cs) == 15
    # все классы различны по модулю H
    Hinv = np.linalg.inv(H.astype(float))
    seen = set()
    for x in cs:
        c = x - np.round(x @ Hinv) @ H
        seen.add(tuple(np.rint(c).astype(int)))
    assert len(seen) == 15


def test_pair_interval_cube():
    geom = cc.CellGeom.build(np.eye(3))
    lo, hi = cc.pair_interval(geom, np.array([2.0, 0.0, 0.0]))
    assert lo == pytest.approx(1.0, abs=1e-9)
    assert hi == pytest.approx(math.sqrt(11.0), abs=1e-9)
    lo0, _ = cc.pair_interval(geom, np.array([0.4, 0.0, 0.0]))
    assert lo0 == pytest.approx(0.0, abs=1e-12)


def test_slsqp_agrees_with_gjk():
    geom = cc.CellGeom.build(np.eye(3))
    rng = np.random.default_rng(7)
    for _ in range(20):
        w = rng.normal(size=3) * 2.5
        lo_gjk, _ = cc.pair_interval(geom, w)
        lo_slsqp, lo_cert = cc.pair_lo_slsqp(geom, w)
        assert lo_slsqp == pytest.approx(lo_gjk, abs=1e-7)
        assert lo_cert <= lo_gjk + 1e-9


def test_realized_gap_synthetic():
    # вручную собранный ConflictData c одним классом и одним интервалом
    geom = cc.CellGeom.build(np.eye(3))
    conf = cc.ConflictData(
        geom=geom, H=np.eye(3, dtype=int), cosets=np.zeros((1, 3), dtype=int),
        cutoff=10.0, claim_cap=8.0,
        pairs={(0, 0): cc.PairData([(3.0, 5.0)])}, build_seconds=0.0)
    g = cc.realized_gap(conf, [0])
    # зазоры: (diam, 3) и (5, 8): отношения 3/1.732=1.732 и 8/5=1.6
    assert g["ratio"] == pytest.approx(3.0 / geom.diam, rel=1e-12)


def test_windowing_regression_bcc42():
    """Регрессия: дальние представители классов не должны терять интервалы.

    BCC, период = оптимальная подрешётка индекса 42: каждая пара классов
    обязана получить хотя бы один интервал, а раскраска всеми 42 цветами
    (косетная) должна дать ровно D_min(G)/diam."""
    import combigeo
    B = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]])
    r = combigeo.find_optimal(B.tolist(), index=42, threads=1)
    H = np.array(r.best.transition)
    conf = cc.build_conflicts(B, H)
    assert len(conf.pairs) == 42 * 43 // 2
    phi = list(range(42))                      # все классы разного цвета
    g = cc.realized_gap(conf, phi)
    expected = r.best.min_distance / conf.geom.diam
    assert g["ratio"] == pytest.approx(expected, rel=1e-9)


def test_a2_index7_control():
    """Полный конвейер на A2/7 обязан дать sqrt(7)/2 (классика)."""
    import combigeo
    B = np.array([[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
    r = combigeo.find_optimal(B.tolist(), index=7, threads=1)
    H = np.array(r.best.transition)
    conf = cc.build_conflicts(B, H)
    res = cc.search_best(conf, k=7, budget_s=60.0, verbose=False)
    assert res["best"]["ratio"] == pytest.approx(math.sqrt(7.0) / 2.0, abs=1e-6)
    ver = cc.verify_coloring(conf, res["best"]["phi"])
    assert ver["max_procedure_deviation"] < 1e-7
    assert ver["cert_gap"]["ratio"] == pytest.approx(math.sqrt(7.0) / 2.0,
                                                     abs=1e-5)
