"""Power (Laguerre) tilings as colourings, and the floor of the paradigm."""

import math

import numpy as np
import pytest

from chromatic_research.core import lattices as lat
from chromatic_research.core import paradigm_floor as pf
from chromatic_research.core import power_coloring as pc
from chromatic_research.campaigns import power_tilings as pt


# ------------------------------------------------- the framework contains ours

@pytest.mark.parametrize("name,build,hermite,expected", [
    ("A2/7", lambda: lat.A(2), [[1, 5], [0, 7]], math.sqrt(7) / 2),
    ("A2/9", lambda: lat.A(2), [[3, 0], [0, 3]], math.sqrt(3)),
    ("Z3/27", lambda: lat.Z(3), np.diag([3, 3, 3]), 2 / math.sqrt(3)),
    ("A3*/27", lambda: lat.Astar(3), np.diag([3, 3, 3]), 2 / math.sqrt(5 / 3)),
])
def test_voronoi_colourings_are_reproduced(name, build, hermite, expected):
    L = build()
    H = np.asarray(hermite, float)
    k = int(round(abs(np.linalg.det(H))))
    rep = pc.evaluate(H @ L, pc.transversal(L, H), np.zeros(k))
    assert rep.sound and rep.colours == k
    assert rep.width == pytest.approx(expected, abs=1e-7)


def test_transversal_is_a_full_set_of_distinct_cosets():
    L = lat.Astar(3)
    H = np.array([[1, 0, 4], [0, 1, 12], [0, 0, 15]], float)
    T = pc.transversal(L, H)
    assert len(T) == 15
    Ginv = np.linalg.inv(H @ L)
    keys = {tuple(np.round(np.mod(t @ Ginv, 1.0), 6)) for t in T}
    assert len(keys) == 15


# ------------------------------------------------------- certified, never rosy

def test_gap_is_a_lower_bound_and_tight_on_a_cube():
    # P = [-1/2, 1/2]^3, so P - P = [-1,1]^3 and dist(g, P-P) = |g|_inf-clipped
    V = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)],
                 float)
    for g, want in [(np.array([3., 0., 0.]), 2.0),
                    (np.array([0., 2.5, 0.]), 1.5),
                    (np.array([2., 2., 0.]), math.sqrt(2)),
                    (np.array([0.5, 0., 0.]), 0.0)]:
        assert pc.certified_gap(V, g) == pytest.approx(want, abs=1e-9)


def test_gap_never_exceeds_the_true_distance():
    rng = np.random.default_rng(7)
    V = rng.normal(size=(12, 4))
    for _ in range(20):
        g = rng.normal(size=4) * 3.0
        lower = pc.certified_gap(V, g)
        # brute force upper bound from random convex combinations
        best = math.inf
        for _ in range(400):
            a = rng.random(12); a /= a.sum()
            b = rng.random(12); b /= b.sum()
            best = min(best, float(np.linalg.norm(g - a @ V + b @ V)))
        assert lower <= best + 1e-9


def test_lll_keeps_the_lattice():
    rng = np.random.default_rng(3)
    B = rng.integers(-9, 9, size=(4, 4)).astype(float) + np.eye(4) * 11
    R = pc.lll_reduce(B)
    U = R @ np.linalg.inv(B)
    assert np.allclose(U, np.rint(U), atol=1e-7)
    assert abs(round(np.linalg.det(U))) == 1
    assert np.linalg.norm(R[0]) <= np.linalg.norm(B[0]) + 1e-9


def test_buried_site_does_not_consume_a_colour():
    L = lat.A(2)
    H = np.array([[3, 0], [0, 3]], float)
    T = pc.transversal(L, H)
    w = np.zeros(9)
    w[3] = -10.0                      # push one site far below its neighbours
    rep = pc.evaluate(H @ L, T, w)
    assert rep.colours < 9


# ------------------------------------------------------------------- the floor

def test_tiling_floor_matches_the_brunn_minkowski_derivation():
    for n in range(1, 13):
        assert pf.tiling_floor(n) == 2 ** n


def test_theta_equals_delta_times_rho_to_the_n():
    for row in pf.landscape():
        if "identity_residual" in row:
            assert row["identity_residual"] < 1e-6 * row["rho"] ** row["n"]


def test_covering_densities_match_independent_closed_forms():
    # three parents whose Theta can be written down exactly and checked
    exact = {"A2": 2 * math.pi / (3 * math.sqrt(3)),          # w2 (1/3) / (sqrt3/2)
             "E8": math.pi ** 4 / 24,                          # w8 * 1 / 1
             "Leech": math.pi ** 12 / math.factorial(12) * 2 ** 12}
    seen = 0
    for row in pf.landscape():
        if row["parent"] in exact:
            assert row["Theta"] == pytest.approx(exact[row["parent"]], rel=1e-12)
            seen += 1
    assert seen == 3


def test_every_record_clears_the_universal_floor():
    for row in pf.landscape():
        assert row["colours"] >= pf.tiling_floor(row["n"])
        if "floor_parent" in row:
            assert row["colours"] >= row["floor_parent"]


def test_eisenstein_window_is_narrow():
    w = pf.eisenstein_window()
    assert w["rho_needed_eisenstein"] == pytest.approx(math.sqrt(7 / 3))
    assert 0.0 < w["window_percent"] < 1.0


def test_index_window_floors_are_below_the_records():
    for row in pf.open_windows():
        assert row["floor_packing"] > 0 and row["floor_volume"] > 0
        assert row["floor"] <= row["record"], row["parent"]
        assert row["room_factor"] >= 1.0


def test_index_window_reproduces_the_inradius_threshold_for_k12():
    row = next(r for r in pf.open_windows() if r["parent"] == "K12")
    # lambda_1(G) > 2R + lambda_1 = 2 sqrt(8/3) + 2, so lambda_1(G)^2 > 27.7
    assert row["lambda1_sub_min"] == pytest.approx(2 * math.sqrt(8 / 3) + 2)
    assert row["lambda1_sub_min"] ** 2 == pytest.approx(27.7307, abs=1e-3)
    # the Eisenstein sublattice (3+omega)K12 has index 7^6, just above the floor
    assert row["floor_packing"] < 7 ** 6 < row["record"]
    assert row["delta_max_proven"] is False       # K12 optimality is open


def test_shape_bound_is_below_the_colour_count():
    # the maximum-volume-tile bound must not exceed a colouring that exists
    L = lat.Astar(3)
    H = np.array([[1, 0, 4], [0, 1, 12], [0, 0, 15]], float)
    G = H @ L
    rep = pc.evaluate(G, pc.transversal(L, H), np.zeros(15))
    scale = rep.diameter                      # normalise the forbidden distance
    bound = pt.shape_bound(G / scale, 3)
    assert bound["k_lower_bound"] <= 15 + 1e-9


def test_campaign_report_is_clean():
    rep = pt.report()
    assert rep["closed_forms_all_ok"]
    assert rep["probes_all_flat"]
    assert all(c["ok"] for c in rep["champions_verified"])


def test_probes_are_seeded_at_the_recorded_champions():
    # the two cheap seeds must be re-derivable, and no probe may claim a gain
    live = {c["case"]: c["width"] for c in pt.verify_champions()}
    for probe in pt.PROBES:
        assert probe["reached"] <= probe["champion"] + 1e-9
        if probe["case"] in live:
            assert live[probe["case"]] == pytest.approx(probe["champion"], abs=1e-7)
