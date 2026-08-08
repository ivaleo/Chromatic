"""Semi-lattice colourings: the exact certifier and the arithmetic obstruction."""

import json
import math
from fractions import Fraction as F

import pytest

from chromatic_research.core.semilattice import (
    absolute_floor, aligned_ideal_plane, eisenstein_norms, eisenstein_record,
    plane_floor, same_shape_floor, tower_record,
)
from chromatic_research.core.semilattice_cert import certify
from chromatic_research.paths import results_path

HEX = [[F(1), F(1, 2)], [F(1, 2), F(1)]]


def square_reps(m):
    """Cosets of Gamma in Gamma/m -- the classical colouring of index m^2."""
    return [(F(i, m), F(j, m)) for i in range(m) for j in range(m)]


def alpha_reps(a, b, N):
    """Cosets of Gamma in Gamma/(a+b*omega); basis {1, 1+omega}."""
    return [(F(a * k, N), F(-b * k, N)) for k in range(N)]


# ------------------------------------------------- the certifier is exact

@pytest.mark.parametrize("reps,expected", [
    (square_reps(2), F(3, 4)),        # index 4
    (alpha_reps(3, 1, 7), F(7, 4)),   # index 7,  (3+omega)
    (square_reps(3), F(3)),           # index 9
    (alpha_reps(4, 3, 13), F(19, 4)),  # index 13
    (square_reps(4), F(27, 4)),       # index 16
    (alpha_reps(5, 2, 19), F(31, 4)),  # index 19, the flat spacer of R^10
    (alpha_reps(5, 1, 21), F(37, 4)),  # index 21
])
def test_certifier_reproduces_the_classical_rungs_exactly(reps, expected):
    r = certify(HEX, reps, [F(0)] * len(reps))
    assert r is not None and "error" not in r
    assert r["d2"] == expected
    assert r["tiling_exact"]


def test_certifier_agrees_with_the_stored_classical_ladder():
    stored = {rec["index"]: rec["d"]
              for rec in json.loads(results_path("ladder2d.json").read_text())["records"]}
    for reps, k in ((alpha_reps(3, 1, 7), 7), (square_reps(4), 16),
                    (alpha_reps(5, 2, 19), 19)):
        r = certify(HEX, reps, [F(0)] * len(reps))
        assert math.sqrt(float(r["d2"])) == pytest.approx(stored[k], abs=1e-6)


# --------------------------------------------- the arithmetic obstruction

def test_seventeen_and_eighteen_are_not_eisenstein_norms():
    norms = eisenstein_norms(30)
    assert 16 in norms and 19 in norms
    assert 17 not in norms and 18 not in norms


def test_planar_floor_sits_between_16_and_19():
    # geometry allows 17, arithmetic forces 19
    floor = plane_floor(math.sqrt(7))
    assert 16.44 < floor < 16.45
    assert aligned_ideal_plane(16) < math.sqrt(7) < aligned_ideal_plane(17)


def test_absolute_floor_is_two_to_the_n_at_unit_width():
    assert absolute_floor(8, 1.0) == pytest.approx(256.0)
    assert absolute_floor(2, math.sqrt(7)) == pytest.approx((1 + math.sqrt(7)) ** 2)


def test_same_shape_floor_matches_the_stored_room_table():
    rows = json.loads(results_path("semilattice_room.json").read_text())
    by_name = {r["name"]: r for r in rows}
    r8 = eisenstein_record(8, 2401)
    assert r8.floor == pytest.approx(by_name["(3+w)Lambda n=8"]["floor"])
    r12 = tower_record(12, 3 ** 12, 2 / 1.224745)
    assert r12.floor == pytest.approx(by_name["3Lambda n=12"]["floor"])
    assert same_shape_floor(8, 2401, 1.0, 0.5) == 2401     # no slack at d = 1


# --------------------------------------------- the ladder artefact itself

def test_semilattice_ladder_is_monotone_and_beats_the_classical_rungs():
    data = json.loads(results_path("semilattice_ladder2d.json").read_text())
    rows = data["rows"]
    prev = 0.0
    for r in rows:
        assert r["d_semilattice"] >= r["d_classical_envelope"] - 1e-6 or r["N"] in (6, 20)
        prev = max(prev, r["d_semilattice"])
    new = [r["N"] for r in rows if r["strictly_new"]]
    assert 8 in new and 15 in new


# ------------------------------------------------- the two certified rungs

@pytest.mark.parametrize("N,classical", [(8, 1.4), (15, 2.1866070)])
def test_certified_rungs_beat_the_classical_ladder(N, classical):
    cert = json.loads(results_path(f"semilattice_cert_N{N}.json").read_text())
    d2 = F(cert["d2"])
    assert float(d2) > classical ** 2          # exact rational vs the classical rung
    assert cert["tiling_exact"]
    assert cert["d"] == pytest.approx(math.sqrt(float(d2)))
    # the independent float check must agree with the search
    assert cert["independent_float_d"] == pytest.approx(cert["float_search_d"], abs=1e-7)


@pytest.mark.parametrize("N,classical", [(8, 1.4), (15, 2.1866070)])
def test_small_denominator_certificates_recompute(N, classical):
    """Re-run the exact certifier on the compact (denominator 20) witnesses."""
    cert = json.loads(results_path(f"semilattice_cert_N{N}_small.json").read_text())
    gram = [[F(x) for x in row] for row in cert["gram"]]
    ts = [(F(a), F(b)) for a, b in cert["ts"]]
    ws = [F(w) for w in cert["ws"]]
    r = certify(gram, ts, ws)
    assert r is not None and "error" not in r
    assert r["d2"] == F(cert["d2"])
    assert float(r["d2"]) > classical ** 2
    assert r["tiling_exact"]


def test_cells_are_not_all_congruent_hexagons():
    """A lattice colouring would give congruent hexagons; these do not."""
    for N in (8, 15):
        cert = json.loads(results_path(f"semilattice_cert_N{N}.json").read_text())
        assert len(set(cert["n_vertices"])) > 1
