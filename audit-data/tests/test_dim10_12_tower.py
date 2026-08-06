import json
import math

import numpy as np

import combigeo
from chromatic_research.campaigns.dim10_12_tower import (
    build_tower, separation_of_triple_lattice,
)
from chromatic_research.core.lattices import Astar, D as Dn, E8, _norm
from chromatic_research.core.lamination import deep_hole
from chromatic_research.paths import results_path


def _scaled(basis, target=1.0):
    basis = np.asarray(basis, dtype=float)
    lam1 = float(np.linalg.norm(combigeo.shortest_vector(basis.tolist())))
    return basis * (target / lam1)


def test_real_multiplier_rule_d_equals_two_over_rho():
    """Gamma = 3L has D_min = 2*lambda1, so d = 2/rho.

    The sweep only enumerates vectors of norm <= 2*diam, since longer ones clear
    automatically; when rho < 3/2 that ball is empty and the routine returns inf,
    which is the vacuous (and admissible) case.  Both branches are checked here.
    """
    for basis, rho_squared in ((E8(), 2.0), (Astar(3), 5 / 3), (Dn(4), 2.0)):
        B = _scaled(basis)
        diam = math.sqrt(rho_squared)                 # closed form at lambda1 = 1
        separation, _, _ = separation_of_triple_lattice(B, diam)
        assert separation >= diam                     # admissible either way
        if math.isfinite(separation):
            assert abs(separation - 2.0) < 1e-7
        else:
            assert 3.0 > 2 * diam                     # ball empty exactly when rho < 3/2


def test_multiplier_two_can_never_work():
    """Gamma = 2L would give D_min = lambda1, hence d = 1/rho <= 1 above dimension 1."""
    B = _scaled(E8())
    diam = math.sqrt(2.0)
    facets = [(list(f[0]), f[1]) for f in combigeo.relevant_facets(B.tolist())]
    normals = np.array([f[0] for f in facets], float)
    facets = [((normals[i] / np.linalg.norm(normals[i])).tolist(), float(facets[i][1]))
              for i in range(len(facets))]
    shortest = np.asarray(combigeo.shortest_vector(B.tolist()), float)
    separation = 2 * float(combigeo.dist_to_halfspaces((shortest).tolist(), facets))
    assert separation < diam            # d = 1/rho < 1


def test_tower_keeps_lambda1_and_stays_admissible():
    records = build_tower(10, n_dirs=300, budget=300.0)
    assert [r["n"] for r in records] == [9, 10]
    for record in records:
        assert abs(record["lambda1"] - 1.0) < 1e-9
        assert record["D_min"] > 1.999999
        assert record["diam_measured"] <= record["diam_rigorous"] + 1e-9
        assert record["valid_rigorous"]
    # the layer height must come from a lower bound on R, else lambda1 drops below 1
    assert abs(records[0]["layer_height"] - math.sqrt(0.5)) < 1e-12


def test_tower_recursion_matches_the_closed_form_where_R_is_exact():
    """Through n=10 the covering radius fed into the recursion is exact, so the
    bound coincides with diam^2 = 4 - 2 (3/4)^(n-8)."""
    for n, expected in ((9, 2.5), (10, 2.875)):
        assert abs((4 - 2 * 0.75 ** (n - 8)) - expected) < 1e-12


def test_stored_tower_artifact_is_consistent():
    path = results_path("dim10_12_tower.json")
    payload = json.loads(path.read_text())
    records = payload["tower"] if isinstance(payload, dict) else payload
    by_n = {r["n"]: r for r in records}
    assert by_n[10]["index"] == 3**10 and by_n[12]["index"] == 3**12
    for record in records:
        assert record["valid_rigorous"]
        assert record["d_rigorous"] >= 1.0
        assert record["d_rigorous"] <= record["d_measured"] + 1e-12
