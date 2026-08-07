"""The product calculus, its alpha-ladder and the dimension-10 record."""

import json
import math

import pytest

from chromatic_research.core.product_calculus import (
    Entry, best_partition, budget, eisenstein_distance, gaussian_distance,
    hurwitz_distance, pareto, product_width, real_distance,
)
from chromatic_research.paths import results_path


# --------------------------------------------------------- the four unit cells

def test_real_distance_reproduces_the_multiplier_rule():
    # D = (m-1) lambda1, i.e. dist(m/2, [-1/2,1/2]) = (m-1)/2
    for m in range(2, 8):
        assert real_distance(m) == pytest.approx((m - 1) / 2)


def test_eisenstein_distance_matches_the_planar_theorem():
    omega = complex(-0.5, math.sqrt(3) / 2)
    # 3+omega is the constant behind D = sqrt(7/3) lambda1
    assert eisenstein_distance(3 + omega) == pytest.approx(math.sqrt(7 / 12))
    # 5+2omega is the plane spacer of index 19 used in dimension 10
    assert eisenstein_distance(5 + 2 * omega) == pytest.approx(math.sqrt(31 / 12))
    # a real multiplier must agree with the segment formula
    assert eisenstein_distance(3 + 0j) == pytest.approx(1.0)


def test_gaussian_and_hurwitz_cells():
    assert gaussian_distance(2 + 1j) == pytest.approx(0.5)
    assert gaussian_distance(3 + 0j) == pytest.approx(1.0)
    # the quaternion of norm 7 reproduces the Eisenstein constant
    assert hurwitz_distance([2, 1, 1, 1]) == pytest.approx(math.sqrt(7 / 12))
    assert hurwitz_distance([2, 2, 0, 0]) == pytest.approx(math.sqrt(0.5))


# ------------------------------------------------------------------- the rule

def test_budget_and_width_of_the_dimension_10_product():
    widths = [math.sqrt(7 / 6), math.sqrt(31 / 4)]          # E8/2401 and A2/19
    assert budget(widths) == pytest.approx(214 / 217)
    assert product_width(widths) == pytest.approx(math.sqrt(217 / 214))
    assert budget(widths) < 1.0


def test_three_lambda_case_is_the_old_sum_of_squares_rule():
    # Gamma = 3 Lambda has d = 2/rho, so sum 1/d^2 <= 1 is sum rho^2 <= 4
    rhos = [math.sqrt(2), math.sqrt(2)]                     # E8 (+) E8
    widths = [2 / r for r in rhos]
    assert budget(widths) == pytest.approx(sum(r * r for r in rhos) / 4)


def test_pareto_drops_dominated_rungs():
    rungs = [Entry(2, 9, 1.7, "a"), Entry(2, 12, 1.5, "b"), Entry(2, 16, 2.6, "c")]
    assert [e.index for e in pareto(rungs)] == [9, 16]


# ---------------------------------------------------------- the optimisation

def test_partition_finds_the_45619_split():
    ladders = {
        2: [Entry(2, 19, math.sqrt(31 / 4), "A2 x (5+2w)")],
        8: [Entry(8, 2401, math.sqrt(7 / 6), "E8 x (3+w)")],
    }
    colours, chain, width = best_partition(ladders, 10)
    assert colours == 45619
    assert sorted(e.index for e in chain) == [19, 2401]
    assert width == pytest.approx(math.sqrt(217 / 214))


def test_partition_refuses_an_over_budget_split():
    # two E8/2401 blocks would need 12/7 of the budget
    ladders = {8: [Entry(8, 2401, math.sqrt(7 / 6), "E8 x (3+w)")]}
    assert best_partition(ladders, 16) is None


# ------------------------------------------------------- the stored artefacts

def test_dim10_product_record_is_consistent():
    path = results_path("dim10_product.json")
    if not path.exists():
        pytest.skip("run chromatic_research.campaigns.dim10_product first")
    record = json.loads(path.read_text())
    assert record["index"] == 45619 == 2401 * 19
    assert record["d"] > 1.0
    assert record["d"] == pytest.approx(math.sqrt(217 / 214), abs=1e-9)
    assert record["D_min"] == pytest.approx(math.sqrt(7.0), abs=1e-9)
    assert record["index"] < 3 ** 10


def test_inradius_floor_rules_out_a_plane_block_of_index_16():
    from chromatic_research.core.minkowski import inradius_floor
    # hexagonal lattice, lambda1 = 1: diam = 2/sqrt3, det = sqrt3/2
    floor = inradius_floor(2, 1.0, 2 / math.sqrt(3), math.sqrt(3) / 2, math.sqrt(7))
    assert floor == pytest.approx(16.44343, abs=1e-4)
    # the minimum over all plane shapes is 16.021, so 17 is the rigorous floor
    assert floor > 16.0


def test_inradius_floor_agrees_with_the_multiplier_rule():
    from chromatic_research.core.minkowski import inradius_floor
    # Gamma = m*Z in dimension 1: lambda1 = diam = det = 1, width d = m-1,
    # and the lemma is tight there, so the floor must return exactly m
    for m in (3, 4, 5):
        assert inradius_floor(1, 1.0, 1.0, 1.0, m - 1) == pytest.approx(m)


def test_ladder2d_puts_the_sqrt7_rung_at_19():
    path = results_path("ladder2d.json")
    if not path.exists():
        pytest.skip("run chromatic_research.campaigns.ladder2d first")
    data = json.loads(path.read_text())
    assert data["first_sqrt7"] == 19
    below = [r for r in data["records"] if r["index"] < 19]
    assert below and all(not r["reaches_sqrt7"] for r in below)
