import math

from chromatic_research.paths import load_json


def _load(name: str) -> dict:
    return load_json(name)


def test_prime_131_exact_replays_agree_on_complete_frontier():
    first = _load("exact_prime_threshold_a5_131_escape_refine_round3.json")
    replay = _load(
        "exact_prime_threshold_a5_131_escape_refine_round3_replay.json"
    )

    expected_row = [1, 52, 59, 40, 56]
    for payload in (first, replay):
        assert payload["target_proved_unsat"] is True
        assert payload["candidate"]["rows"] == [expected_row]
        assert math.isclose(
            payload["frontier_threshold"],
            0.9902161610015159,
            rel_tol=0.0,
            abs_tol=2e-15,
        )
        target = payload["decisions"][0]
        assert target["threshold"] == 1.0
        assert target["proved_unsat"] is True
        assert target["enumeration"][0]["tested"] == 296_765_305

    assert (
        first["enumerator"]["sha256"]
        != replay["enumerator"]["sha256"]
    )


def test_prime_131_rational_and_independent_audits_agree():
    expected = {
        10_000: 0.9901626516977177,
        100_000: 0.9902110602582538,
        1_000_000: 0.9902159621226475,
    }
    for denominator, ratio in expected.items():
        direct = _load(
            f"active_metric_a5_131_escape_frontier_exact_d{denominator}.json"
        )
        independent = _load(
            "active_metric_a5_131_escape_frontier_"
            f"independent_exact_d{denominator}.json"
        )

        assert direct["diagnostic_status"] == "invalid-candidate"
        assert independent["diagnostic_status"] == "invalid-candidate"
        assert direct["certified_upper_bound"] is None
        assert independent["certified_upper_bound"] is None
        assert direct["voronoi"]["facets"] == 62
        assert direct["voronoi"]["vertices"] == 720
        assert direct["short_vector_certificate"]["exact_vector_count"] == 30
        assert direct["independent_forbidden_check"]["kernel_conflicts"] == 13
        assert independent["voronoi_relevant_audit"]["parity_classes"] == 31
        assert independent["voronoi_relevant_audit"]["facet_count"] == 62
        assert independent["exact_polytope_graph"]["vertices"] == 720
        assert independent["exact_polytope_graph"]["edges"] == 1800
        assert independent["kernel"]["short_vector_count"] == 30
        assert math.isclose(
            direct["separation"]["distance_ratio"],
            ratio,
            rel_tol=0.0,
            abs_tol=2e-15,
        )
        assert math.isclose(
            independent["projection_audit"]["distance_ratio"],
            ratio,
            rel_tol=0.0,
            abs_tol=2e-15,
        )


def test_index_130_exact_fixed_metric_bracket():
    infeasible = _load("exact_threshold_a5_130_on132_t09766882.json")
    feasible = _load("exact_threshold_a5_130_on132_t09766881.json")

    assert infeasible["threshold"] == 0.9766882
    assert infeasible["enumeration"]["status"] == "INFEASIBLE"
    assert infeasible["enumeration"]["tested"] == 749_112_551
    assert infeasible["enumeration"]["total_tuples"] == 749_112_551
    assert feasible["threshold"] == 0.9766881
    assert feasible["enumeration"]["status"] == "FEASIBLE"
    assert math.isclose(
        feasible["candidate"]["minimum_conflict_ratio"],
        0.9766881156247794,
        rel_tol=0.0,
        abs_tol=2e-15,
    )


def test_index_129_exact_fixed_metric_bracket():
    infeasible = _load("exact_threshold_a5_129_on132_t09570142.json")
    feasible = _load("exact_threshold_a5_129_on132_t09570141.json")

    assert infeasible["threshold"] == 0.9570142
    assert infeasible["small_rows_total"] == 121
    assert infeasible["small_rows_completed"] == 121
    assert infeasible["all_small_rows_proved_unsat"] is True
    assert feasible["threshold"] == 0.9570141
    assert feasible["small_rows_completed"] == 121
    assert feasible["all_small_rows_proved_unsat"] is False
    assert math.isclose(
        feasible["best_candidate"]["minimum_conflict_ratio"],
        0.9570141199349721,
        rel_tol=0.0,
        abs_tol=2e-15,
    )
