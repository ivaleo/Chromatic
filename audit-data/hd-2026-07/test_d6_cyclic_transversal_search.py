import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name(
    "d6_cyclic_transversal_search.py"
)
SPEC = importlib.util.spec_from_file_location(
    "d6_cyclic_transversal_search",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_transversal_block_tiles_the_quotient_and_color_formula_matches():
    colors = 5
    phases = [0, 1, 3]
    block = MODULE.transversal_block(phases, colors)
    assert block.tolist() == [0, 4, 11]
    verification = MODULE.verify_transversal_coloring(
        np.asarray([[1, 0]], dtype=np.int64),
        [3, 5],
        phases,
        colors,
    )
    assert verification["partitions_quotient"]
    assert verification["color_formula_matches"]
    assert verification["difference_residues"] == sorted(
        {
            int((left - right) % 15)
            for left in block
            for right in block
        }
    )


def test_conflict_mask_is_exact_for_arbitrary_block_difference_set():
    phases = [0, 1, 3]
    forbidden = np.arange(15, dtype=np.int64)[:, None]
    mask = MODULE.transversal_conflict_mask(
        forbidden,
        [1],
        phases,
        5,
    )
    assert np.flatnonzero(mask).tolist() == (
        MODULE.transversal_difference_residues(phases, 5).tolist()
    )


def test_row_and_phase_coordinate_scores_match_direct_recounts():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1], [2, 3]],
        dtype=np.int64,
    )
    row = np.asarray([4, 7], dtype=np.int64)
    phases = np.asarray([0, 1, 3], dtype=np.int64)
    ratios = np.asarray([0.1, 0.4, 0.7, 0.9])
    hard = ratios < 0.5
    weights = (1.0 - ratios) ** 2

    row_scores = MODULE.transversal_row_coordinate_score_vectors(
        forbidden,
        row,
        phases,
        5,
        1,
        hard_mask=hard,
        weights=weights,
    )
    direct_row = []
    for value in range(15):
        candidate = row.copy()
        candidate[1] = value
        mask = MODULE.transversal_conflict_mask(
            forbidden,
            candidate,
            phases,
            5,
        )
        if MODULE.primitive_cyclic_row(candidate, 15):
            direct_row.append(
                (
                    int(np.count_nonzero(mask & hard)),
                    int(np.count_nonzero(mask)),
                    float(weights[mask].sum()),
                )
            )
        else:
            direct_row.append((len(forbidden) + 1,) * 2 + (np.inf,))
    for value, expected in enumerate(direct_row):
        assert row_scores[0][value] == expected[0]
        assert row_scores[1][value] == expected[1]
        assert np.isclose(row_scores[2][value], expected[2])

    phase_scores = MODULE.transversal_phase_score_vectors(
        forbidden,
        row,
        phases,
        5,
        1,
        hard_mask=hard,
        weights=weights,
    )
    for value in range(5):
        candidate = phases.copy()
        candidate[1] = value
        mask = MODULE.transversal_conflict_mask(
            forbidden,
            row,
            candidate,
            5,
        )
        assert phase_scores[0][value] == np.count_nonzero(mask & hard)
        assert phase_scores[1][value] == np.count_nonzero(mask)
        assert np.isclose(
            phase_scores[2][value],
            weights[mask].sum(),
        )


def test_alternating_descent_finds_small_transversal_coloring():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    result = MODULE.alternating_transversal_descent(
        forbidden,
        [0.0, 0.5, 0.9],
        5,
        2,
        hard_ratio=0.6,
        restarts=10,
        sweeps=8,
        top=4,
        seed=71,
    )
    assert result["success"]
    assert result["verification"]["valid"]


def test_pair_escape_never_misreports_its_score():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1], [2, 3]],
        dtype=np.int64,
    )
    ratios = np.asarray([0.1, 0.4, 0.7, 0.9])
    hard = ratios < 0.5
    weights = (1.0 - ratios) ** 2
    row, phases, key = MODULE.transversal_pair_escape(
        forbidden,
        [4, 7],
        [0, 1, 3],
        5,
        hard_mask=hard,
        weights=weights,
        pair_top=4,
        pair_trials=6,
        rng=np.random.default_rng(91),
    )
    mask = MODULE.transversal_conflict_mask(
        forbidden,
        row,
        phases,
        5,
    )
    assert MODULE.primitive_cyclic_row(row, 15)
    assert key[:2] == (
        int(np.count_nonzero(mask & hard)),
        int(np.count_nonzero(mask)),
    )
    assert np.isclose(key[2], weights[mask].sum())


def test_joint_highs_mip_finds_and_independently_verifies_small_coloring():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    result = MODULE.joint_transversal_highs_mip(
        forbidden,
        5,
        2,
        time_limit=5.0,
    )
    assert result["feasible"]
    assert result["verification"]["valid"]
    assert result["phases"][0] == 0


def test_joint_highs_mip_proves_impossible_one_dimensional_case():
    result = MODULE.joint_transversal_highs_mip(
        np.asarray([[1], [3]], dtype=np.int64),
        3,
        2,
        time_limit=5.0,
    )
    assert result["status"] == "INFEASIBLE"
    assert result["proven_infeasible"]
    assert not result["feasible"]


def test_joint_cpsat_matches_highs_on_feasible_and_infeasible_cases():
    feasible_forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    feasible = MODULE.joint_transversal_cpsat(
        feasible_forbidden,
        5,
        2,
        time_limit=5.0,
        workers=1,
    )
    assert feasible["feasible"]
    assert feasible["verification"]["valid"]

    infeasible = MODULE.joint_transversal_cpsat(
        np.asarray([[1], [3]], dtype=np.int64),
        3,
        2,
        time_limit=5.0,
        workers=1,
    )
    assert infeasible["status"] == "INFEASIBLE"
    assert infeasible["proven_infeasible"]

    soft = MODULE.joint_transversal_cpsat(
        np.asarray([[1], [3]], dtype=np.int64),
        3,
        2,
        time_limit=5.0,
        workers=1,
        minimize_conflicts=True,
    )
    assert soft["feasible"]
    assert not soft["valid_coloring"]
    assert soft["status"] == "OPTIMAL"
    assert soft["exact_conflict_count"] >= 1
    assert soft["objective_value"] >= 1


def test_cegar_adds_full_conflicts_and_returns_verified_coloring():
    forbidden = np.asarray(
        [[1, 0], [0, 1], [1, 1]],
        dtype=np.int64,
    )
    ratios = np.asarray([0.0, 0.5, 0.9])
    result = MODULE.transversal_cegar(
        forbidden,
        ratios,
        5,
        2,
        initial_ratio=0.0,
        max_rounds=4,
        time_limit=5.0,
    )
    assert result["outcome"] == "COLORING"
    assert result["valid_combinatorial_witness"]
    assert result["coloring"]["verification"]["valid"]
    assert result["final_core_indices"] == sorted(
        result["final_core_indices"]
    )
