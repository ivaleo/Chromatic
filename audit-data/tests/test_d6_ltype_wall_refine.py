import numpy as np

from chromatic_research.campaigns.d6_ltype_wall_refine import constrained_max_min_step


def test_constrained_step_respects_wall_and_maximizes_active_value() -> None:
    step, predicted = constrained_max_min_step(
        np.asarray([0.5]),
        np.asarray([[1.0]]),
        1.0,
        current_wall_slack=-0.2,
        wall_gradient_values=np.asarray([1.0]),
        wall_margin=0.1,
    )
    assert np.allclose(step, [0.1], atol=1e-12)
    assert np.isclose(predicted, 0.6, atol=1e-12)
    assert -0.2 + step[0] <= -0.1 + 1e-12


def test_constrained_step_can_move_deeper_when_objective_requires_it() -> None:
    step, predicted = constrained_max_min_step(
        np.asarray([0.25]),
        np.asarray([[-2.0]]),
        0.4,
        current_wall_slack=-0.05,
        wall_gradient_values=np.asarray([1.0]),
        wall_margin=0.05,
    )
    assert np.allclose(step, [-0.4], atol=1e-12)
    assert np.isclose(predicted, 1.05, atol=1e-12)
