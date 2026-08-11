"""
Tests for scaling policies.

The two properties that matter for an autoscaler are that it respects its
configured bounds and that it does not oscillate, so those get the most
attention here.
"""

import pytest

from quantum_auto_scaler.scaling.policies import PolicyType, ScalingPolicy


@pytest.fixture
def policy():
    return ScalingPolicy(PolicyType.REACTIVE, min_scale=2, max_scale=10)


# -- Configuration validation -----------------------------------------------


def test_inverted_thresholds_are_rejected():
    with pytest.raises(ValueError, match="scale_down_threshold"):
        ScalingPolicy(scale_up_threshold=0.2, scale_down_threshold=0.9)


def test_equal_thresholds_are_rejected():
    with pytest.raises(ValueError):
        ScalingPolicy(scale_up_threshold=0.5, scale_down_threshold=0.5)


def test_min_scale_below_one_is_rejected():
    with pytest.raises(ValueError, match="min_scale"):
        ScalingPolicy(min_scale=0)


def test_max_below_min_is_rejected():
    with pytest.raises(ValueError, match="max_scale"):
        ScalingPolicy(min_scale=10, max_scale=5)


# -- Bounds are actually applied --------------------------------------------


def test_scale_up_is_capped_at_max(policy):
    """Regression: max_scale was stored but never applied."""
    result = policy.recommend(current_scale=9, current_util=0.99)

    assert result["target_scale"] == 10
    assert result["clamped"] is True
    assert result["at_max"] is True


def test_scale_down_is_floored_at_min(policy):
    result = policy.recommend(current_scale=3, current_util=0.01)

    assert result["target_scale"] >= 2
    assert result["at_min"] is True


def test_raw_target_is_reported_alongside_the_clamp(policy):
    result = policy.recommend(current_scale=10, current_util=0.99)

    assert result["raw_target"] > result["target_scale"]


def test_action_becomes_hold_when_already_at_the_ceiling(policy):
    result = policy.recommend(current_scale=10, current_util=0.99)

    assert result["action"] == "hold"
    assert result["suppressed_reason"] == "at_bound"


def test_within_bounds_is_not_reported_as_clamped(policy):
    result = policy.recommend(current_scale=4, current_util=0.9)

    assert result["clamped"] is False
    assert result["action"] == "scale_up"


# -- Cooldown / flapping ----------------------------------------------------


def test_second_action_inside_cooldown_is_suppressed():
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=60)
    policy.recommend(current_scale=4, current_util=0.95, now=1000.0)

    second = policy.recommend(current_scale=6, current_util=0.05, now=1010.0)

    assert second["action"] == "hold"
    assert second["suppressed_reason"] == "cooldown"


def test_action_allowed_once_cooldown_expires():
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=60)
    policy.recommend(current_scale=4, current_util=0.95, now=1000.0)

    later = policy.recommend(current_scale=6, current_util=0.05, now=1100.0)

    assert later["action"] == "scale_down"


def test_alternating_load_does_not_flap_under_cooldown():
    """The canonical autoscaler failure: oscillating on either side of a threshold."""
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=30)
    scale, actions = 4, []
    for i in range(10):
        util = 0.95 if i % 2 == 0 else 0.05
        result = policy.recommend(scale, util, now=1000.0 + i)
        scale = result["target_scale"]
        actions.append(result["action"])

    assert actions.count("hold") >= 8


def test_no_cooldown_means_every_decision_applies():
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=0)
    first = policy.recommend(4, 0.95, now=1000.0)
    second = policy.recommend(6, 0.05, now=1000.5)

    assert first["action"] == "scale_up"
    assert second["action"] == "scale_down"


def test_hold_does_not_start_a_cooldown():
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=60)
    policy.recommend(4, 0.5, now=1000.0)  # hold

    result = policy.recommend(4, 0.95, now=1001.0)
    assert result["action"] == "scale_up"


def test_seconds_until_ready_counts_down():
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=60)
    policy.recommend(4, 0.95, now=1000.0)

    assert policy.seconds_until_ready(now=1020.0) == pytest.approx(40.0)


def test_seconds_until_ready_is_zero_without_cooldown(policy):
    assert policy.seconds_until_ready() == 0.0


def test_reset_clears_cooldown_and_history():
    policy = ScalingPolicy(min_scale=1, max_scale=100, cooldown_seconds=60)
    policy.recommend(4, 0.95, now=1000.0)
    policy.reset()

    assert policy.history == []
    assert policy.recommend(4, 0.05, now=1001.0)["action"] == "scale_down"


def test_history_records_every_decision(policy):
    policy.recommend(4, 0.9)
    policy.recommend(4, 0.5)

    assert len(policy.history) == 2


# -- Policy behaviour -------------------------------------------------------


def test_reactive_scales_up_above_threshold(policy):
    assert policy.evaluate(0.95)["action"] == "scale_up"


def test_reactive_scales_down_below_threshold(policy):
    assert policy.evaluate(0.1)["action"] == "scale_down"


def test_reactive_holds_between_thresholds(policy):
    assert policy.evaluate(0.5)["action"] == "hold"


def test_predictive_weights_the_forecast():
    """A rising forecast should trigger a scale-up that current load alone would not."""
    reactive = ScalingPolicy(PolicyType.REACTIVE)
    predictive = ScalingPolicy(PolicyType.PREDICTIVE)

    # 0.6 is below the 0.8 up-threshold, so reactive holds. Blended with a
    # forecast of 1.0 the effective figure is 0.84, which crosses it.
    assert reactive.evaluate(0.6)["action"] == "hold"
    assert predictive.evaluate(0.6, predicted_util=1.0)["action"] == "scale_up"


def test_predictive_ignores_a_forecast_that_stays_low():
    p = ScalingPolicy(PolicyType.PREDICTIVE)
    assert p.evaluate(0.2, predicted_util=0.5)["action"] == "hold"


def test_quantum_policy_considers_utilization():
    """Regression: the quantum branch ignored `current` entirely."""
    p = ScalingPolicy(PolicyType.QUANTUM)

    saturated = p.evaluate(0.99, quantum_score=0.25)
    assert saturated["action"] != "scale_down"


def test_quantum_policy_scales_up_on_high_score():
    p = ScalingPolicy(PolicyType.QUANTUM)
    assert p.evaluate(0.8, quantum_score=0.9)["action"] == "scale_up"


def test_hybrid_blends_all_three_signals():
    p = ScalingPolicy(PolicyType.HYBRID)
    assert p.evaluate(0.9, predicted_util=0.9, quantum_score=0.9)["action"] == "scale_up"
    assert p.evaluate(0.05, predicted_util=0.05, quantum_score=0.05)["action"] == "scale_down"


def test_every_policy_type_returns_a_known_action():
    for ptype in PolicyType:
        p = ScalingPolicy(ptype, min_scale=1, max_scale=50)
        result = p.recommend(5, 0.5, 0.5, 0.5)
        assert result["action"] in {"scale_up", "scale_down", "hold"}
        assert 1 <= result["target_scale"] <= 50
