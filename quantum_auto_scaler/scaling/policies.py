"""
Brion Quantum - Scaling Policies v1.0
=======================================
Configurable scaling policies: reactive, predictive, and quantum-enhanced.

Developed by Brion Quantum AI Team
"""
from __future__ import annotations


__version__ = "1.0.1"

try:
    import numpy as np
except ImportError:  # optional dependency: pip install numpy
    np = None
import time
from typing import Any, Dict, List, Optional
from enum import Enum


class PolicyType(Enum):
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    QUANTUM = "quantum"
    HYBRID = "hybrid"


class ScalingPolicy:
    """Base scaling policy with configurable thresholds."""

    def __init__(self, policy_type: PolicyType = PolicyType.REACTIVE,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 min_scale: int = 1, max_scale: int = 100,
                 cooldown_seconds: float = 0.0):
        if not 0.0 <= scale_down_threshold < scale_up_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= scale_down_threshold < scale_up_threshold <= 1; "
                f"got down={scale_down_threshold}, up={scale_up_threshold}"
            )
        if min_scale < 1:
            raise ValueError("min_scale must be at least 1")
        if max_scale < min_scale:
            raise ValueError("max_scale must be >= min_scale")

        self.policy_type = policy_type
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Minimum gap between two scaling actions. Utilization sitting near a
        # threshold otherwise flips the decision on every evaluation, and each
        # scale changes utilization enough to trip the opposite threshold --
        # the classic autoscaler oscillation.
        self.cooldown_seconds = cooldown_seconds
        self._last_action_at: Optional[float] = None
        self._last_action: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    def evaluate(self, current_util: float, predicted_util: float = 0.0,
                 quantum_score: float = 0.5) -> Dict[str, Any]:
        """Evaluate scaling policy and return recommendation."""
        if self.policy_type == PolicyType.REACTIVE:
            return self._reactive(current_util)
        elif self.policy_type == PolicyType.PREDICTIVE:
            return self._predictive(current_util, predicted_util)
        elif self.policy_type == PolicyType.QUANTUM:
            return self._quantum(current_util, quantum_score)
        else:
            return self._hybrid(current_util, predicted_util, quantum_score)

    # -- Bounded, cooldown-aware recommendation -----------------------------

    def recommend(self, current_scale: int, current_util: float,
                  predicted_util: float = 0.0, quantum_score: float = 0.5,
                  now: Optional[float] = None) -> Dict[str, Any]:
        """
        Turn a policy decision into a concrete, bounded target scale.

        `evaluate` returns only a multiplier, and min_scale/max_scale were
        previously stored but never applied, so a configured ceiling had no
        effect on anything. This clamps the result into [min_scale, max_scale]
        and suppresses actions inside the cooldown window.
        """
        now = time.time() if now is None else now
        decision = dict(self.evaluate(current_util, predicted_util, quantum_score))
        action = decision["action"]

        raw_target = int(round(current_scale * decision["factor"]))
        target = max(self.min_scale, min(self.max_scale, raw_target))

        # A scale action that cannot move the count is not an action.
        if target == current_scale and action != "hold":
            action = "hold"
            decision["suppressed_reason"] = (
                "at_bound" if raw_target != target else "no_change"
            )

        if action != "hold" and self._in_cooldown(now):
            action = "hold"
            target = current_scale
            decision["suppressed_reason"] = "cooldown"

        decision.update({
            "action": action,
            "current_scale": current_scale,
            "raw_target": raw_target,
            "target_scale": target,
            "clamped": raw_target != target,
            "at_min": target == self.min_scale,
            "at_max": target == self.max_scale,
            "timestamp": now,
        })

        if action != "hold":
            self._last_action_at = now
            self._last_action = action
        self.history.append(decision)
        return decision

    def _in_cooldown(self, now: float) -> bool:
        if self.cooldown_seconds <= 0 or self._last_action_at is None:
            return False
        return (now - self._last_action_at) < self.cooldown_seconds

    def seconds_until_ready(self, now: Optional[float] = None) -> float:
        """Seconds remaining before another scaling action is permitted."""
        if self.cooldown_seconds <= 0 or self._last_action_at is None:
            return 0.0
        now = time.time() if now is None else now
        return max(0.0, self.cooldown_seconds - (now - self._last_action_at))

    def reset(self):
        """Clear cooldown state and decision history."""
        self._last_action_at = None
        self._last_action = None
        self.history.clear()

    def _reactive(self, util: float) -> Dict[str, Any]:
        if util > self.scale_up_threshold:
            return {"action": "scale_up", "factor": 1.0 + (util - self.scale_up_threshold) * 2}
        elif util < self.scale_down_threshold:
            return {"action": "scale_down", "factor": 0.75}
        return {"action": "hold", "factor": 1.0}

    def _predictive(self, current: float, predicted: float) -> Dict[str, Any]:
        effective = 0.4 * current + 0.6 * predicted
        if effective > self.scale_up_threshold:
            return {"action": "scale_up", "factor": 1.0 + (effective - self.scale_up_threshold) * 2}
        elif effective < self.scale_down_threshold:
            return {"action": "scale_down", "factor": 0.75}
        return {"action": "hold", "factor": 1.0}

    def _quantum(self, current: float, q_score: float) -> Dict[str, Any]:
        # Blend the quantum score with observed utilization. Weighting the
        # score alone ignored `current` entirely, so this policy would scale
        # down a saturated cluster whenever the score happened to be low.
        combined = 0.4 * current + 0.6 * q_score
        if combined > 0.6:
            return {"action": "scale_up", "factor": 1.0 + combined}
        elif combined < 0.3:
            return {"action": "scale_down", "factor": max(0.5, combined * 2)}
        return {"action": "hold", "factor": 1.0}

    def _hybrid(self, current: float, predicted: float,
                q_score: float) -> Dict[str, Any]:
        combined = 0.3 * current + 0.3 * predicted + 0.4 * q_score
        if combined > self.scale_up_threshold:
            return {"action": "scale_up", "factor": 1.0 + (combined - self.scale_up_threshold) * 2}
        elif combined < self.scale_down_threshold:
            return {"action": "scale_down", "factor": 0.75}
        return {"action": "hold", "factor": 1.0}
