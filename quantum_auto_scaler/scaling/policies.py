"""
Brion Quantum - Scaling Policies v1.0
=======================================
Configurable scaling policies: reactive, predictive, and quantum-enhanced.

Developed by Brion Quantum AI Team
"""

__version__ = "1.0.0"

import numpy as np
from typing import Dict, Any, Optional
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
                 min_scale: int = 1, max_scale: int = 100):
        self.policy_type = policy_type
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_scale = min_scale
        self.max_scale = max_scale

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
        if q_score > 0.6:
            return {"action": "scale_up", "factor": 1.0 + q_score}
        elif q_score < 0.3:
            return {"action": "scale_down", "factor": max(0.5, q_score * 2)}
        return {"action": "hold", "factor": 1.0}

    def _hybrid(self, current: float, predicted: float,
                q_score: float) -> Dict[str, Any]:
        combined = 0.3 * current + 0.3 * predicted + 0.4 * q_score
        if combined > self.scale_up_threshold:
            return {"action": "scale_up", "factor": 1.0 + (combined - self.scale_up_threshold) * 2}
        elif combined < self.scale_down_threshold:
            return {"action": "scale_down", "factor": 0.75}
        return {"action": "hold", "factor": 1.0}
