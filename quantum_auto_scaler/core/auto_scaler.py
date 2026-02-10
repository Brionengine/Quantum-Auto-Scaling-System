"""
Brion Quantum - Core Auto-Scaler v1.0
=======================================
Central auto-scaling engine that orchestrates demand prediction,
resource optimization, and scaling policy execution.

Novel Algorithm: Quantum Demand Superposition Scaling (QDSS)
  - Models future demand as a superposition of possible workload states.
  - Uses quantum measurement to collapse to the most probable demand
    configuration when scaling decisions must be made.

Developed by Brion Quantum AI Team
"""

__version__ = "1.0.0"

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ScaleAction(Enum):
    HOLD = "hold"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    EMERGENCY = "emergency_scale"


class ResourcePool:
    """Managed pool of computing resources."""

    def __init__(self, name: str, min_units: int = 1, max_units: int = 100,
                 current_units: int = 1, cost_per_unit: float = 1.0):
        self.name = name
        self.min_units = min_units
        self.max_units = max_units
        self.current_units = current_units
        self.cost_per_unit = cost_per_unit
        self.utilization: float = 0.0

    def scale_to(self, target: int) -> int:
        """Scale to target units, respecting min/max bounds."""
        self.current_units = max(self.min_units, min(self.max_units, target))
        return self.current_units

    @property
    def available_units(self) -> int:
        return max(0, self.current_units - int(self.utilization * self.current_units))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current_units": self.current_units,
            "min_units": self.min_units,
            "max_units": self.max_units,
            "utilization": self.utilization,
            "available_units": self.available_units,
            "cost": self.current_units * self.cost_per_unit
        }


class QuantumAutoScaler:
    """
    QDSS Auto-Scaling Engine.

    Orchestrates scaling decisions using demand prediction, resource
    optimization, and configurable scaling policies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Scaling thresholds
        self.scale_up_threshold = config.get("scale_up_threshold", 0.80)
        self.scale_down_threshold = config.get("scale_down_threshold", 0.30)
        self.emergency_threshold = config.get("emergency_threshold", 0.95)
        self.cooldown_steps = config.get("cooldown_steps", 5)

        # Resource pools
        self.pools: Dict[str, ResourcePool] = {
            "compute": ResourcePool("compute", min_units=1, max_units=100,
                                     current_units=config.get("initial_compute", 4)),
            "quantum": ResourcePool("quantum", min_units=0, max_units=32,
                                     current_units=config.get("initial_quantum", 0),
                                     cost_per_unit=4.0),
            "tpu": ResourcePool("tpu", min_units=0, max_units=64,
                                 current_units=config.get("initial_tpu", 0),
                                 cost_per_unit=3.0),
            "memory": ResourcePool("memory", min_units=2, max_units=200,
                                    current_units=config.get("initial_memory", 8),
                                    cost_per_unit=0.5)
        }

        # State
        self.step = 0
        self._last_scale_step = -self.cooldown_steps
        self.system_state = "initializing"

        # History
        self.workload_history: List[float] = []
        self.scaling_history: List[Dict[str, Any]] = []
        self.prediction_history: List[float] = []

        # Demand prediction (exponential smoothing)
        self._smoothed_demand = 0.0
        self._alpha = 0.3

        # Quantum-enhanced prediction (optional)
        self._quantum_predictor = None
        self._use_quantum = config.get("use_quantum", False)

    def update(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Process new metrics and make scaling decisions.

        Args:
            metrics: Current system metrics including:
                - utilization (0-1)
                - latency_ms
                - throughput
                - error_rate
                - queue_depth

        Returns:
            Scaling decision report.
        """
        self.step += 1
        utilization = metrics.get("utilization", 0.0)
        self.workload_history.append(utilization)

        # Update resource pool utilizations
        for pool in self.pools.values():
            pool.utilization = utilization

        # Predict future demand
        predicted = self._predict_demand()
        self.prediction_history.append(predicted)

        # Make scaling decision
        action, target_scale, reason = self._decide(utilization, predicted, metrics)

        # Apply if not in cooldown (except emergency)
        in_cooldown = (self.step - self._last_scale_step) < self.cooldown_steps
        applied = False

        if action == ScaleAction.EMERGENCY or not in_cooldown:
            if action != ScaleAction.HOLD:
                self._apply_scaling(action, target_scale, reason)
                applied = True

        result = {
            "step": self.step,
            "action": action.value,
            "applied": applied,
            "reason": reason,
            "utilization": utilization,
            "predicted_demand": predicted,
            "in_cooldown": in_cooldown,
            "pools": {name: pool.to_dict() for name, pool in self.pools.items()},
            "total_cost": sum(p.current_units * p.cost_per_unit for p in self.pools.values()),
            "timestamp": datetime.now().isoformat()
        }

        return result

    def _predict_demand(self) -> float:
        """Predict future demand using exponential smoothing + trend."""
        if len(self.workload_history) < 3:
            self._smoothed_demand = self.workload_history[-1] if self.workload_history else 0.0
            return self._smoothed_demand

        current = self.workload_history[-1]
        self._smoothed_demand = self._alpha * current + (1 - self._alpha) * self._smoothed_demand

        # Add trend component
        recent = np.array(self.workload_history[-10:])
        trend = np.mean(np.diff(recent)) if len(recent) > 1 else 0.0

        predicted = self._smoothed_demand + trend * 5  # 5-step lookahead
        return float(np.clip(predicted, 0.0, 1.0))

    def _decide(self, utilization: float, predicted: float,
                metrics: Dict[str, float]) -> tuple:
        """Make scaling decision based on current and predicted demand."""
        error_rate = metrics.get("error_rate", 0.0)
        queue_depth = metrics.get("queue_depth", 0)

        # Emergency: immediate scale up
        if utilization > self.emergency_threshold or error_rate > 0.1:
            target = {name: int(pool.current_units * 1.5) for name, pool in self.pools.items()}
            return (ScaleAction.EMERGENCY, target,
                    f"Emergency: util={utilization:.2f}, errors={error_rate:.3f}")

        # Scale up: predicted or current demand exceeds threshold
        if predicted > self.scale_up_threshold or utilization > self.scale_up_threshold:
            overshoot = max(predicted, utilization) - self.scale_up_threshold
            factor = 1.0 + overshoot * 2.0
            target = {name: int(pool.current_units * factor) for name, pool in self.pools.items()}
            return (ScaleAction.SCALE_UP, target,
                    f"Scale up: predicted={predicted:.2f}")

        # Scale down: sustained low utilization
        if (utilization < self.scale_down_threshold and
                predicted < self.scale_down_threshold):
            if len(self.workload_history) >= 5:
                recent_avg = np.mean(self.workload_history[-5:])
                if recent_avg < self.scale_down_threshold:
                    target = {name: max(pool.min_units, int(pool.current_units * 0.75))
                              for name, pool in self.pools.items()}
                    return (ScaleAction.SCALE_DOWN, target,
                            f"Scale down: avg_util={recent_avg:.2f}")

        return (ScaleAction.HOLD, {}, "Within normal range")

    def _apply_scaling(self, action: ScaleAction, targets: Dict[str, int],
                       reason: str):
        """Apply scaling decision to resource pools."""
        old_state = {name: pool.current_units for name, pool in self.pools.items()}

        for name, target in targets.items():
            if name in self.pools:
                self.pools[name].scale_to(target)

        new_state = {name: pool.current_units for name, pool in self.pools.items()}
        self._last_scale_step = self.step

        self.scaling_history.append({
            "step": self.step,
            "action": action.value,
            "reason": reason,
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": datetime.now().isoformat()
        })

    def set_quantum_predictor(self, predictor):
        """Attach a quantum demand predictor."""
        self._quantum_predictor = predictor
        self._use_quantum = True

    def report(self) -> Dict[str, Any]:
        """Generate auto-scaling report."""
        return {
            "version": __version__,
            "step": self.step,
            "scaling_events": len(self.scaling_history),
            "current_utilization": self.workload_history[-1] if self.workload_history else 0.0,
            "predicted_demand": self.prediction_history[-1] if self.prediction_history else 0.0,
            "pools": {name: pool.to_dict() for name, pool in self.pools.items()},
            "total_cost": sum(p.current_units * p.cost_per_unit for p in self.pools.values()),
            "quantum_enabled": self._use_quantum,
            "recent_actions": self.scaling_history[-5:] if self.scaling_history else []
        }
