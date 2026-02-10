"""
Brion Quantum - TPU Scaler v1.0
=================================
Google TPU-specific auto-scaling for the Brion Quantum TRC allocation.
Manages spot vs on-demand TPU allocation across multiple zones.

TPU Resources (Google TRC Program):
  - 64 spot v6e chips: europe-west4-a
  - 32 on-demand v4 chips: us-central2-b
  - 64 spot v5e chips: europe-west4-b
  - 64 spot v6e chips: us-east1-d
  - 64 spot v5e chips: us-central1-a
  - 32 spot v4 chips: us-central2-b

Developed by Brion Quantum AI Team
"""

__version__ = "1.0.0"

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class TPUType(Enum):
    V4 = "v4"
    V5E = "v5e"
    V6E = "v6e"


class TPUAllocationMode(Enum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"


class TPUZone:
    """Represents a TPU zone allocation from the TRC program."""

    def __init__(self, zone: str, tpu_type: TPUType, mode: TPUAllocationMode,
                 total_chips: int):
        self.zone = zone
        self.tpu_type = tpu_type
        self.mode = mode
        self.total_chips = total_chips
        self.allocated_chips: int = 0
        self.active: bool = False

    @property
    def available_chips(self) -> int:
        return self.total_chips - self.allocated_chips

    @property
    def utilization(self) -> float:
        return self.allocated_chips / max(1, self.total_chips)

    def allocate(self, num_chips: int) -> int:
        """Allocate chips, return actual allocated count."""
        can_allocate = min(num_chips, self.available_chips)
        self.allocated_chips += can_allocate
        return can_allocate

    def release(self, num_chips: int) -> int:
        """Release chips, return actual released count."""
        can_release = min(num_chips, self.allocated_chips)
        self.allocated_chips -= can_release
        return can_release

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone": self.zone,
            "tpu_type": self.tpu_type.value,
            "mode": self.mode.value,
            "total_chips": self.total_chips,
            "allocated_chips": self.allocated_chips,
            "available_chips": self.available_chips,
            "utilization": self.utilization,
            "active": self.active
        }


class TPUScaler:
    """
    TPU-specific auto-scaler for Brion Quantum TRC allocation.

    Manages TPU chip allocation across multiple zones and generations,
    preferring spot instances for cost efficiency while maintaining
    on-demand for critical workloads.
    """

    def __init__(self):
        # Initialize TRC allocations
        self.zones: Dict[str, TPUZone] = {
            "europe-west4-a": TPUZone("europe-west4-a", TPUType.V6E,
                                       TPUAllocationMode.SPOT, 64),
            "us-central2-b-ondemand": TPUZone("us-central2-b", TPUType.V4,
                                                TPUAllocationMode.ON_DEMAND, 32),
            "europe-west4-b": TPUZone("europe-west4-b", TPUType.V5E,
                                       TPUAllocationMode.SPOT, 64),
            "us-east1-d": TPUZone("us-east1-d", TPUType.V6E,
                                   TPUAllocationMode.SPOT, 64),
            "us-central1-a": TPUZone("us-central1-a", TPUType.V5E,
                                      TPUAllocationMode.SPOT, 64),
            "us-central2-b-spot": TPUZone("us-central2-b", TPUType.V4,
                                           TPUAllocationMode.SPOT, 32),
        }

        # Scaling history
        self.allocation_history: List[Dict[str, Any]] = []

        # Performance by zone
        self.zone_performance: Dict[str, List[float]] = {}

    @property
    def total_chips(self) -> int:
        return sum(z.total_chips for z in self.zones.values())

    @property
    def total_allocated(self) -> int:
        return sum(z.allocated_chips for z in self.zones.values())

    @property
    def total_available(self) -> int:
        return sum(z.available_chips for z in self.zones.values())

    def allocate_chips(self, num_chips: int,
                       prefer_type: Optional[TPUType] = None,
                       prefer_spot: bool = True,
                       critical: bool = False) -> Dict[str, Any]:
        """
        Allocate TPU chips across available zones.

        Args:
            num_chips: Number of chips to allocate
            prefer_type: Preferred TPU type (v4, v5e, v6e)
            prefer_spot: Prefer spot instances (cheaper)
            critical: If True, prefer on-demand for reliability

        Returns:
            Allocation result with zone breakdown.
        """
        remaining = num_chips
        allocations = {}

        # Sort zones by preference
        sorted_zones = self._sort_zones(prefer_type, prefer_spot, critical)

        for zone_key in sorted_zones:
            if remaining <= 0:
                break

            zone = self.zones[zone_key]
            if zone.available_chips > 0:
                allocated = zone.allocate(remaining)
                if allocated > 0:
                    allocations[zone_key] = allocated
                    remaining -= allocated
                    zone.active = True

        result = {
            "requested": num_chips,
            "allocated": num_chips - remaining,
            "shortfall": remaining,
            "zone_allocations": allocations,
            "success": remaining == 0,
            "timestamp": datetime.now().isoformat()
        }

        self.allocation_history.append(result)
        return result

    def release_chips(self, zone_key: Optional[str] = None,
                      num_chips: Optional[int] = None) -> Dict[str, Any]:
        """
        Release TPU chips.

        Args:
            zone_key: Specific zone to release from (None = release from all)
            num_chips: Number to release (None = release all)
        """
        released = {}

        if zone_key and zone_key in self.zones:
            zone = self.zones[zone_key]
            count = num_chips or zone.allocated_chips
            actual = zone.release(count)
            released[zone_key] = actual
            if zone.allocated_chips == 0:
                zone.active = False
        else:
            # Release from all zones
            remaining = num_chips or self.total_allocated
            for key, zone in self.zones.items():
                if remaining <= 0:
                    break
                actual = zone.release(remaining)
                if actual > 0:
                    released[key] = actual
                    remaining -= actual
                if zone.allocated_chips == 0:
                    zone.active = False

        return {
            "released": released,
            "total_released": sum(released.values()),
            "timestamp": datetime.now().isoformat()
        }

    def _sort_zones(self, prefer_type, prefer_spot, critical):
        """Sort zones by allocation preference."""
        def score(key):
            zone = self.zones[key]
            s = 0.0

            # Available chips
            s += zone.available_chips * 0.01

            # Type preference
            if prefer_type and zone.tpu_type == prefer_type:
                s += 10.0

            # Newer TPU types get preference by default
            type_scores = {TPUType.V6E: 3, TPUType.V5E: 2, TPUType.V4: 1}
            s += type_scores.get(zone.tpu_type, 0)

            # Spot vs on-demand preference
            if critical and zone.mode == TPUAllocationMode.ON_DEMAND:
                s += 5.0
            elif prefer_spot and zone.mode == TPUAllocationMode.SPOT:
                s += 5.0

            # Performance history bonus
            perf = self.zone_performance.get(key, [])
            if perf:
                s += np.mean(perf[-5:]) * 2.0

            return s

        return sorted(self.zones.keys(), key=score, reverse=True)

    def record_performance(self, zone_key: str, performance: float):
        """Record zone performance for future allocation decisions."""
        if zone_key not in self.zone_performance:
            self.zone_performance[zone_key] = []
        self.zone_performance[zone_key].append(performance)

    def auto_scale(self, demand: float) -> Dict[str, Any]:
        """
        Auto-scale TPU allocation based on demand.

        Args:
            demand: Normalized demand (0-1)

        Returns:
            Scaling action taken.
        """
        target_chips = int(demand * self.total_chips)
        current = self.total_allocated

        if target_chips > current:
            # Scale up
            return self.allocate_chips(target_chips - current)
        elif target_chips < current * 0.7:
            # Scale down (with hysteresis)
            return self.release_chips(num_chips=current - target_chips)
        else:
            return {"action": "hold", "current": current, "target": target_chips}

    def report(self) -> Dict[str, Any]:
        """Generate TPU scaling report."""
        return {
            "version": __version__,
            "total_chips": self.total_chips,
            "total_allocated": self.total_allocated,
            "total_available": self.total_available,
            "utilization": self.total_allocated / max(1, self.total_chips),
            "zones": {k: z.to_dict() for k, z in self.zones.items()},
            "active_zones": sum(1 for z in self.zones.values() if z.active),
            "allocation_events": len(self.allocation_history)
        }
