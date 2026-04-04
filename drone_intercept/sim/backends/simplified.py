"""Simplified first-order dynamics backend (no PX4/Gazebo dependency)."""

from __future__ import annotations

import numpy as np

from drone_intercept.sim.backends.base import PhysicsBackend

_MAX_VEL = 10.0        # m/s per axis
_MAX_YAW_RATE = 2.0    # rad/s
_VEL_TAU = 0.3         # velocity tracking time constant (s)


class SimplifiedBackend(PhysicsBackend):
    """First-order velocity lag with Euler integration.

    Mimics an autopilot velocity controller without any external
    simulator dependency. Designed to be swapped for PX4 + Gazebo later.
    """

    def reset(self, rng: np.random.Generator) -> None:
        self.position = np.array(
            [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(2.0, 4.0)],
            dtype=np.float32,
        )
        self.velocity = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.battery = 1.0

    def step(self, action: np.ndarray, dt: float) -> None:
        vel_cmd = action[:3]
        yaw_rate_cmd = action[3]

        # First-order velocity tracking
        alpha = 1.0 - np.exp(-dt / _VEL_TAU)
        self.velocity = (
            self.velocity + alpha * (vel_cmd - self.velocity)
        ).astype(np.float32)

        # Integrate position
        self.position = (self.position + self.velocity * dt).astype(np.float32)

        # Yaw
        self.yaw += float(yaw_rate_cmd) * dt

        # Battery drain (simple linear)
        self.battery = max(0.0, self.battery - 0.0002)

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.array([-_MAX_VEL, -_MAX_VEL, -_MAX_VEL, -_MAX_YAW_RATE], dtype=np.float32)
        high = np.array([_MAX_VEL, _MAX_VEL, _MAX_VEL, _MAX_YAW_RATE], dtype=np.float32)
        return low, high
