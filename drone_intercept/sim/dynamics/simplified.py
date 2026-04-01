"""Simplified first-order velocity-lag dynamics (no external dependencies)."""

from __future__ import annotations

import numpy as np

from drone_intercept.sim.dynamics.base import DynamicsBackend, DynamicsState

# Dynamics constants
_VEL_TAU = 0.3   # velocity tracking time constant (s)


class SimplifiedDynamics(DynamicsBackend):
    """First-order velocity tracking model that mimics an autopilot.

    The drone's velocity converges toward the commanded velocity with a
    time constant of 0.3 s (exponential lag). Position is integrated via
    explicit Euler. Battery drains linearly.

    This is the default backend — fast, deterministic, zero dependencies.
    """

    def __init__(self, vel_tau: float = _VEL_TAU) -> None:
        self._vel_tau = vel_tau
        self._state = DynamicsState()

    def reset(self, rng: np.random.Generator) -> DynamicsState:
        self._state = DynamicsState(
            position=np.array(
                [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(2.0, 4.0)],
                dtype=np.float32,
            ),
            velocity=np.zeros(3, dtype=np.float32),
            yaw=0.0,
            battery=1.0,
        )
        return self._state

    def step(
        self,
        vel_cmd: np.ndarray,
        yaw_rate_cmd: float,
        dt: float,
    ) -> DynamicsState:
        alpha = 1.0 - np.exp(-dt / self._vel_tau)

        self._state.velocity = (
            self._state.velocity + alpha * (vel_cmd - self._state.velocity)
        ).astype(np.float32)

        self._state.position = (
            self._state.position + self._state.velocity * dt
        ).astype(np.float32)

        self._state.yaw += float(yaw_rate_cmd) * dt
        self._state.battery = max(0.0, self._state.battery - 0.0002)

        return self._state

    def close(self) -> None:
        pass  # nothing to release
