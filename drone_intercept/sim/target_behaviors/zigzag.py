"""Zigzag target — moves forward with periodic lateral direction changes."""

from __future__ import annotations

import numpy as np

from drone_intercept.sim.target_behaviors.base import TargetBehavior


class ZigzagTarget(TargetBehavior):
    """Target that zigzags: constant forward motion with periodic lateral flips."""

    def __init__(self, speed: float = 5.0, period: float = 3.0) -> None:
        super().__init__(speed=speed)
        self.period = period

    def reset(self, rng: np.random.Generator) -> None:
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(15.0, 30.0)
        self.position = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), rng.uniform(1.5, 4.0)],
            dtype=np.float32,
        )
        # Forward direction
        self._forward_angle = rng.uniform(0, 2 * np.pi)
        self._lateral_sign = 1.0
        self._timer = 0.0
        self._update_velocity()

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt
        self._timer += dt
        if self._timer >= self.period:
            self._timer -= self.period
            self._lateral_sign *= -1.0
            self._update_velocity()

    def _update_velocity(self) -> None:
        fwd = np.array(
            [np.cos(self._forward_angle), np.sin(self._forward_angle), 0.0]
        )
        lat = np.array(
            [-np.sin(self._forward_angle), np.cos(self._forward_angle), 0.0]
        )
        direction = 0.7 * fwd + 0.3 * self._lateral_sign * lat
        direction = direction / np.linalg.norm(direction)
        self.velocity = (direction * self.speed).astype(np.float32)
