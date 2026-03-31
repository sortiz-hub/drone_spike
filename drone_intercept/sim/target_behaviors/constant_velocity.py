"""Constant-velocity target — moves in a straight line."""

from __future__ import annotations

import numpy as np

from drone_intercept.sim.target_behaviors.base import TargetBehavior


class ConstantVelocityTarget(TargetBehavior):
    """Target that flies at constant velocity in a random direction."""

    def reset(self, rng: np.random.Generator) -> None:
        # Spawn 15-30m away from origin, at ~same altitude
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(15.0, 30.0)
        self.position = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), rng.uniform(1.5, 4.0)],
            dtype=np.float32,
        )
        # Random heading
        heading = rng.uniform(0, 2 * np.pi)
        self.velocity = np.array(
            [self.speed * np.cos(heading), self.speed * np.sin(heading), 0.0],
            dtype=np.float32,
        )

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt
