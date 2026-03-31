"""Waypoint-following target — moves between random waypoints."""

from __future__ import annotations

import numpy as np

from drone_intercept.sim.target_behaviors.base import TargetBehavior

_NUM_WAYPOINTS = 6
_WAYPOINT_RADIUS = 30.0
_ARRIVAL_DIST = 1.0


class WaypointTarget(TargetBehavior):
    """Target that flies between randomly-generated waypoints."""

    def reset(self, rng: np.random.Generator) -> None:
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(15.0, 30.0)
        self.position = np.array(
            [dist * np.cos(angle), dist * np.sin(angle), rng.uniform(1.5, 4.0)],
            dtype=np.float32,
        )
        self._waypoints = [
            np.array(
                [
                    rng.uniform(-_WAYPOINT_RADIUS, _WAYPOINT_RADIUS),
                    rng.uniform(-_WAYPOINT_RADIUS, _WAYPOINT_RADIUS),
                    rng.uniform(1.5, 5.0),
                ],
                dtype=np.float32,
            )
            for _ in range(_NUM_WAYPOINTS)
        ]
        self._wp_idx = 0
        self._update_velocity()

    def step(self, dt: float) -> None:
        self.position = self.position + self.velocity * dt
        # Check arrival at current waypoint
        to_wp = self._waypoints[self._wp_idx] - self.position
        if np.linalg.norm(to_wp) < _ARRIVAL_DIST:
            self._wp_idx = (self._wp_idx + 1) % len(self._waypoints)
            self._update_velocity()

    def _update_velocity(self) -> None:
        direction = self._waypoints[self._wp_idx] - self.position
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            self.velocity = (direction / dist * self.speed).astype(np.float32)
        else:
            self.velocity = np.zeros(3, dtype=np.float32)
