"""Reward function for the interception environment."""

from __future__ import annotations

import numpy as np


def compute_reward(
    distance: float,
    action: np.ndarray,
    captured: bool,
    crashed: bool,
    altitude: float,
    min_altitude: float = 0.5,
    min_obstacle_distance: float | None = None,
    obstacle_crashed: bool = False,
) -> float:
    # Distance shaping — incentivise closing in
    reward = -0.1 * distance

    # Control effort penalty
    control_effort = float(np.linalg.norm(action[:3]))
    reward -= 0.01 * control_effort

    # Collision-risk proxy: low-altitude penalty
    if altitude < min_altitude * 2:
        reward -= 0.05 * max(0.0, min_altitude * 2 - altitude)

    # Obstacle proximity penalty (Phase 3)
    if min_obstacle_distance is not None and min_obstacle_distance < 3.0:
        reward -= 0.1 * max(0.0, 3.0 - min_obstacle_distance)

    # Terminal bonuses
    if captured:
        reward += 100.0
    if crashed or obstacle_crashed:
        reward -= 100.0

    return reward
