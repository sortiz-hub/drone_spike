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
) -> float:
    # Distance shaping — incentivise closing in
    reward = -0.1 * distance

    # Control effort penalty
    control_effort = float(np.linalg.norm(action[:3]))
    reward -= 0.01 * control_effort

    # Collision-risk proxy: low-altitude penalty
    if altitude < min_altitude * 2:
        reward -= 0.05 * max(0.0, min_altitude * 2 - altitude)

    # Terminal bonuses
    if captured:
        reward += 100.0
    if crashed:
        reward -= 100.0

    return reward
