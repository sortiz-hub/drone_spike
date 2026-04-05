"""Reward functions for the interception environment.

Two modes:
  - "original": Static distance penalty. Weak shaping — agent mostly learns
    from the +100 capture spike.
  - "shaped": Delta-distance + proximity bonus. Gives a clear positive signal
    every step the drone closes in, and accelerating reward near capture range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Selects and tunes the reward function.

    Attributes:
        mode: "original" or "shaped".
        capture_bonus: Reward for successful capture.
        crash_penalty: Penalty for crashing (positive value, applied as negative).
        effort_weight: Penalty weight for control effort.
        proximity_scale: (shaped only) Scale for exponential proximity bonus.
        delta_weight: (shaped only) Weight for distance-closing reward.
    """

    mode: str = "shaped"
    capture_bonus: float | None = None   # None = use mode default
    crash_penalty: float | None = None   # None = use mode default
    effort_weight: float = 0.01
    # shaped mode
    proximity_scale: float = 5.0
    delta_weight: float = 1.0

    def effective_capture_bonus(self) -> float:
        if self.capture_bonus is not None:
            return self.capture_bonus
        return 100.0 if self.mode == "original" else 10.0

    def effective_crash_penalty(self) -> float:
        if self.crash_penalty is not None:
            return self.crash_penalty
        return 100.0 if self.mode == "original" else 10.0


def compute_reward(
    distance: float,
    action: np.ndarray,
    captured: bool,
    crashed: bool,
    altitude: float,
    prev_distance: float | None = None,
    min_altitude: float = 0.5,
    min_obstacle_distance: float | None = None,
    obstacle_crashed: bool = False,
    config: RewardConfig | None = None,
) -> float:
    cfg = config or RewardConfig()

    if cfg.mode == "original":
        return _reward_original(
            distance, action, captured, crashed, altitude,
            min_altitude, min_obstacle_distance, obstacle_crashed,
            cfg,
        )
    elif cfg.mode == "shaped":
        return _reward_shaped(
            distance, action, captured, crashed, altitude,
            prev_distance, min_altitude, min_obstacle_distance,
            obstacle_crashed, cfg,
        )
    else:
        raise ValueError(f"Unknown reward mode '{cfg.mode}'. Choose 'original' or 'shaped'.")


def _reward_original(
    distance: float,
    action: np.ndarray,
    captured: bool,
    crashed: bool,
    altitude: float,
    min_altitude: float,
    min_obstacle_distance: float | None,
    obstacle_crashed: bool,
    cfg: RewardConfig,
) -> float:
    """Original reward: static distance penalty + terminal bonuses."""
    # Distance shaping — incentivise closing in
    reward = -0.1 * distance

    # Control effort penalty
    control_effort = float(np.linalg.norm(action[:3]))
    reward -= cfg.effort_weight * control_effort

    # Collision-risk proxy: low-altitude penalty
    if altitude < min_altitude * 2:
        reward -= 0.05 * max(0.0, min_altitude * 2 - altitude)

    # Obstacle proximity penalty (Phase 3)
    if min_obstacle_distance is not None and min_obstacle_distance < 3.0:
        reward -= 0.1 * max(0.0, 3.0 - min_obstacle_distance)

    # Terminal bonuses (original: +100/-100)
    if captured:
        reward += cfg.effective_capture_bonus()
    if crashed or obstacle_crashed:
        reward -= cfg.effective_crash_penalty()

    return reward


def _reward_shaped(
    distance: float,
    action: np.ndarray,
    captured: bool,
    crashed: bool,
    altitude: float,
    prev_distance: float | None,
    min_altitude: float,
    min_obstacle_distance: float | None,
    obstacle_crashed: bool,
    cfg: RewardConfig,
) -> float:
    """Shaped reward: delta-distance + exponential proximity + terminal bonuses.

    Key improvements over original:
    - Delta-distance: positive reward every step the drone closes in.
      At typical closing speed of ~5m/s with dt=0.1: ~0.5 reward/step.
    - Proximity bonus: exp(-d/scale) gives accelerating reward near target.
      At 5m: 0.37, at 2m: 0.67, at 1m: 0.82 — strong pull into capture range.
    - Together these give a smooth, informative gradient from spawn to capture.
    """
    reward = 0.0

    # Delta-distance: reward closing in, penalize drifting away
    if prev_distance is not None:
        reward += cfg.delta_weight * (prev_distance - distance)

    # Proximity bonus: exponential, peaks near target
    reward += np.exp(-distance / cfg.proximity_scale)

    # Control effort penalty
    control_effort = float(np.linalg.norm(action[:3]))
    reward -= cfg.effort_weight * control_effort

    # Low-altitude penalty
    if altitude < min_altitude * 2:
        reward -= 0.05 * max(0.0, min_altitude * 2 - altitude)

    # Obstacle proximity penalty (Phase 3)
    if min_obstacle_distance is not None and min_obstacle_distance < 3.0:
        reward -= 0.1 * max(0.0, 3.0 - min_obstacle_distance)

    # Terminal bonuses (shaped: +10/-10)
    if captured:
        reward += cfg.effective_capture_bonus()
    if crashed or obstacle_crashed:
        reward -= cfg.effective_crash_penalty()

    return reward
