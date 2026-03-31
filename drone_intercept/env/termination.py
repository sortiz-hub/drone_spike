"""Episode termination conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TerminationConfig:
    capture_distance: float = 1.5
    capture_max_rel_speed: float = 2.0
    min_altitude: float = 0.3
    max_altitude: float = 50.0
    arena_radius: float = 100.0
    max_steps: int = 1000


@dataclass
class TerminationResult:
    terminated: bool = False
    truncated: bool = False
    reason: str = ""


def check_termination(
    drone_pos: np.ndarray,
    drone_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    step_count: int,
    cfg: TerminationConfig,
) -> TerminationResult:
    rel_pos = target_pos - drone_pos
    distance = float(np.linalg.norm(rel_pos))
    rel_speed = float(np.linalg.norm(target_vel - drone_vel))

    # Capture
    if distance < cfg.capture_distance and rel_speed < cfg.capture_max_rel_speed:
        return TerminationResult(terminated=True, reason="capture")

    # Crash — ground collision
    if drone_pos[2] < cfg.min_altitude:
        return TerminationResult(terminated=True, reason="crash_ground")

    # Crash — too high
    if drone_pos[2] > cfg.max_altitude:
        return TerminationResult(terminated=True, reason="crash_ceiling")

    # Out of bounds (horizontal)
    horiz_dist = float(np.linalg.norm(drone_pos[:2]))
    if horiz_dist > cfg.arena_radius:
        return TerminationResult(terminated=True, reason="out_of_bounds")

    # Timeout
    if step_count >= cfg.max_steps:
        return TerminationResult(truncated=True, reason="timeout")

    return TerminationResult()
