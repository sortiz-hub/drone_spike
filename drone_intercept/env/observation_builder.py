"""Assembles the observation vector from world state.

Phase 1: 14D (self pos/vel + relative target pos/vel + distance + battery)
Phase 2: 15D (adds track_confidence)
Phase 3: 15D + N_sectors (adds obstacle sector distances)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# --- Phase 1: 14D ---
OBS_DIM = 14
OBS_SELF_POS = slice(0, 3)
OBS_SELF_VEL = slice(3, 6)
OBS_REL_POS = slice(6, 9)
OBS_REL_VEL = slice(9, 12)
OBS_DISTANCE = 12
OBS_BATTERY = 13

# --- Phase 2: 15D (adds track confidence) ---
OBS_DIM_PHASE2 = 15
OBS_TRACK_CONFIDENCE = 14

# Conservative bounds for normalization / space definition
_POS_BOUND = 200.0
_VEL_BOUND = 30.0
_DIST_BOUND = 300.0


def _base_low_high() -> tuple[list[float], list[float]]:
    """Return base observation bounds shared across phases."""
    low = [
        -_POS_BOUND, -_POS_BOUND, 0.0,           # self pos
        -_VEL_BOUND, -_VEL_BOUND, -_VEL_BOUND,   # self vel
        -_DIST_BOUND, -_DIST_BOUND, -_DIST_BOUND, # rel pos
        -_VEL_BOUND * 2, -_VEL_BOUND * 2, -_VEL_BOUND * 2,  # rel vel
        0.0,  # distance
        0.0,  # battery
    ]
    high = [
        _POS_BOUND, _POS_BOUND, _POS_BOUND,
        _VEL_BOUND, _VEL_BOUND, _VEL_BOUND,
        _DIST_BOUND, _DIST_BOUND, _DIST_BOUND,
        _VEL_BOUND * 2, _VEL_BOUND * 2, _VEL_BOUND * 2,
        _DIST_BOUND,
        1.0,
    ]
    return low, high


def observation_space(
    phase: int = 1, n_obstacle_sectors: int = 0, perception_range: float = 20.0,
) -> gym.spaces.Box:
    low, high = _base_low_high()
    if phase >= 2:
        low.append(0.0)   # track_confidence
        high.append(1.0)
    if n_obstacle_sectors > 0:
        low.extend([0.0] * n_obstacle_sectors)
        high.extend([perception_range] * n_obstacle_sectors)
    return gym.spaces.Box(
        low=np.array(low, dtype=np.float32),
        high=np.array(high, dtype=np.float32),
        dtype=np.float32,
    )


def build_observation(
    drone_pos: np.ndarray,
    drone_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    battery: float,
    track_confidence: float | None = None,
    sector_distances: np.ndarray | None = None,
) -> np.ndarray:
    rel_pos = target_pos - drone_pos
    rel_vel = target_vel - drone_vel
    distance = float(np.linalg.norm(rel_pos))
    fields: list[float] = [
        *drone_pos,
        *drone_vel,
        *rel_pos,
        *rel_vel,
        distance,
        battery,
    ]
    if track_confidence is not None:
        fields.append(track_confidence)
    if sector_distances is not None:
        fields.extend(sector_distances.tolist())
    return np.array(fields, dtype=np.float32)
