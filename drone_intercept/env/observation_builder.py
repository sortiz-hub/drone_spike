"""Assembles the 14D observation vector from world state."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

# Observation indices for external consumers
OBS_DIM = 14
OBS_SELF_POS = slice(0, 3)
OBS_SELF_VEL = slice(3, 6)
OBS_REL_POS = slice(6, 9)
OBS_REL_VEL = slice(9, 12)
OBS_DISTANCE = 12
OBS_BATTERY = 13

# Conservative bounds for normalization / space definition
_POS_BOUND = 200.0
_VEL_BOUND = 30.0
_DIST_BOUND = 300.0


def observation_space() -> gym.spaces.Box:
    low = np.array(
        [
            -_POS_BOUND, -_POS_BOUND, 0.0,        # self pos
            -_VEL_BOUND, -_VEL_BOUND, -_VEL_BOUND, # self vel
            -_DIST_BOUND, -_DIST_BOUND, -_DIST_BOUND,  # rel pos
            -_VEL_BOUND * 2, -_VEL_BOUND * 2, -_VEL_BOUND * 2,  # rel vel
            0.0,   # distance
            0.0,   # battery
        ],
        dtype=np.float32,
    )
    high = np.array(
        [
            _POS_BOUND, _POS_BOUND, _POS_BOUND,
            _VEL_BOUND, _VEL_BOUND, _VEL_BOUND,
            _DIST_BOUND, _DIST_BOUND, _DIST_BOUND,
            _VEL_BOUND * 2, _VEL_BOUND * 2, _VEL_BOUND * 2,
            _DIST_BOUND,
            1.0,
        ],
        dtype=np.float32,
    )
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def build_observation(
    drone_pos: np.ndarray,
    drone_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    battery: float,
) -> np.ndarray:
    rel_pos = target_pos - drone_pos
    rel_vel = target_vel - drone_vel
    distance = float(np.linalg.norm(rel_pos))
    return np.array(
        [
            *drone_pos,
            *drone_vel,
            *rel_pos,
            *rel_vel,
            distance,
            battery,
        ],
        dtype=np.float32,
    )
