"""Gaussian noise injection for simulating noisy target detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseConfig:
    """Configuration for detection noise injection.

    Attributes:
        pos_std: Standard deviation of position noise (m).
        vel_std: Standard deviation of velocity noise (m/s).
        detection_prob: Probability of receiving a detection each step (0-1).
    """

    pos_std: float = 1.0
    vel_std: float = 0.5
    detection_prob: float = 0.95


def inject_noise(
    true_pos: np.ndarray,
    true_vel: np.ndarray,
    rng: np.random.Generator,
    cfg: NoiseConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Add Gaussian noise to a ground-truth detection.

    Returns (noisy_pos, noisy_vel) or (None, None) if detection is missed.
    """
    if rng.random() > cfg.detection_prob:
        return None, None

    noisy_pos = true_pos + rng.normal(0.0, cfg.pos_std, size=3).astype(np.float32)
    noisy_vel = true_vel + rng.normal(0.0, cfg.vel_std, size=3).astype(np.float32)
    return noisy_pos, noisy_vel
