"""Simple Kalman filter tracker for smoothing noisy target detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerConfig:
    """Kalman filter tuning parameters.

    Attributes:
        process_noise: Process noise scalar (higher = trust measurements more).
        measurement_noise_pos: Measurement noise for position.
        measurement_noise_vel: Measurement noise for velocity.
        confidence_decay: Per-step decay when no measurement is received.
    """

    process_noise: float = 0.1
    measurement_noise_pos: float = 1.0
    measurement_noise_vel: float = 0.5
    confidence_decay: float = 0.05


class KalmanTracker:
    """6-state (pos + vel) linear Kalman filter for target tracking.

    State vector: [x, y, z, vx, vy, vz]
    Constant-velocity motion model.
    """

    def __init__(self, cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or TrackerConfig()
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6, dtype=np.float64)
        # Covariance
        self.P = np.eye(6, dtype=np.float64) * 100.0
        # Track confidence (0 = no track, 1 = high confidence)
        self.confidence = 0.0
        self._initialized = False

    def reset(self) -> None:
        self.state = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.confidence = 0.0
        self._initialized = False

    def predict(self, dt: float) -> None:
        """Predict step: propagate state forward using constant-velocity model."""
        # State transition: x' = x + v*dt
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.state = F @ self.state

        # Process noise
        q = self.cfg.process_noise
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q * dt**2
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q

        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        measured_pos: np.ndarray | None,
        measured_vel: np.ndarray | None,
    ) -> None:
        """Update step: incorporate a noisy measurement (or decay confidence on miss)."""
        if measured_pos is None:
            # No detection — decay confidence
            self.confidence = max(0.0, self.confidence - self.cfg.confidence_decay)
            return

        if not self._initialized:
            # First measurement — initialize state directly
            self.state[:3] = measured_pos.astype(np.float64)
            if measured_vel is not None:
                self.state[3:] = measured_vel.astype(np.float64)
            self.P = np.eye(6, dtype=np.float64) * 10.0
            self.confidence = 0.5
            self._initialized = True
            return

        # Measurement vector and matrix
        z = np.zeros(6, dtype=np.float64)
        z[:3] = measured_pos
        H = np.eye(6, dtype=np.float64)

        r_pos = self.cfg.measurement_noise_pos
        r_vel = self.cfg.measurement_noise_vel
        R = np.diag([r_pos, r_pos, r_pos, r_vel, r_vel, r_vel]).astype(np.float64)

        if measured_vel is not None:
            z[3:] = measured_vel
        else:
            # Only update position — mask velocity rows
            H = np.zeros((3, 6), dtype=np.float64)
            H[0, 0] = H[1, 1] = H[2, 2] = 1.0
            z = z[:3]
            R = R[:3, :3]

        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        innovation = z - H @ self.state
        self.state = self.state + K @ innovation

        # Covariance update
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        # Confidence increases toward 1.0
        self.confidence = min(1.0, self.confidence + 0.1)

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].astype(np.float32)

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:].astype(np.float32)
