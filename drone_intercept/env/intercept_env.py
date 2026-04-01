"""Gymnasium environment for drone interception.

Phase 1 — simplified dynamics with truth sensing.
Phase 2 — adds noisy detections + Kalman tracker.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from drone_intercept.env import observation_builder, rewards
from drone_intercept.env.termination import TerminationConfig, check_termination
from drone_intercept.sim.target_behaviors import (
    ConstantVelocityTarget,
    TargetBehavior,
    WaypointTarget,
    ZigzagTarget,
)

_TARGET_REGISTRY: dict[str, type[TargetBehavior]] = {
    "constant_velocity": ConstantVelocityTarget,
    "waypoint": WaypointTarget,
    "zigzag": ZigzagTarget,
}

# Simplified drone dynamics constants
_MAX_VEL = 10.0        # m/s per axis
_MAX_YAW_RATE = 2.0    # rad/s
_VEL_TAU = 0.3         # velocity tracking time constant (s)
_GRAVITY = 9.81


class InterceptEnv(gym.Env):
    """Drone interception environment with simplified double-integrator dynamics.

    The drone tracks commanded velocity via first-order lag (mimicking an
    autopilot velocity controller). No PX4/Gazebo dependency — designed so
    the dynamics layer can be swapped for ROS 2 + PX4 SITL later.

    Args:
        sensing_mode: "truth" for Phase 1 (simulator truth) or
                      "tracked" for Phase 2 (noisy detections + Kalman filter).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        target_behavior: str = "constant_velocity",
        target_speed: float = 5.0,
        dt: float = 0.1,
        termination: TerminationConfig | None = None,
        render_mode: str | None = None,
        sensing_mode: str = "truth",
        noise_config: Any | None = None,
        tracker_config: Any | None = None,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.render_mode = render_mode
        self.term_cfg = termination or TerminationConfig()
        self.sensing_mode = sensing_mode

        # Target
        if target_behavior not in _TARGET_REGISTRY:
            raise ValueError(
                f"Unknown target_behavior '{target_behavior}'. "
                f"Choose from: {list(_TARGET_REGISTRY)}"
            )
        self._target = _TARGET_REGISTRY[target_behavior](speed=target_speed)

        # Phase 2: tracker + noise
        self._tracker = None
        self._noise_cfg = None
        if sensing_mode == "tracked":
            from drone_intercept.sim.noise import NoiseConfig
            from drone_intercept.sim.tracker import KalmanTracker, TrackerConfig

            self._noise_cfg = noise_config or NoiseConfig()
            self._tracker = KalmanTracker(tracker_config or TrackerConfig())

        # Spaces
        phase = 2 if sensing_mode == "tracked" else 1
        self.observation_space = observation_builder.observation_space(phase=phase)
        self.action_space = gym.spaces.Box(
            low=np.array([-_MAX_VEL, -_MAX_VEL, -_MAX_VEL, -_MAX_YAW_RATE], dtype=np.float32),
            high=np.array([_MAX_VEL, _MAX_VEL, _MAX_VEL, _MAX_YAW_RATE], dtype=np.float32),
            dtype=np.float32,
        )

        # State (set in reset)
        self._drone_pos = np.zeros(3, dtype=np.float32)
        self._drone_vel = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0
        self._battery = 1.0
        self._step_count = 0
        self._rng = np.random.default_rng()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed)

        # Drone starts near origin, hovering
        self._drone_pos = np.array(
            [
                self._rng.uniform(-2.0, 2.0),
                self._rng.uniform(-2.0, 2.0),
                self._rng.uniform(2.0, 4.0),
            ],
            dtype=np.float32,
        )
        self._drone_vel = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0
        self._battery = 1.0
        self._step_count = 0

        # Reset target
        self._target.reset(self._rng)

        # Reset tracker
        if self._tracker is not None:
            self._tracker.reset()

        obs = self._build_obs()
        info = self._build_info(reason="")
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        vel_cmd = action[:3]
        yaw_rate_cmd = action[3]

        # First-order velocity tracking (simplified autopilot)
        alpha = 1.0 - np.exp(-self.dt / _VEL_TAU)
        self._drone_vel = (
            self._drone_vel + alpha * (vel_cmd - self._drone_vel)
        ).astype(np.float32)

        # Integrate position
        self._drone_pos = (self._drone_pos + self._drone_vel * self.dt).astype(
            np.float32
        )

        # Yaw
        self._yaw += float(yaw_rate_cmd) * self.dt

        # Battery drain (simple linear)
        self._battery = max(0.0, self._battery - 0.0002)

        # Advance target
        self._target.step(self.dt)

        # Advance tracker (Phase 2)
        if self._tracker is not None:
            from drone_intercept.sim.noise import inject_noise

            self._tracker.predict(self.dt)
            noisy_pos, noisy_vel = inject_noise(
                self._target.position, self._target.velocity,
                self._rng, self._noise_cfg,
            )
            self._tracker.update(noisy_pos, noisy_vel)

        self._step_count += 1

        # Termination (always uses true state for ground truth)
        term = check_termination(
            self._drone_pos,
            self._drone_vel,
            self._target.position,
            self._target.velocity,
            self._step_count,
            self.term_cfg,
        )
        captured = term.reason == "capture"
        crashed = term.reason.startswith("crash")

        # Reward
        distance = float(np.linalg.norm(self._target.position - self._drone_pos))
        reward = rewards.compute_reward(
            distance=distance,
            action=action,
            captured=captured,
            crashed=crashed,
            altitude=float(self._drone_pos[2]),
        )

        obs = self._build_obs()
        info = self._build_info(term.reason)
        info["distance"] = distance
        info["captured"] = captured

        return obs, reward, term.terminated, term.truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sensed_target(self) -> tuple[np.ndarray, np.ndarray, float | None]:
        """Return target pos/vel and optional track_confidence based on sensing mode."""
        if self._tracker is not None:
            return (
                self._tracker.position,
                self._tracker.velocity,
                self._tracker.confidence,
            )
        return self._target.position, self._target.velocity, None

    def _build_obs(self) -> np.ndarray:
        target_pos, target_vel, confidence = self._get_sensed_target()
        return observation_builder.build_observation(
            drone_pos=self._drone_pos,
            drone_vel=self._drone_vel,
            target_pos=target_pos,
            target_vel=target_vel,
            battery=self._battery,
            track_confidence=confidence,
        )

    def _build_info(self, reason: str) -> dict[str, Any]:
        info = {
            "drone_pos": self._drone_pos.copy(),
            "drone_vel": self._drone_vel.copy(),
            "target_pos": self._target.position.copy(),
            "target_vel": self._target.velocity.copy(),
            "yaw": self._yaw,
            "battery": self._battery,
            "step": self._step_count,
            "reason": reason,
        }
        if self._tracker is not None:
            info["tracked_pos"] = self._tracker.position.copy()
            info["tracked_vel"] = self._tracker.velocity.copy()
            info["track_confidence"] = self._tracker.confidence
        return info
