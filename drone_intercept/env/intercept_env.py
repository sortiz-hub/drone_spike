"""Gymnasium environment for drone interception.

Phase 1 — simplified dynamics with truth sensing.
Phase 2 — adds noisy detections + Kalman tracker.
Phase 3 — adds obstacles with sector-distance perception.
Phase 4 — adds target prediction for lead pursuit.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from drone_intercept.env import observation_builder
from drone_intercept.env.rewards import RewardConfig, compute_reward
from drone_intercept.env.termination import TerminationConfig, check_termination
from drone_intercept.sim.backends import get_backend_registry
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


class InterceptEnv(gym.Env):
    """Drone interception environment with simplified double-integrator dynamics.

    The drone tracks commanded velocity via first-order lag (mimicking an
    autopilot velocity controller). No PX4/Gazebo dependency — designed so
    the dynamics layer can be swapped for ROS 2 + PX4 SITL later.

    Args:
        sensing_mode: "truth" for Phase 1 or "tracked" for Phase 2.
        obstacle_config: If provided, enables Phase 3 obstacles.
        predictor_config: If provided, enables Phase 4 target prediction.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        target_behavior: str = "constant_velocity",
        target_speed: float = 5.0,
        dt: float = 0.1,
        physics_backend: str = "simplified",
        termination: TerminationConfig | None = None,
        render_mode: str | None = None,
        sensing_mode: str = "truth",
        noise_config: Any | None = None,
        tracker_config: Any | None = None,
        obstacle_config: Any | None = None,
        predictor_config: Any | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.render_mode = render_mode
        self.term_cfg = termination or TerminationConfig()
        self.sensing_mode = sensing_mode
        self._reward_cfg = reward_config or RewardConfig()

        # Physics backend
        backend_registry = get_backend_registry()
        if physics_backend not in backend_registry:
            raise ValueError(
                f"Unknown physics_backend '{physics_backend}'. "
                f"Choose from: {list(backend_registry)}"
            )
        self._backend = backend_registry[physics_backend]()
        self._is_gazebo = physics_backend == "px4_gazebo"

        # Gazebo target visual (only when using PX4 backend)
        self._target_visual = None
        if self._is_gazebo:
            try:
                from drone_intercept.sim.gz_target_visual import GzTargetVisual
                self._target_visual = GzTargetVisual()
            except ImportError:
                pass

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

        # Phase 3: obstacles
        self._obstacle_cfg = None
        self._obstacles: list = []
        n_obstacle_sectors = 0
        perception_range = 20.0
        if obstacle_config is not None:
            from drone_intercept.sim.obstacles import ObstacleConfig

            self._obstacle_cfg = obstacle_config if not isinstance(obstacle_config, bool) else ObstacleConfig()
            n_obstacle_sectors = self._obstacle_cfg.n_sectors
            perception_range = self._obstacle_cfg.perception_range

        # Phase 4: predictor
        self._predictor = None
        n_predictions = 0
        if predictor_config is not None:
            from drone_intercept.sim.predictor import ConstantVelocityPredictor, PredictorConfig

            self._predictor = ConstantVelocityPredictor(
                predictor_config if not isinstance(predictor_config, bool) else PredictorConfig()
            )
            n_predictions = self._predictor.n_predictions

        # Spaces
        phase = 2 if sensing_mode == "tracked" else 1
        self.observation_space = observation_builder.observation_space(
            phase=phase,
            n_obstacle_sectors=n_obstacle_sectors,
            perception_range=perception_range,
            n_predictions=n_predictions,
        )
        act_low, act_high = self._backend.action_bounds()
        self.action_space = gym.spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # State (drone state lives in self._backend)
        self._step_count = 0
        self._prev_distance: float | None = None
        self._rng = np.random.default_rng()

    @property
    def _drone_pos(self) -> np.ndarray:
        return self._backend.position

    @property
    def _drone_vel(self) -> np.ndarray:
        return self._backend.velocity

    @property
    def _yaw(self) -> float:
        return self._backend.yaw

    @property
    def _battery(self) -> float:
        return self._backend.battery

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed)

        # Drone state
        self._backend.reset(self._rng)
        self._step_count = 0
        self._prev_distance = None

        # Reset target
        self._target.reset(self._rng)

        # Spawn/move target visual in Gazebo
        if self._target_visual is not None:
            tp = self._target.position
            self._target_visual.spawn(float(tp[0]), float(tp[1]), float(tp[2]))

        # Reset tracker
        if self._tracker is not None:
            self._tracker.reset()

        # Generate obstacles (Phase 3)
        if self._obstacle_cfg is not None:
            from drone_intercept.sim.obstacles import generate_obstacles

            self._obstacles = generate_obstacles(self._rng, self._obstacle_cfg)
        else:
            self._obstacles = []

        obs = self._build_obs()
        info = self._build_info(reason="")
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Advance drone dynamics
        self._backend.step(action, self.dt)

        # Advance target
        self._target.step(self.dt)

        # Update target visual in Gazebo
        if self._target_visual is not None:
            self._target_visual.update_from_array(self._target.position)

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

        # Obstacle collision check (Phase 3)
        obstacle_crashed = False
        min_obstacle_dist = None
        if self._obstacles:
            from drone_intercept.sim.obstacles import (
                check_obstacle_collision,
                compute_sector_distances,
            )

            obstacle_crashed = check_obstacle_collision(self._drone_pos, self._obstacles)
            sector_dists = compute_sector_distances(
                self._drone_pos, self._obstacles, self._obstacle_cfg,
            )
            min_obstacle_dist = float(np.min(sector_dists))

        # Termination (always uses true state for ground truth)
        term = check_termination(
            self._drone_pos,
            self._drone_vel,
            self._target.position,
            self._target.velocity,
            self._step_count,
            self.term_cfg,
        )
        if obstacle_crashed and not term.terminated:
            term.terminated = True
            term.reason = "crash_obstacle"

        captured = term.reason == "capture"
        crashed = term.reason.startswith("crash")

        # Reward
        distance = float(np.linalg.norm(self._target.position - self._drone_pos))
        reward = compute_reward(
            distance=distance,
            action=action,
            captured=captured,
            crashed=crashed,
            altitude=float(self._drone_pos[2]),
            prev_distance=self._prev_distance,
            min_obstacle_distance=min_obstacle_dist,
            obstacle_crashed=obstacle_crashed,
            config=self._reward_cfg,
        )
        self._prev_distance = distance

        obs = self._build_obs()
        info = self._build_info(term.reason)
        info["distance"] = distance
        info["captured"] = captured

        return obs, reward, term.terminated, term.truncated, info

    def close(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()
        super().close()

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

    def _get_sector_distances(self) -> np.ndarray | None:
        """Return obstacle sector distances if obstacles are active."""
        if not self._obstacles or self._obstacle_cfg is None:
            return None
        from drone_intercept.sim.obstacles import compute_sector_distances

        return compute_sector_distances(
            self._drone_pos, self._obstacles, self._obstacle_cfg,
        )

    def _get_predictions(self) -> list[np.ndarray] | None:
        """Return predicted target positions if predictor is active."""
        if self._predictor is None:
            return None
        target_pos, target_vel, _ = self._get_sensed_target()
        return self._predictor.predict(target_pos, target_vel)

    def _build_obs(self) -> np.ndarray:
        target_pos, target_vel, confidence = self._get_sensed_target()
        return observation_builder.build_observation(
            drone_pos=self._drone_pos,
            drone_vel=self._drone_vel,
            target_pos=target_pos,
            target_vel=target_vel,
            battery=self._battery,
            track_confidence=confidence,
            sector_distances=self._get_sector_distances(),
            predicted_positions=self._get_predictions(),
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
        if self._obstacles:
            sector_dists = self._get_sector_distances()
            if sector_dists is not None:
                info["sector_distances"] = sector_dists.tolist()
                info["min_obstacle_distance"] = float(np.min(sector_dists))
        if self._predictor is not None:
            preds = self._get_predictions()
            if preds is not None:
                info["predicted_positions"] = [p.tolist() for p in preds]
        return info
