"""Abstract base class for dynamics backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DynamicsState:
    """Snapshot of drone state returned by a dynamics backend."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    yaw: float = 0.0
    battery: float = 1.0


class DynamicsBackend(abc.ABC):
    """Interface that every dynamics implementation must satisfy.

    The environment calls these methods without knowing whether the
    physics are simulated locally or running in Gazebo + PX4 SITL.
    """

    @abc.abstractmethod
    def reset(self, rng: np.random.Generator) -> DynamicsState:
        """Reset the drone to an initial hovering state and return it."""

    @abc.abstractmethod
    def step(
        self,
        vel_cmd: np.ndarray,
        yaw_rate_cmd: float,
        dt: float,
    ) -> DynamicsState:
        """Apply a velocity / yaw-rate command for one timestep.

        Args:
            vel_cmd: Desired velocity [vx, vy, vz] in m/s.
            yaw_rate_cmd: Desired yaw rate in rad/s.
            dt: Timestep duration in seconds.

        Returns:
            Updated DynamicsState after the step.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release any held resources (ROS nodes, Gazebo connection, etc.)."""
