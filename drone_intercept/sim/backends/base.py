"""Abstract base class for drone physics backends."""

from __future__ import annotations

import abc

import numpy as np


class PhysicsBackend(abc.ABC):
    """ABC for drone physics backends.

    Each backend owns the drone state (position, velocity, yaw, battery)
    and defines how it evolves in response to control inputs.
    """

    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.yaw: float = 0.0
        self.battery: float = 1.0

    @abc.abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Reset drone to initial state."""

    @abc.abstractmethod
    def step(self, action: np.ndarray, dt: float) -> None:
        """Advance drone dynamics by one timestep.

        Args:
            action: Control input, already clipped to action_space bounds.
            dt: Simulation timestep in seconds.
        """

    @abc.abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (low, high) arrays defining the action space bounds."""
