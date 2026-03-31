"""Base class for scripted target behaviors."""

from __future__ import annotations

import abc

import numpy as np


class TargetBehavior(abc.ABC):
    """A scripted target that moves according to some pattern."""

    def __init__(self, speed: float = 5.0) -> None:
        self.speed = speed
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

    @abc.abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Reset target to a new initial state."""

    @abc.abstractmethod
    def step(self, dt: float) -> None:
        """Advance target by one timestep."""
