"""Target trajectory prediction for lead pursuit (Phase 4).

Provides predicted future target positions based on current tracked state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictorConfig:
    """Configuration for target prediction.

    Attributes:
        horizons: Prediction time horizons in seconds.
    """

    horizons: tuple[float, ...] = (0.5, 1.0)


class ConstantVelocityPredictor:
    """Predicts future target positions using constant-velocity assumption.

    Given current position and velocity, extrapolates linearly:
        predicted_pos(t) = pos + vel * t
    """

    def __init__(self, cfg: PredictorConfig | None = None) -> None:
        self.cfg = cfg or PredictorConfig()
        self._predictions: list[np.ndarray] = [
            np.zeros(3, dtype=np.float32) for _ in self.cfg.horizons
        ]

    def predict(
        self, position: np.ndarray, velocity: np.ndarray,
    ) -> list[np.ndarray]:
        """Compute predicted positions for each configured horizon."""
        self._predictions = []
        for t in self.cfg.horizons:
            pred = (position + velocity * t).astype(np.float32)
            self._predictions.append(pred)
        return self._predictions

    @property
    def predictions(self) -> list[np.ndarray]:
        return self._predictions

    @property
    def n_predictions(self) -> int:
        return len(self.cfg.horizons)
