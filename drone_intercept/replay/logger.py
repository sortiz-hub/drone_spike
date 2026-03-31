"""Per-step JSONL episode logger and episode summary writer."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


@dataclass
class StepRecord:
    t: int
    drone_pos: list[float]
    drone_vel: list[float]
    target_pos: list[float]
    target_vel: list[float]
    action: list[float]
    reward: float
    done: bool
    distance: float


@dataclass
class EpisodeSummary:
    episode_id: int
    total_reward: float
    success: bool
    reason: str
    steps: int
    capture_time: float | None
    min_distance: float


class EpisodeLogger:
    """Logs per-step data to JSONL and writes episode summaries."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._steps: list[StepRecord] = []
        self._episode_id = 0
        self._total_reward = 0.0
        self._min_distance = float("inf")

    def on_reset(self) -> None:
        self._steps = []
        self._total_reward = 0.0
        self._min_distance = float("inf")

    def on_step(
        self,
        info: dict[str, Any],
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        distance = float(info.get("distance", 0.0))
        self._min_distance = min(self._min_distance, distance)
        self._total_reward += reward

        record = StepRecord(
            t=info["step"],
            drone_pos=_to_serializable(info["drone_pos"]),
            drone_vel=_to_serializable(info["drone_vel"]),
            target_pos=_to_serializable(info["target_pos"]),
            target_vel=_to_serializable(info["target_vel"]),
            action=_to_serializable(action),
            reward=float(reward),
            done=terminated or truncated,
            distance=distance,
        )
        self._steps.append(record)

    def on_episode_end(self, info: dict[str, Any], dt: float = 0.1) -> EpisodeSummary:
        reason = info.get("reason", "")
        success = reason == "capture"
        capture_time = len(self._steps) * dt if success else None

        summary = EpisodeSummary(
            episode_id=self._episode_id,
            total_reward=self._total_reward,
            success=success,
            reason=reason,
            steps=len(self._steps),
            capture_time=capture_time,
            min_distance=self._min_distance,
        )

        # Write per-step JSONL
        ep_file = self.log_dir / f"episode_{self._episode_id:05d}.jsonl"
        with open(ep_file, "w") as f:
            for step in self._steps:
                f.write(json.dumps(asdict(step)) + "\n")

        # Append to summary CSV
        summary_file = self.log_dir / "episode_summaries.csv"
        write_header = not summary_file.exists()
        with open(summary_file, "a") as f:
            if write_header:
                f.write(
                    "episode_id,total_reward,success,reason,steps,capture_time,min_distance\n"
                )
            f.write(
                f"{summary.episode_id},{summary.total_reward:.2f},"
                f"{summary.success},{summary.reason},{summary.steps},"
                f"{summary.capture_time},{summary.min_distance:.3f}\n"
            )

        self._episode_id += 1
        return summary

    @staticmethod
    def load_episode(path: str | Path) -> list[StepRecord]:
        records = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                records.append(StepRecord(**data))
        return records
