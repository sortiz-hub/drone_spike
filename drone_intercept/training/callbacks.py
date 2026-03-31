"""SB3 callbacks for logging, plotting, and checkpointing."""

from __future__ import annotations

import os
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class InterceptCallback(BaseCallback):
    """Logs episode metrics and saves periodic checkpoints + trajectory plots."""

    def __init__(
        self,
        save_dir: str | Path = "models",
        log_dir: str | Path = "logs",
        save_freq: int = 50_000,
        plot_freq: int = 100,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq
        self.plot_freq = plot_freq

        self._episode_count = 0
        self._episode_rewards: list[float] = []
        self._episode_successes: list[bool] = []

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Check for episode ends in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                self._episode_rewards.append(ep_info["r"])
                self._episode_count += 1
                captured = info.get("captured", False)
                self._episode_successes.append(captured)

                if self.verbose >= 1 and self._episode_count % 50 == 0:
                    recent_rewards = self._episode_rewards[-50:]
                    recent_success = self._episode_successes[-50:]
                    avg_r = sum(recent_rewards) / len(recent_rewards)
                    success_rate = sum(recent_success) / len(recent_success)
                    print(
                        f"[Episode {self._episode_count}] "
                        f"avg_reward={avg_r:.1f}  "
                        f"success_rate={success_rate:.1%}"
                    )

        # Periodic checkpoint
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_dir / f"ppo_intercept_{self.num_timesteps}"
            self.model.save(str(path))
            if self.verbose >= 1:
                print(f"[Checkpoint] Saved to {path}")

        return True

    def _on_training_end(self) -> None:
        path = self.save_dir / "ppo_intercept_final"
        self.model.save(str(path))
        if self.verbose >= 1:
            print(f"[Final] Saved to {path}")
