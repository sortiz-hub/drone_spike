"""2D top-down trajectory plotter for episode replay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from drone_intercept.replay.logger import EpisodeLogger, StepRecord


def plot_episode(
    steps: list[StepRecord],
    title: str = "Episode Trajectory",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot 2D top-down view of drone and target trajectories."""
    drone_x = [s.drone_pos[0] for s in steps]
    drone_y = [s.drone_pos[1] for s in steps]
    target_x = [s.target_pos[0] for s in steps]
    target_y = [s.target_pos[1] for s in steps]
    distances = [s.distance for s in steps]
    rewards = [s.reward for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Trajectory plot ---
    ax = axes[0]
    ax.plot(drone_x, drone_y, "b-", linewidth=1.5, label="Drone")
    ax.plot(target_x, target_y, "r--", linewidth=1.5, label="Target")
    ax.plot(drone_x[0], drone_y[0], "bs", markersize=10, label="Drone start")
    ax.plot(target_x[0], target_y[0], "r^", markersize=10, label="Target start")

    # Mark endpoint
    last = steps[-1]
    if last.done and last.distance < 1.5:
        ax.plot(drone_x[-1], drone_y[-1], "g*", markersize=18, label="Capture")
    elif last.done:
        ax.plot(drone_x[-1], drone_y[-1], "kx", markersize=14, label="Terminal")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-Down Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- Distance over time ---
    ax = axes[1]
    timesteps = np.arange(len(distances))
    ax.plot(timesteps, distances, "k-", linewidth=1)
    ax.axhline(y=1.5, color="g", linestyle="--", alpha=0.5, label="Capture dist")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to target (m)")
    ax.set_title("Distance Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Reward over time ---
    ax = axes[2]
    ax.plot(timesteps, rewards, "m-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Time")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_episode_from_file(
    jsonl_path: str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Load a JSONL episode file and plot it."""
    steps = EpisodeLogger.load_episode(jsonl_path)
    name = Path(jsonl_path).stem
    plot_episode(steps, title=name, save_path=save_path, show=show)
