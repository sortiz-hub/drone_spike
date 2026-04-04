"""2D top-down trajectory plotter and video exporter for episode replay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
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
    drone_z = [s.drone_pos[2] for s in steps]
    target_x = [s.target_pos[0] for s in steps]
    target_y = [s.target_pos[1] for s in steps]
    target_z = [s.target_pos[2] for s in steps]
    distances = [s.distance for s in steps]
    rewards = [s.reward for s in steps]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

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

    # --- Altitude over time ---
    ax = axes[2]
    ax.plot(timesteps, drone_z, "b-", linewidth=1.5, label="Drone")
    ax.plot(timesteps, target_z, "r--", linewidth=1.5, label="Target")
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Reward over time (clip y-axis to avoid capture bonus crushing scale) ---
    ax = axes[3]
    ax.plot(timesteps, rewards, "m-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Time")
    r_arr = np.array(rewards)
    r_lo, r_hi = np.percentile(r_arr, [1, 99])
    r_pad = max(0.1, 0.1 * (r_hi - r_lo))
    ax.set_ylim(r_lo - r_pad, r_hi + r_pad)
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


def animate_episode(
    steps: list[StepRecord],
    save_path: str | Path = "episode.mp4",
    fps: int = 10,
    title: str = "Episode Replay",
) -> None:
    """Animate a 2D top-down episode replay and save as video.

    Uses matplotlib.animation — no Gazebo or simulator required.
    Supports .mp4 (ffmpeg) and .gif (pillow).
    """
    drone_x = [s.drone_pos[0] for s in steps]
    drone_y = [s.drone_pos[1] for s in steps]
    drone_z = [s.drone_pos[2] for s in steps]
    target_x = [s.target_pos[0] for s in steps]
    target_y = [s.target_pos[1] for s in steps]
    target_z = [s.target_pos[2] for s in steps]
    distances = [s.distance for s in steps]
    rewards = [s.reward for s in steps]

    fig, (ax_traj, ax_dist, ax_alt, ax_rew) = plt.subplots(1, 4, figsize=(28, 6))

    # Compute axis limits with padding
    all_x = drone_x + target_x
    all_y = drone_y + target_y
    pad = max(5.0, 0.1 * (max(all_x) - min(all_x)), 0.1 * (max(all_y) - min(all_y)))
    ax_traj.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax_traj.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_title("Top-Down Trajectory")
    ax_traj.grid(True, alpha=0.3)

    ax_dist.set_xlim(0, len(steps))
    ax_dist.set_ylim(0, max(distances) * 1.1)
    ax_dist.axhline(y=1.5, color="g", linestyle="--", alpha=0.5, label="Capture dist")
    ax_dist.set_xlabel("Step")
    ax_dist.set_ylabel("Distance (m)")
    ax_dist.set_title("Distance Over Time")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # --- Altitude panel ---
    all_z = drone_z + target_z
    ax_alt.set_xlim(0, len(steps))
    z_pad = max(1.0, 0.1 * (max(all_z) - min(all_z)))
    ax_alt.set_ylim(min(all_z) - z_pad, max(all_z) + z_pad)
    ax_alt.set_xlabel("Step")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude Over Time")
    ax_alt.grid(True, alpha=0.3)

    # --- Reward panel (clip y-axis to avoid capture bonus crushing scale) ---
    ax_rew.set_xlim(0, len(steps))
    r_arr = np.array(rewards)
    r_lo, r_hi = float(np.percentile(r_arr, 1)), float(np.percentile(r_arr, 99))
    rew_pad = max(0.1, 0.1 * (r_hi - r_lo))
    ax_rew.set_ylim(r_lo - rew_pad, r_hi + rew_pad)
    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Reward")
    ax_rew.set_title("Reward Over Time")
    ax_rew.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Animated artists
    (drone_trail,) = ax_traj.plot([], [], "b-", linewidth=1.5, label="Drone")
    (target_trail,) = ax_traj.plot([], [], "r--", linewidth=1.5, label="Target")
    (drone_dot,) = ax_traj.plot([], [], "bo", markersize=8)
    (target_dot,) = ax_traj.plot([], [], "r^", markersize=8)
    (dist_line,) = ax_dist.plot([], [], "k-", linewidth=1)
    (drone_alt,) = ax_alt.plot([], [], "b-", linewidth=1.5, label="Drone")
    (target_alt,) = ax_alt.plot([], [], "r--", linewidth=1.5, label="Target")
    ax_alt.legend(fontsize=8)
    (rew_line,) = ax_rew.plot([], [], "m-", linewidth=1)
    ax_traj.legend(fontsize=8)

    def update(frame: int):  # noqa: ANN202
        i = frame + 1
        drone_trail.set_data(drone_x[:i], drone_y[:i])
        target_trail.set_data(target_x[:i], target_y[:i])
        drone_dot.set_data([drone_x[frame]], [drone_y[frame]])
        target_dot.set_data([target_x[frame]], [target_y[frame]])
        dist_line.set_data(np.arange(i), distances[:i])
        drone_alt.set_data(np.arange(i), drone_z[:i])
        target_alt.set_data(np.arange(i), target_z[:i])
        rew_line.set_data(np.arange(i), rewards[:i])
        return drone_trail, target_trail, drone_dot, target_dot, dist_line, drone_alt, target_alt, rew_line

    anim = animation.FuncAnimation(
        fig, update, frames=len(steps), interval=1000 // fps, blit=True,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(save_path), writer="pillow", fps=fps)
    elif animation.writers.is_available("ffmpeg"):
        anim.save(str(save_path), writer="ffmpeg", fps=fps)
    else:
        # Fallback: save as GIF when ffmpeg unavailable
        fallback = save_path.with_suffix(".gif")
        anim.save(str(fallback), writer="pillow", fps=fps)
        print(f"ffmpeg not found — saved as {fallback} instead")
    plt.close(fig)


def animate_episode_from_file(
    jsonl_path: str | Path,
    save_path: str | Path = "episode.mp4",
    fps: int = 10,
) -> None:
    """Load a JSONL episode file and export as video."""
    steps = EpisodeLogger.load_episode(jsonl_path)
    name = Path(jsonl_path).stem
    animate_episode(steps, save_path=save_path, fps=fps, title=name)


def plot_episode_from_file(
    jsonl_path: str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Load a JSONL episode file and plot it."""
    steps = EpisodeLogger.load_episode(jsonl_path)
    name = Path(jsonl_path).stem
    plot_episode(steps, title=name, save_path=save_path, show=show)
