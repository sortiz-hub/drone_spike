"""Step 11: Batch episode viewer — browse, filter, and plot logged episodes.

Scans a log directory for JSONL episodes + summary CSV, then lets you
browse them with filters and generate plots.

Usage:
    python scripts/11_batch_viewer.py                           # auto-detect latest eval dir
    python scripts/11_batch_viewer.py logs/eval_zigzag          # specific dir
    python scripts/11_batch_viewer.py logs/eval_zigzag --filter success
    python scripts/11_batch_viewer.py logs/eval_zigzag --filter fail
    python scripts/11_batch_viewer.py logs/eval_zigzag --sort reward
    python scripts/11_batch_viewer.py logs/eval_zigzag --top 5 --plot
    python scripts/11_batch_viewer.py logs/eval_zigzag --plot-all --out plots/
    python scripts/11_batch_viewer.py logs/eval_zigzag --compare 0 3 7
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from drone_intercept.replay.logger import EpisodeLogger, StepRecord
from drone_intercept.replay.plotter import plot_episode


@dataclass
class EpisodeInfo:
    episode_id: int
    total_reward: float
    success: bool
    reason: str
    steps: int
    capture_time: float | None
    min_distance: float
    jsonl_path: Path


def load_summaries(log_dir: Path) -> list[EpisodeInfo]:
    """Load episode summaries from CSV + match to JSONL files."""
    csv_path = log_dir / "episode_summaries.csv"
    if not csv_path.exists():
        print(f"No episode_summaries.csv found in {log_dir}")
        sys.exit(1)

    episodes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep_id = int(row["episode_id"])
            jsonl = log_dir / f"episode_{ep_id:05d}.jsonl"
            if not jsonl.exists():
                continue
            episodes.append(EpisodeInfo(
                episode_id=ep_id,
                total_reward=float(row["total_reward"]),
                success=row["success"] == "True",
                reason=row["reason"],
                steps=int(row["steps"]),
                capture_time=float(row["capture_time"]) if row["capture_time"] != "None" else None,
                min_distance=float(row["min_distance"]),
                jsonl_path=jsonl,
            ))
    return episodes


def filter_episodes(episodes: list[EpisodeInfo], mode: str) -> list[EpisodeInfo]:
    if mode == "success":
        return [e for e in episodes if e.success]
    elif mode == "fail":
        return [e for e in episodes if not e.success]
    return episodes


def sort_episodes(episodes: list[EpisodeInfo], key: str, reverse: bool = True) -> list[EpisodeInfo]:
    sort_keys = {
        "reward": lambda e: e.total_reward,
        "distance": lambda e: e.min_distance,
        "steps": lambda e: e.steps,
        "capture_time": lambda e: e.capture_time or float("inf"),
        "id": lambda e: e.episode_id,
    }
    return sorted(episodes, key=sort_keys.get(key, sort_keys["id"]), reverse=reverse)


def print_table(episodes: list[EpisodeInfo]) -> None:
    header = f"{'ID':>4}  {'Result':<16} {'Reward':>8} {'Steps':>6} {'Min Dist':>9} {'Cap Time':>9}"
    print(header)
    print("-" * len(header))
    for e in episodes:
        result = "CAPTURE" if e.success else e.reason
        cap_t = f"{e.capture_time:.1f}s" if e.capture_time else "-"
        print(f"{e.episode_id:>4}  {result:<16} {e.total_reward:>8.1f} {e.steps:>6} {e.min_distance:>8.2f}m {cap_t:>9}")


def print_stats(episodes: list[EpisodeInfo]) -> None:
    total = len(episodes)
    successes = sum(1 for e in episodes if e.success)
    rewards = [e.total_reward for e in episodes]
    min_dists = [e.min_distance for e in episodes]
    cap_times = [e.capture_time for e in episodes if e.capture_time]

    print(f"\n--- Stats ({total} episodes) ---")
    print(f"Success rate:    {successes}/{total} ({successes/total:.0%})")
    print(f"Reward:          {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}  (min={min(rewards):.1f}, max={max(rewards):.1f})")
    print(f"Min distance:    {np.mean(min_dists):.2f}m avg  (best={min(min_dists):.2f}m)")
    if cap_times:
        print(f"Capture time:    {np.mean(cap_times):.1f}s avg  (fastest={min(cap_times):.1f}s, slowest={max(cap_times):.1f}s)")

    # Failure breakdown
    failures = [e for e in episodes if not e.success]
    if failures:
        reasons = {}
        for e in failures:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        print(f"Failures:        {', '.join(f'{r}={c}' for r, c in sorted(reasons.items()))}")


def plot_compare(episodes: list[EpisodeInfo], ids: list[int], out: Path | None = None) -> None:
    """Overlay trajectories of selected episodes on a single plot."""
    selected = [e for e in episodes if e.episode_id in ids]
    if not selected:
        print(f"No episodes found with IDs: {ids}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

    for ep, color in zip(selected, colors):
        steps = EpisodeLogger.load_episode(ep.jsonl_path)
        dx = [s.drone_pos[0] for s in steps]
        dy = [s.drone_pos[1] for s in steps]
        tx = [s.target_pos[0] for s in steps]
        ty = [s.target_pos[1] for s in steps]
        dists = [s.distance for s in steps]
        label = f"Ep {ep.episode_id} ({'cap' if ep.success else ep.reason})"

        axes[0].plot(dx, dy, "-", color=color, linewidth=1.5, label=f"D {label}")
        axes[0].plot(tx, ty, "--", color=color, linewidth=1, alpha=0.5)
        axes[0].plot(dx[0], dy[0], "s", color=color, markersize=6)
        if steps[-1].done and steps[-1].distance < 1.5:
            axes[0].plot(dx[-1], dy[-1], "*", color=color, markersize=12)

        axes[1].plot(dists, "-", color=color, linewidth=1, label=label)

    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title("Trajectories")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(y=1.5, color="g", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distance (m)")
    axes[1].set_title("Distance Over Time")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Episode Comparison: {[e.episode_id for e in selected]}", fontweight="bold")
    plt.tight_layout()

    if out:
        out.mkdir(parents=True, exist_ok=True)
        save = out / f"compare_{'_'.join(str(i) for i in ids)}.png"
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    plt.close(fig)


def plot_distribution(episodes: list[EpisodeInfo], out: Path | None = None) -> None:
    """Plot reward and capture time distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    rewards = [e.total_reward for e in episodes]
    axes[0].hist(rewards, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Total Reward")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Distribution")
    axes[0].grid(True, alpha=0.3)

    min_dists = [e.min_distance for e in episodes]
    axes[1].hist(min_dists, bins=20, edgecolor="black", alpha=0.7, color="orange")
    axes[1].axvline(x=1.5, color="g", linestyle="--", label="Capture dist")
    axes[1].set_xlabel("Min Distance (m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Closest Approach Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cap_times = [e.capture_time for e in episodes if e.capture_time]
    if cap_times:
        axes[2].hist(cap_times, bins=20, edgecolor="black", alpha=0.7, color="green")
        axes[2].set_xlabel("Capture Time (s)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Capture Time Distribution")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No captures", ha="center", va="center", fontsize=14)
        axes[2].set_title("Capture Time Distribution")

    n = len(episodes)
    s = sum(1 for e in episodes if e.success)
    fig.suptitle(f"Batch Stats — {s}/{n} captures ({s/n:.0%})", fontweight="bold")
    plt.tight_layout()

    if out:
        out.mkdir(parents=True, exist_ok=True)
        save = out / "distributions.png"
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved: {save}")
    plt.show()
    plt.close(fig)


def find_log_dir() -> Path:
    """Auto-detect the most recent eval log directory."""
    logs = Path("logs")
    if not logs.exists():
        print("No logs/ directory found. Run an evaluation first.")
        sys.exit(1)
    candidates = sorted(
        [d for d in logs.iterdir() if d.is_dir() and (d / "episode_summaries.csv").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print("No eval directories with episode_summaries.csv found in logs/")
        sys.exit(1)
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch episode viewer")
    parser.add_argument("log_dir", type=str, nargs="?", default=None,
                        help="Log directory (auto-detects if omitted)")
    parser.add_argument("--filter", type=str, default="all",
                        choices=["all", "success", "fail"])
    parser.add_argument("--sort", type=str, default="id",
                        choices=["id", "reward", "distance", "steps", "capture_time"])
    parser.add_argument("--reverse", action="store_true",
                        help="Reverse sort order")
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N episodes")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the listed episodes individually")
    parser.add_argument("--plot-all", action="store_true",
                        help="Plot every episode (saves to --out dir)")
    parser.add_argument("--compare", type=int, nargs="+", default=None,
                        help="Overlay trajectories of these episode IDs")
    parser.add_argument("--dist", action="store_true",
                        help="Show reward/distance/time distributions")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for saved plots")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else find_log_dir()
    print(f"Log directory: {log_dir}\n")

    all_episodes = load_summaries(log_dir)
    if not all_episodes:
        print("No episodes found.")
        sys.exit(1)

    # Stats on full set
    print_stats(all_episodes)
    print()

    # Filter and sort
    episodes = filter_episodes(all_episodes, args.filter)
    desc = args.sort != "distance"  # lower distance = better, so ascending
    if args.reverse:
        desc = not desc
    episodes = sort_episodes(episodes, args.sort, reverse=desc)

    if args.top:
        episodes = episodes[:args.top]

    # Table
    print_table(episodes)

    # Distributions
    if args.dist:
        out = Path(args.out) if args.out else None
        plot_distribution(all_episodes, out)

    # Compare overlay
    if args.compare:
        out = Path(args.out) if args.out else None
        plot_compare(all_episodes, args.compare, out)

    # Individual plots
    if args.plot or args.plot_all:
        to_plot = all_episodes if args.plot_all else episodes
        out = Path(args.out) if args.out else log_dir / "plots"
        out.mkdir(parents=True, exist_ok=True)
        for ep in to_plot:
            steps = EpisodeLogger.load_episode(ep.jsonl_path)
            result = "capture" if ep.success else ep.reason
            save = out / f"ep_{ep.episode_id:05d}_{result}.png"
            plot_episode(
                steps,
                title=f"Episode {ep.episode_id} ({result}) — reward={ep.total_reward:.1f}",
                save_path=save,
                show=False,
            )
        print(f"\nPlots saved to {out}/ ({len(to_plot)} episodes)")


if __name__ == "__main__":
    main()
