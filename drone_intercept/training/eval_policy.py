"""Evaluate a trained policy and produce episode logs + plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from drone_intercept.env.intercept_env import InterceptEnv
from drone_intercept.env.termination import TerminationConfig
from drone_intercept.replay.logger import EpisodeLogger
from drone_intercept.replay.plotter import plot_episode


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    target_behavior: str = "constant_velocity",
    target_speed: float = 5.0,
    log_dir: str = "logs/eval",
    plot: bool = True,
    seed: int = 0,
    sensing_mode: str = "truth",
) -> dict:
    """Run evaluation episodes and return aggregate metrics."""
    cfg = TerminationConfig()
    env = InterceptEnv(
        target_behavior=target_behavior,
        target_speed=target_speed,
        termination=cfg,
        sensing_mode=sensing_mode,
    )
    model = PPO.load(model_path)
    logger = EpisodeLogger(log_dir=log_dir)

    successes = 0
    total_rewards = []
    capture_times = []
    min_distances = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        logger.on_reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            logger.on_step(info, action, reward, terminated, truncated)

        summary = logger.on_episode_end(info, dt=env.dt)
        total_rewards.append(summary.total_reward)
        min_distances.append(summary.min_distance)
        if summary.success:
            successes += 1
            capture_times.append(summary.capture_time)

        # Plot first few episodes
        if plot and ep < 5:
            steps = EpisodeLogger.load_episode(
                Path(log_dir) / f"episode_{ep:05d}.jsonl"
            )
            plot_episode(
                steps,
                title=f"Eval Episode {ep} ({'capture' if summary.success else summary.reason})",
                save_path=Path(log_dir) / f"plot_episode_{ep:05d}.png",
                show=False,
            )

    env.close()

    results = {
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_min_distance": np.mean(min_distances),
        "avg_capture_time": np.mean(capture_times) if capture_times else None,
    }

    print("\n=== Evaluation Results ===")
    print(f"Episodes:       {results['n_episodes']}")
    print(f"Success rate:   {results['success_rate']:.1%}")
    print(f"Avg reward:     {results['avg_reward']:.1f} +/- {results['std_reward']:.1f}")
    print(f"Avg min dist:   {results['avg_min_distance']:.2f} m")
    if results["avg_capture_time"]:
        print(f"Avg capture t:  {results['avg_capture_time']:.1f} s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained interception policy")
    parser.add_argument("model_path", type=str, help="Path to saved PPO model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--target", type=str, default="constant_velocity",
        choices=["constant_velocity", "waypoint", "zigzag"],
    )
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--log-dir", type=str, default="logs/eval")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sensing-mode", type=str, default="truth",
        choices=["truth", "tracked"],
        help="Sensing mode: truth (Phase 1) or tracked (Phase 2)",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        target_behavior=args.target,
        target_speed=args.target_speed,
        log_dir=args.log_dir,
        plot=not args.no_plot,
        seed=args.seed,
        sensing_mode=args.sensing_mode,
    )


if __name__ == "__main__":
    main()
