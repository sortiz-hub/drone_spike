"""PPO training entry point for drone interception."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from drone_intercept.env.intercept_env import InterceptEnv
from drone_intercept.training.callbacks import InterceptCallback


def make_env(
    target_behavior: str = "constant_velocity",
    target_speed: float = 5.0,
    dt: float = 0.1,
    max_steps: int = 1000,
    sensing_mode: str = "truth",
    obstacles: bool = False,
    prediction: bool = False,
    dynamics: str = "simplified",
) -> InterceptEnv:
    from drone_intercept.env.termination import TerminationConfig

    cfg = TerminationConfig(max_steps=max_steps)
    obstacle_config = None
    if obstacles:
        from drone_intercept.sim.obstacles import ObstacleConfig
        obstacle_config = ObstacleConfig()
    predictor_config = None
    if prediction:
        from drone_intercept.sim.predictor import PredictorConfig
        predictor_config = PredictorConfig()
    return Monitor(
        InterceptEnv(
            target_behavior=target_behavior,
            target_speed=target_speed,
            dt=dt,
            termination=cfg,
            sensing_mode=sensing_mode,
            obstacle_config=obstacle_config,
            predictor_config=predictor_config,
            dynamics=dynamics,
        )
    )


def train(
    total_timesteps: int = 500_000,
    target_behavior: str = "constant_velocity",
    target_speed: float = 5.0,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    sensing_mode: str = "truth",
    obstacles: bool = False,
    prediction: bool = False,
    dynamics: str = "simplified",
) -> PPO:
    """Train a PPO policy on the interception environment."""
    env = make_vec_env(
        lambda: make_env(
            target_behavior=target_behavior,
            target_speed=target_speed,
            sensing_mode=sensing_mode,
            obstacles=obstacles,
            prediction=prediction,
            dynamics=dynamics,
        ),
        n_envs=n_envs,
        seed=seed,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        n_steps=2048,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        seed=seed,
    )

    callback = InterceptCallback(
        save_dir=save_dir,
        log_dir=log_dir,
        save_freq=50_000,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for drone interception")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--target", type=str, default="constant_velocity",
        choices=["constant_velocity", "waypoint", "zigzag"],
    )
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument(
        "--sensing-mode", type=str, default="truth",
        choices=["truth", "tracked"],
        help="Sensing mode: truth (Phase 1) or tracked (Phase 2)",
    )
    parser.add_argument(
        "--obstacles", action="store_true",
        help="Enable obstacles in the environment (Phase 3)",
    )
    parser.add_argument(
        "--prediction", action="store_true",
        help="Enable target prediction in observation (Phase 4)",
    )
    parser.add_argument(
        "--dynamics", type=str, default="simplified",
        choices=["simplified", "gazebo"],
        help="Dynamics backend: simplified (no deps) or gazebo (ROS 2 + PX4)",
    )
    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        target_behavior=args.target,
        target_speed=args.target_speed,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        sensing_mode=args.sensing_mode,
        obstacles=args.obstacles,
        prediction=args.prediction,
        dynamics=args.dynamics,
    )


if __name__ == "__main__":
    main()
