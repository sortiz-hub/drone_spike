"""Step 9: Full training run — train a strong PPO policy.

Trains against constant-velocity target first, then evaluates against
all target types to measure generalization.

Usage:
    python scripts/09_train_full.py
    python scripts/09_train_full.py --timesteps 1000000
    python scripts/09_train_full.py --target zigzag
"""

import argparse
from pathlib import Path

from drone_intercept.training.train_ppo import train
from drone_intercept.training.eval_policy import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training + cross-evaluation")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--target", type=str, default="constant_velocity",
                        choices=["constant_velocity", "waypoint", "zigzag"])
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--reward-mode", type=str, default="original",
                        choices=["original", "shaped"],
                        help="Reward function: original or shaped")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume training. No value = auto-detect from save-dir. Or pass a path.")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    # --- Train ---
    print(f"\n{'='*50}")
    print(f"Training PPO | {args.timesteps} steps | target={args.target}")
    print(f"{'='*50}\n")

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
        device=args.device,
        reward_mode=args.reward_mode,
        resume=args.resume if args.resume != "auto" else None,
    )

    if args.skip_eval:
        print("\nSkipping evaluation.")
        return

    model_path = f"{args.save_dir}/ppo_intercept_final.zip"
    if not Path(model_path).exists():
        print(f"\nModel not found at {model_path}, skipping eval.")
        return

    # --- Evaluate against all target types ---
    targets = ["constant_velocity", "waypoint", "zigzag"]
    results = {}

    for target in targets:
        print(f"\n{'='*50}")
        print(f"Evaluating vs {target} | {args.eval_episodes} episodes")
        print(f"{'='*50}\n")

        eval_dir = f"{args.log_dir}/eval_{target}"
        r = evaluate(
            model_path=model_path,
            n_episodes=args.eval_episodes,
            target_behavior=target,
            target_speed=args.target_speed,
            log_dir=eval_dir,
            plot=True,
            seed=0,
            device=args.device,
        )
        results[target] = r

    # --- Summary ---
    print(f"\n{'='*50}")
    print("CROSS-EVALUATION SUMMARY")
    print(f"Trained on: {args.target} | {args.timesteps} steps")
    print(f"{'='*50}")
    print(f"{'Target':<22} {'Success':>8} {'Avg Reward':>12} {'Avg Cap Time':>13}")
    print(f"{'-'*55}")
    for target, r in results.items():
        cap_t = f"{r['avg_capture_time']:.1f}s" if r["avg_capture_time"] else "N/A"
        print(f"{target:<22} {r['success_rate']:>7.0%} {r['avg_reward']:>11.1f} {cap_t:>13}")


if __name__ == "__main__":
    main()
