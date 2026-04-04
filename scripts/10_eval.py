"""Step 10: Evaluate a trained model against any target type.

Auto-detects model from default save-dir if no path given.

Usage:
    python scripts/10_eval.py                                    # auto-detect model, constant_velocity
    python scripts/10_eval.py --target zigzag                    # eval vs zigzag
    python scripts/10_eval.py --target zigzag --target-speed 8   # harder target
    python scripts/10_eval.py --model models/ppo_intercept_250000.zip
    python scripts/10_eval.py --episodes 200 --no-plot
"""

import argparse
from pathlib import Path

from drone_intercept.training.eval_policy import evaluate

DEFAULT_MODEL = "models/ppo_intercept_final.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained interception policy")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Path to model (default: {DEFAULT_MODEL})")
    parser.add_argument("--target", type=str, default="constant_velocity",
                        choices=["constant_velocity", "waypoint", "zigzag"])
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log directory (default: logs/eval_{target})")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--all-targets", action="store_true",
                        help="Evaluate against all 3 target types")
    args = parser.parse_args()

    model_path = args.model or DEFAULT_MODEL
    if not Path(model_path).exists():
        print(f"No model found at {model_path}")
        print("Train one first: python scripts/09_train_full.py")
        raise SystemExit(1)

    targets = ["constant_velocity", "waypoint", "zigzag"] if args.all_targets else [args.target]
    results = {}

    for target in targets:
        log_dir = args.log_dir or f"logs/eval_{target}"
        print(f"\n{'='*50}")
        print(f"Evaluating vs {target} (speed={args.target_speed}) | {args.episodes} episodes")
        print(f"Model: {model_path}")
        print(f"{'='*50}\n")

        r = evaluate(
            model_path=model_path,
            n_episodes=args.episodes,
            target_behavior=target,
            target_speed=args.target_speed,
            log_dir=log_dir,
            plot=not args.no_plot,
            seed=args.seed,
            device=args.device,
        )
        results[target] = r

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"{'Target':<22} {'Success':>8} {'Avg Reward':>12} {'Avg Cap Time':>13}")
        print(f"{'-'*55}")
        for target, r in results.items():
            cap_t = f"{r['avg_capture_time']:.1f}s" if r["avg_capture_time"] else "N/A"
            print(f"{target:<22} {r['success_rate']:>7.0%} {r['avg_reward']:>11.1f} {cap_t:>13}")


if __name__ == "__main__":
    main()
