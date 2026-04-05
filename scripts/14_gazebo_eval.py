import warnings
warnings.filterwarnings("ignore", message="Unable to import Axes3D")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

"""Step 14: Evaluate a trained model in PX4+Gazebo with visual target.

Loads a model trained with the simplified backend and runs it against
the real PX4 simulator. Watch the Gazebo window to see the chase.

Run INSIDE the Docker container (with PX4 SITL + MAVROS running):
    source /opt/ros/humble/setup.bash
    cd /workspace/drone_spike
    python3 scripts/14_gazebo_eval.py
    python3 scripts/14_gazebo_eval.py --model models/ppo_intercept_final.zip --episodes 5
    python3 scripts/14_gazebo_eval.py --target zigzag --target-speed 3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rclpy
except ImportError:
    print("ERROR: ROS 2 not available. Run inside the Docker container.")
    sys.exit(1)

from stable_baselines3 import PPO

from drone_intercept.env.intercept_env import InterceptEnv
from drone_intercept.env.termination import TerminationConfig
from drone_intercept.replay.logger import EpisodeLogger
from drone_intercept.replay.plotter import plot_episode, animate_episode

DEFAULT_MODEL = "models/ppo_intercept_final.zip"


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model in PX4+Gazebo")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--target", type=str, default="constant_velocity",
                        choices=["constant_velocity", "waypoint", "zigzag"])
    parser.add_argument("--target-speed", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--log-dir", type=str, default="logs/gazebo_eval")
    parser.add_argument("--pause-between", type=float, default=3.0,
                        help="Seconds to pause between episodes (for Gazebo to settle)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video/plot generation")
    args = parser.parse_args()

    model_path = args.model
    if not Path(model_path).exists():
        print(f"No model found at {model_path}")
        print("Train one first: python scripts/09_train_full.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    cfg = TerminationConfig(
        max_steps=args.max_steps,
        arena_radius=500.0,    # larger arena for Gazebo (real sim distances)
        max_altitude=100.0,
    )
    print(f"Creating PX4+Gazebo env (target={args.target}, speed={args.target_speed})...")
    env = InterceptEnv(
        physics_backend="px4_gazebo",
        target_behavior=args.target,
        target_speed=args.target_speed,
        dt=0.1,
        termination=cfg,
    )

    logger = EpisodeLogger(log_dir=args.log_dir)

    results = []
    for ep in range(args.episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep}/{args.episodes}")
        print(f"{'='*50}")

        obs, info = env.reset(seed=ep)
        logger.on_reset()
        print(f"Drone: [{info['drone_pos'][0]:.1f}, {info['drone_pos'][1]:.1f}, {info['drone_pos'][2]:.1f}]")
        print(f"Target: [{info['target_pos'][0]:.1f}, {info['target_pos'][1]:.1f}, {info['target_pos'][2]:.1f}]")

        total_reward = 0.0
        min_dist = float("inf")

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            logger.on_step(info, action, reward, terminated, truncated)

            total_reward += reward
            dist = info["distance"]
            min_dist = min(min_dist, dist)

            if step % 25 == 0:
                print(f"  step={step:3d}  dist={dist:.1f}m  reward={reward:.2f}")

            if terminated or truncated:
                break

        summary = logger.on_episode_end(info, dt=env.dt)
        result_str = "CAPTURE" if summary.success else summary.reason
        results.append({
            "episode": ep,
            "result": result_str,
            "reward": total_reward,
            "steps": summary.steps,
            "min_dist": min_dist,
            "capture_time": summary.capture_time,
        })

        print(f"  Result: {result_str}")
        print(f"  Reward: {total_reward:.1f}, Min dist: {min_dist:.2f}m")
        if summary.capture_time:
            print(f"  Capture time: {summary.capture_time:.1f}s")

        # Generate plot + video from JSONL log
        if not args.no_video:
            ep_file = Path(args.log_dir) / f"episode_{ep:05d}.jsonl"
            if ep_file.exists():
                steps = EpisodeLogger.load_episode(ep_file)
                # Static plot
                plot_path = Path(args.log_dir) / f"plot_ep{ep:03d}_{result_str.lower()}.png"
                plot_episode(steps, title=f"Gazebo Ep {ep} ({result_str})",
                             save_path=plot_path, show=False)
                # Animated replay (mp4 if ffmpeg available, gif fallback)
                vid_path = Path(args.log_dir) / f"video_ep{ep:03d}_{result_str.lower()}.mp4"
                animate_episode(steps, save_path=vid_path, fps=10,
                                title=f"Gazebo Ep {ep} ({result_str})")
                print(f"  Plot: {plot_path}")
                print(f"  Video: {vid_path}")

        if ep < args.episodes - 1:
            print(f"  Waiting {args.pause_between}s for Gazebo to settle...")
            time.sleep(args.pause_between)

    env.close()

    # Summary
    print(f"\n{'='*50}")
    print("GAZEBO EVALUATION SUMMARY")
    print(f"Model: {model_path}")
    print(f"Target: {args.target} @ {args.target_speed} m/s")
    print(f"{'='*50}")
    print(f"{'Ep':>3} {'Result':<16} {'Reward':>8} {'Steps':>6} {'Min Dist':>9} {'Cap Time':>9}")
    print(f"{'-'*53}")

    captures = 0
    for r in results:
        cap_t = f"{r['capture_time']:.1f}s" if r["capture_time"] else "-"
        print(f"{r['episode']:>3} {r['result']:<16} {r['reward']:>8.1f} {r['steps']:>6} {r['min_dist']:>8.2f}m {cap_t:>9}")
        if r["result"] == "CAPTURE":
            captures += 1

    print(f"\nSuccess rate: {captures}/{len(results)} ({captures/len(results):.0%})")
    print(f"Logs saved to: {args.log_dir}/")


if __name__ == "__main__":
    main()
