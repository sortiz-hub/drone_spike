"""Step 7: Short training run — does PPO learn anything?

Run via CLI for full control:
    python -m drone_intercept.training.train_ppo --timesteps 50000 --device auto

This script does the same with minimal settings for a quick sanity check.

Usage:
    python scripts/07_short_training.py              # auto-detect
    python scripts/07_short_training.py --device cuda # force GPU
    python scripts/07_short_training.py --device cpu  # force CPU
"""

import argparse

from drone_intercept.training.train_ppo import train

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cuda", "cpu"])
args = parser.parse_args()

model = train(total_timesteps=50_000, n_envs=2, device=args.device, save_dir="models", log_dir="logs")
print("PASS — check logs for reward trend")
