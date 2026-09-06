# CLAUDE.md

This file provides behavioral rules for Claude Code. For project documentation, see **`.agent/README.md`**.

---

## Project

**drone_spike** — Drone interception RL spike. Training pursuit-evasion policies using Gymnasium + Stable-Baselines3 (PPO). Phase 1 uses simplified dynamics; Phase 2+ will integrate PX4 + Gazebo + ROS 2.

## Quick Commands

```bash
# Install
pip install -e ".[dev]"

# Train PPO (constant-velocity target, 500k steps)
python -m drone_intercept.training.train_ppo --timesteps 500000

# Train against zigzag target
python -m drone_intercept.training.train_ppo --target zigzag --timesteps 500000

# Evaluate trained policy
python -m drone_intercept.training.eval_policy models/ppo_intercept_final.zip --episodes 100

# Plot a logged episode
python -c "from drone_intercept.replay.plotter import plot_episode_from_file; plot_episode_from_file('logs/eval/episode_00000.jsonl')"

# Run all validation scripts (quick mode)
python scripts/run_all.py --quick

# Full training via script
python scripts/09_train_full.py

# Evaluate across all targets
python scripts/10_eval.py --all-targets

# Docker (PX4 + Gazebo + ROS 2 Humble)
cd docker && docker compose up -d
docker exec -it drone-sim bash
```

## Scope and Focus

- This is a standalone spike repo for drone interception RL experimentation
- Part of the RL-EnergyPlus platform ecosystem (parent: `rl-energyplus`)

## Key Modules

Everything lives under `drone_intercept/` (`env/`, `sim/`, `training/`, `replay/`).
Two non-obvious details: the observation vector is **14D**, and `env/rewards.py`
carries two reward modes (`original` and `shaped`) — check which one a run used
before comparing results.
