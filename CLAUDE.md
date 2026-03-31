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
```

## Scope and Focus

- This is a standalone spike repo for drone interception RL experimentation
- Part of the RL-EnergyPlus platform ecosystem (parent: `rl-platform-root`)
- When asked to investigate or search, **start in this repo** unless told otherwise

## AI Behavior Guidelines

### Git Operations
- **DO NOT** propose or attempt git commit operations unless explicitly requested
- **DO NOT** run destructive git commands unless explicitly requested
- Follow commit message conventions: `type: description` (docs, feat, fix, chore, refactor)

### Services and Containers
- **DO NOT** start long-running services (PX4 SITL, Gazebo, ROS 2, training runs) unless explicitly requested

### Documentation Usage
Before answering architecture questions or implementing features:
1. **READ `.agent/README.md`** for documentation navigation
2. **CHECK `.agent/specs/`** for existing feature specs before planning new work

## Tech Stack

- **Dynamics**: Simplified first-order velocity model (Phase 1), Gazebo + PX4 SITL (Phase 2+)
- **RL Interface**: Python Gymnasium
- **Training**: Stable-Baselines3 (PPO)
- **Visualization**: matplotlib (2D trajectory plots)
- **Language**: Python 3.10+

## Key Modules

```
drone_intercept/
  env/intercept_env.py        # Gymnasium environment (InterceptEnv)
  env/observation_builder.py  # 14D obs vector
  env/rewards.py              # Reward: distance + effort + capture/crash
  env/termination.py          # Capture / crash / timeout / OOB
  sim/target_behaviors/       # ConstantVelocity, Waypoint, Zigzag
  training/train_ppo.py       # CLI training entry point
  training/eval_policy.py     # CLI evaluation entry point
  replay/logger.py            # JSONL episode logger
  replay/plotter.py           # 2D trajectory plotter
```
