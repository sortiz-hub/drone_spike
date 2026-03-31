# CLAUDE.md

This file provides behavioral rules for Claude Code. For project documentation, see **`.agent/README.md`**.

---

## Project

**drone_spike** — Drone interception RL spike. Training pursuit-evasion policies using PX4 + Gazebo + ROS 2 + Gymnasium + Stable-Baselines3.

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
- **DO NOT** start long-running services (PX4 SITL, Gazebo, ROS 2) unless explicitly requested

### Documentation Usage
Before answering architecture questions or implementing features:
1. **READ `.agent/README.md`** for documentation navigation
2. **CHECK `.agent/specs/`** for existing feature specs before planning new work

## Tech Stack

- **Simulator**: Gazebo + PX4 SITL
- **Middleware**: ROS 2
- **RL Interface**: Python Gymnasium
- **Training**: Stable-Baselines3 (PPO)
- **Language**: Python
- **OS**: Linux / WSL2
