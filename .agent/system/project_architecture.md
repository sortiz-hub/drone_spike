# drone_spike — Architecture Overview

## 1. Purpose

Spike repository for developing drone interception RL agents. The system trains a decision policy that outputs velocity setpoints to a PX4 autopilot to intercept a moving target in simulation.

## 2. Core Principle

> You are not training "a drone". You are training **a decision policy over an estimated world state**.

RL operates at the **decision/guidance layer** — not at the motor level. PX4 handles stabilization and low-level control.

## 3. Runtime Architecture

```
Gazebo world (physics, target, obstacles)
   ↕
PX4 SITL (state estimation, stabilization, offboard)
   ↕
ROS 2 nodes (perception, tracking, prediction, commands)
   ↕
Gymnasium env wrapper (obs/action/reward/termination)
   ↕
SB3 PPO (training) or trained policy (inference)
```

## 4. Planned Project Structure

```
drone_spike/
  sim/
    world_config/           # Gazebo world definitions
    target_behaviors/       # Scripted target motion patterns
  ros2_nodes/
    vehicle_state_node.py   # Reads PX4 state estimation
    target_provider_node.py # Publishes target state
    tracker_node.py         # [Phase 2] Kalman filter tracker
    predictor_node.py       # [Phase 4] Target prediction
    offboard_command_node.py # Sends velocity setpoints to PX4
  env/
    intercept_env.py        # Gymnasium environment
    rewards.py              # Reward functions
    termination.py          # Episode end conditions
    observation_builder.py  # Observation assembly
  training/
    train_ppo.py            # Training entry point
    eval_policy.py          # Policy evaluation
    callbacks.py            # SB3 callbacks
  replay/
    logger.py               # Per-step JSONL logger
    plotter.py              # 2D top-down trajectory plots
  models/                   # Saved policy checkpoints
  logs/                     # Training logs and episode data
```

## 5. Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Simulator | Gazebo | Physics, world, sensors |
| Flight controller | PX4 SITL | State estimation, stabilization, offboard |
| Middleware | ROS 2 | Node communication |
| RL interface | Gymnasium | Environment wrapper |
| Training | Stable-Baselines3 (PPO) | Policy optimization |
| Language | Python | All application code |
| OS | Linux / WSL2 | Required for PX4 + Gazebo |

## 6. Phased Approach

| Phase | Focus | Sensing |
|-------|-------|---------|
| 1 - Cheated Interception (MVP) | Basic pursuit in open space | Simulator truth |
| 2 - Tracked Target | Pursuit under uncertainty | Noisy detections + Kalman |
| 3 - Obstacle-Aware | Safe pursuit | Depth/occupancy + tracked target |
| 4 - Prediction-Aware | Lead pursuit | Predicted target trajectory |

See `specs/SPEC025-drone-interception-rl/delivery-strategy.md` for detailed milestones.

## 7. Key Design Decisions

- **RL at guidance layer**: Leverage PX4 for stabilization; faster convergence
- **Velocity setpoints**: Natural PX4 Offboard interface; avoids motor complexity
- **PPO first**: Proven for continuous control; simple SB3 baseline
- **Compact vector obs**: No images initially; fast training iterations
- **Scripted targets first**: Debuggable curriculum before adversarial
