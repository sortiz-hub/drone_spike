# drone_spike — Architecture Overview

## 1. Purpose

Spike repository for developing drone interception RL agents. The system trains a decision policy that outputs velocity setpoints to a PX4 autopilot to intercept a moving target in simulation.

## 2. Core Principle

> You are not training "a drone". You are training **a decision policy over an estimated world state**.

RL operates at the **decision/guidance layer** — not at the motor level. PX4 handles stabilization and low-level control.

## 3. Runtime Architecture

### Phase 1 (Current — Simplified Dynamics)

The current implementation uses a first-order velocity lag model instead of PX4 SITL + Gazebo + ROS 2. This allows immediate training without simulator dependencies. The architecture is designed so the dynamics layer can be swapped for the full stack later.

```
Simplified physics (first-order velocity tracking)
   ↕
Gymnasium env wrapper (obs/action/reward/termination)
   ↕
SB3 PPO (training) or trained policy (inference)
```

### Phase 2+ (Planned — Full Stack)

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

## 4. Project Structure

```
drone_spike/
  drone_intercept/               # Main Python package
    sim/
      target_behaviors/
        base.py                  # TargetBehavior ABC
        constant_velocity.py     # Straight-line target
        waypoint.py              # Waypoint-following target
        zigzag.py                # Zigzag evasion target
    env/
      intercept_env.py           # Gymnasium environment (InterceptEnv)
      observation_builder.py     # 14D observation vector assembly
      rewards.py                 # Reward computation
      termination.py             # Episode end conditions
    training/
      train_ppo.py               # PPO training entry point (CLI)
      eval_policy.py             # Policy evaluation (CLI)
      callbacks.py               # SB3 training callbacks
    replay/
      logger.py                  # Per-step JSONL logger + episode summaries
      plotter.py                 # 2D top-down trajectory plots
    ros2_nodes/                  # ROS 2 integration (placeholder for Phase 2+)
  models/                        # Saved policy checkpoints (gitignored)
  logs/                          # Training logs and episode data (gitignored)
  pyproject.toml                 # Package config + dependencies
```

## 5. Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dynamics | Simplified first-order model (Phase 1) | Velocity tracking proxy |
| RL interface | Gymnasium | `reset()` / `step()` / reward / termination |
| Training | Stable-Baselines3 (PPO) | Policy optimization |
| Logging | JSONL + CSV | Episode trajectory and summary data |
| Visualization | matplotlib | 2D trajectory plots |
| Language | Python 3.10+ | All application code |

### Future Stack (Phase 2+)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Simulator | Gazebo | Physics, world, sensors |
| Flight controller | PX4 SITL | State estimation, stabilization, offboard |
| Middleware | ROS 2 | Node communication |

## 6. Phased Approach

| Phase | Focus | Sensing | Status |
|-------|-------|---------|--------|
| 1 - Cheated Interception (MVP) | Basic pursuit in open space | Simulator truth | **Implemented** |
| 2 - Tracked Target | Pursuit under uncertainty | Noisy detections + Kalman | Not started |
| 3 - Obstacle-Aware | Safe pursuit | Depth/occupancy + tracked target | Not started |
| 4 - Prediction-Aware | Lead pursuit | Predicted target trajectory | Not started |

See `specs/SPEC025-drone-interception-rl/delivery-strategy.md` for detailed milestones.

## 7. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL layer | Decision/guidance, not motor control | Leverage PX4 for stabilization; faster convergence |
| Phase 1 dynamics | Simplified first-order lag | Instant training, no simulator setup; swap later |
| Velocity tracking tau | 0.3s time constant | Mimics realistic autopilot response lag |
| Action space | Continuous velocity + yaw rate | Natural PX4 Offboard interface |
| Observation | 14D compact vector | Fast training; no images initially |
| Algorithm | PPO | Proven for continuous control; simple SB3 baseline |
| Targets | Scripted behaviors | Debuggable curriculum before adversarial |

## 8. Observation, Action, Reward Reference

### Observation (14D vector)

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-2 | self_pos | [-200, 200] m | Drone position (x, y, z) |
| 3-5 | self_vel | [-30, 30] m/s | Drone velocity (vx, vy, vz) |
| 6-8 | rel_target_pos | [-300, 300] m | Target - Drone position |
| 9-11 | rel_target_vel | [-60, 60] m/s | Target_vel - Drone_vel |
| 12 | distance | [0, 300] m | Euclidean distance to target |
| 13 | battery | [0, 1] | Normalized battery level |

### Action (4D continuous)

| Index | Field | Range | Unit |
|-------|-------|-------|------|
| 0 | vx_cmd | [-10, 10] | m/s |
| 1 | vy_cmd | [-10, 10] | m/s |
| 2 | vz_cmd | [-10, 10] | m/s |
| 3 | yaw_rate | [-2, 2] | rad/s |

### Reward

```
r = -0.1 * distance
    - 0.01 * control_effort
    - 0.05 * collision_risk (altitude < 1.0m)
    + 100.0 (capture)
    - 100.0 (crash)
```

### Termination

| Condition | Trigger | Type |
|-----------|---------|------|
| Capture | dist < 1.5m AND rel_speed < 2.0 m/s | terminated |
| Crash (ground) | altitude < 0.3m | terminated |
| Crash (ceiling) | altitude > 50m | terminated |
| Out of bounds | horiz_dist > 100m | terminated |
| Timeout | step >= 1000 | truncated |
