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
      backends/
        base.py                  # PhysicsBackend ABC
        simplified.py            # First-order velocity model (default)
      target_behaviors/
        base.py                  # TargetBehavior ABC
        constant_velocity.py     # Straight-line target
        waypoint.py              # Waypoint-following target
        zigzag.py                # Zigzag evasion target
      noise.py                   # Gaussian noise injection (Phase 2)
      tracker.py                 # Kalman filter target tracker (Phase 2)
      obstacles.py               # Obstacle generation + sector perception (Phase 3)
      predictor.py               # Constant-velocity target predictor (Phase 4)
    env/
      intercept_env.py           # Gymnasium environment (InterceptEnv)
      observation_builder.py     # Observation vector assembly (14D/15D)
      rewards.py                 # Reward computation
      termination.py             # Episode end conditions
    training/
      train_ppo.py               # PPO training entry point (CLI)
      eval_policy.py             # Policy evaluation (CLI)
      callbacks.py               # SB3 training callbacks
    replay/
      logger.py                  # Per-step JSONL logger + episode summaries
      plotter.py                 # 2D trajectory plots + video export
    ros2_nodes/                  # ROS 2 integration (placeholder)
  models/                        # Saved policy checkpoints (gitignored)
  logs/                          # Training logs and episode data (gitignored)
  scripts/                       # Validation scripts 01–12 + run_all.py
  docker/
    Dockerfile                   # PX4 + Gazebo + ROS 2 Humble image
    docker-compose.yml           # Container orchestration
  requirements-cuda.txt          # GPU PyTorch install (--extra-index-url)
  .vscode/launch.json            # Debug config for current file with venv
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

### Physics Backend Abstraction

`InterceptEnv` accepts a `physics_backend` parameter to select the dynamics implementation:

- **`simplified`** (default) — First-order velocity lag model (`sim/backends/simplified.py`). No external dependencies.
- Future backends (e.g., Gazebo) will implement the same `PhysicsBackend` ABC (`sim/backends/base.py`).

### Reward Modes

`env/rewards.py` supports two modes via `RewardConfig`:

- **`original`** — Distance penalty + effort penalty + capture/crash bonuses (the default from Phase 1).
- **`shaped`** — Enhanced shaping with smoother distance gradients and additional terms for training stability.

Select with `--reward-mode original|shaped` on the training CLI.

### Future Stack (Phase 2+)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Simulator | Gazebo | Physics, world, sensors |
| Flight controller | PX4 SITL | State estimation, stabilization, offboard |
| Middleware | ROS 2 | Node communication |

## 6. Docker Infrastructure

The `docker/` directory provides a containerized PX4 + Gazebo + ROS 2 Humble environment for Phase 2+ development:

```bash
cd docker && docker compose up -d    # Start simulator stack
docker exec -it drone-sim bash       # Shell into the container
```

- `docker/Dockerfile` — Builds an image with PX4 SITL, Gazebo, and ROS 2 Humble
- `docker/docker-compose.yml` — Orchestrates the container with proper volume mounts and network config

## 7. Phased Approach

| Phase | Focus | Sensing | Status |
|-------|-------|---------|--------|
| 1 - Cheated Interception (MVP) | Basic pursuit in open space | Simulator truth | **Implemented** |
| 2 - Tracked Target | Pursuit under uncertainty | Noisy detections + Kalman | **Implemented** |
| 3 - Obstacle-Aware | Safe pursuit | Sector distances + tracked target | **Implemented** |
| 4 - Prediction-Aware | Lead pursuit | Predicted target trajectory | **Implemented** |

See `specs/SPEC025-drone-interception-rl/delivery-strategy.md` for detailed milestones.

## 8. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL layer | Decision/guidance, not motor control | Leverage PX4 for stabilization; faster convergence |
| Phase 1 dynamics | Simplified first-order lag | Instant training, no simulator setup; swap later |
| Velocity tracking tau | 0.3s time constant | Mimics realistic autopilot response lag |
| Action space | Continuous velocity + yaw rate | Natural PX4 Offboard interface |
| Observation | 14D–29D (phase-dependent) | Compact vector; grows with phase features |
| Algorithm | PPO | Proven for continuous control; simple SB3 baseline |
| Targets | Scripted behaviors | Debuggable curriculum before adversarial |

## 9. Observation, Action, Reward Reference

### Observation — Phase 1 (14D vector, sensing_mode="truth")

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-2 | self_pos | [-200, 200] m | Drone position (x, y, z) |
| 3-5 | self_vel | [-30, 30] m/s | Drone velocity (vx, vy, vz) |
| 6-8 | rel_target_pos | [-300, 300] m | Target - Drone position |
| 9-11 | rel_target_vel | [-60, 60] m/s | Target_vel - Drone_vel |
| 12 | distance | [0, 300] m | Euclidean distance to target |
| 13 | battery | [0, 1] | Normalized battery level |

### Observation — Phase 2 (15D vector, sensing_mode="tracked")

Extends Phase 1 with tracker output instead of truth state:

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-13 | (same as Phase 1) | | rel_target uses tracked estimate |
| 14 | track_confidence | [0, 1] | Kalman tracker confidence |

### Observation — Phase 3 (adds obstacle sectors)

Appends sector distances to the Phase 1 or Phase 2 vector:

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 14 or 15 | sector_0 .. sector_N | [0, 20] m | Distance to nearest obstacle per angular sector |

Total dimensions: 22 (truth+obstacles) or 23 (tracked+obstacles) with 8 default sectors.

### Observation — Phase 4 (adds predicted target positions)

Appends relative predicted target positions (constant-velocity extrapolation):

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| +0..+2 | predicted_t05 | [-300, 300] m | Predicted target position at t+0.5s (relative to drone) |
| +3..+5 | predicted_t10 | [-300, 300] m | Predicted target position at t+1.0s (relative to drone) |

Full stack dimensions: 29 (tracked + 8 sectors + 2 predictions).

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
    - 0.1 * obstacle_proximity (within 3m, Phase 3)
    + 100.0 (capture)
    - 100.0 (crash or obstacle collision)
```

### Termination

| Condition | Trigger | Type |
|-----------|---------|------|
| Capture | dist < 1.5m AND rel_speed < 2.0 m/s | terminated |
| Crash (ground) | altitude < 0.3m | terminated |
| Crash (ceiling) | altitude > 50m | terminated |
| Crash (obstacle) | collides with obstacle (Phase 3) | terminated |
| Out of bounds | horiz_dist > 100m | terminated |
| Timeout | step >= 1000 | truncated |
