# SPEC025 - Architecture

## 1. Overview

The system follows the layered autonomy pipeline from the blueprint, with RL operating at the **decision/policy layer** (Layer 5). Phase 1 shortcuts layers 2-4 by using simulator truth.

```
Phase 1 (cheated):
  Simulator truth state -> RL Policy -> Velocity setpoints -> PX4 Offboard

Phase 2+ (realistic):
  Sensors -> Perception -> Tracking -> Prediction -> RL Policy -> Velocity setpoints -> PX4 Offboard
```

## 2. Component Architecture

### 2.1 Runtime Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Simulator | Gazebo + PX4 SITL | Physics, flight dynamics, world state |
| Middleware | ROS 2 | Communication between nodes |
| RL Interface | Python Gymnasium wrapper | `reset()` / `step()` / reward / termination |
| Training | Stable-Baselines3 (PPO) | Policy optimization |
| Logging | JSONL / CSV + matplotlib | Episode replay and analysis |

### 2.2 Key Components

```
drone_intercept/
  sim/
    world_config/           # Gazebo world definitions
    target_behaviors/       # Scripted target motion patterns
  ros2_nodes/
    vehicle_state_node.py   # Reads PX4 state estimation
    target_provider_node.py # Publishes target state (truth or tracked)
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
    callbacks.py            # SB3 callbacks (logging, video, checkpoints)
  replay/
    logger.py               # Per-step JSONL logger
    plotter.py              # 2D top-down trajectory plots
  models/                   # Saved policy checkpoints
  logs/                     # Training logs and episode data
```

### 2.3 Data Flow (Phase 1)

```
Gazebo World
    |
    v
PX4 SITL (state estimation, stabilization)
    |
    v
ROS 2 Nodes:
  vehicle_state_node  --> interceptor state (pos, vel, orientation)
  target_provider_node --> target state (pos, vel) [simulator truth]
    |
    v
Gymnasium Env (intercept_env.py):
  observation_builder --> obs vector (14D)
  rewards.py          --> scalar reward
  termination.py      --> done flag
    |
    v
SB3 PPO:
  policy.predict(obs) --> action (4D velocity command)
    |
    v
offboard_command_node --> PX4 Offboard mode --> drone motion
```

## 3. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RL layer | Decision/guidance, not motor control | Leverage PX4 for stabilization; faster training convergence |
| Initial sensing | Simulator truth (cheated) | Isolate policy learning from perception noise |
| Action space | Continuous velocity + yaw rate | Natural interface to PX4 Offboard; avoids motor-level complexity |
| Algorithm | PPO | Proven for continuous control; simple SB3 setup; good baseline |
| Observation | Compact vector (no images) | Fast training; add image obs later if needed |
| Target | Scripted behaviors, not adversarial | Debuggable; curriculum-friendly; adversarial is Phase 5+ |

## 4. Phase-Specific Architecture Changes

### Phase 2: Tracked Target
- Add `tracker_node.py` (Kalman filter / EKF)
- `target_provider_node` injects Gaussian noise into detections
- Observation gains `track_confidence` field
- `observation_builder` switches from truth to tracked state

### Phase 3: Obstacle-Aware
- Gazebo worlds include obstacles
- Add obstacle perception (depth sensor or occupancy grid)
- Observation gains `local_obstacles` sector distances
- Reward gains obstacle proximity penalty

### Phase 4: Prediction-Aware
- Add `predictor_node.py` (constant velocity / learned)
- Observation gains `predicted_target_t05`, `predicted_target_t10`
- Enables lead pursuit strategies
