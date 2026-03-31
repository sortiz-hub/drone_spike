# SPEC025 - Delivery Strategy

## Overview

Four phases mapping to progressive environment complexity. **Phase 1 is the current focus** with detailed milestones. Phases 2-4 are outlined for planning — detail will be added when each phase begins.

Full technical reference: `drone_interception_rl_blueprint.md` (blueprint sections 7-10, 13-14, 19-24 cover the progression in detail).

---

## Phase 1: Cheated Interception (MVP)

**Goal:** Train a PPO policy that intercepts a scripted target using simulator truth state in open space.

**Status:** `not-started`

##### M1.1: PX4 SITL + Gazebo Environment Setup
- **Status**: `not-started`
- **Description**: Bring up PX4 SITL with Gazebo, verify a drone can arm and hover. Run a ROS 2 Offboard velocity command example.
- **Acceptance Criteria**:
  - [ ] PX4 SITL launches with Gazebo world (open arena, no obstacles)
  - [ ] ROS 2 bridge operational (mavros or px4_ros_com)
  - [ ] Drone arms, takes off, and accepts velocity setpoints via Offboard mode
  - [ ] Scripted test: drone moves to a waypoint via velocity commands

##### M1.2: Target Actor
- **Status**: `not-started`
- **Description**: Create a scripted target entity in Gazebo that moves with configurable behavior. Expose its truth state via ROS 2.
- **Acceptance Criteria**:
  - [ ] Target spawns in Gazebo with visible model
  - [ ] Constant-velocity behavior implemented and configurable (speed, direction)
  - [ ] Waypoint-following behavior implemented
  - [ ] Target state published on ROS 2 topic (position, velocity)

##### M1.3: Gymnasium Environment Wrapper
- **Status**: `not-started`
- **Description**: Create a Gymnasium-compatible environment wrapping the PX4 + Gazebo + target setup with proper observation/action/reward spaces.
- **Acceptance Criteria**:
  - [ ] `InterceptEnv` implements `gymnasium.Env` with correct spaces
  - [ ] `reset()` resets drone + target positions, returns initial obs
  - [ ] `step(action)` sends velocity command, advances sim, returns (obs, reward, terminated, truncated, info)
  - [ ] Observation: 14D vector (self pos/vel + relative target pos/vel + distance + battery)
  - [ ] Action: 4D continuous (vx, vy, vz, yaw_rate) with configurable bounds
  - [ ] Reward: distance shaping + capture bonus (+100) + crash penalty (-100)
  - [ ] Termination: capture (d<1.5m, v_rel<2.0m/s), crash, timeout, out-of-bounds
  - [ ] Scripted baseline policy achieves some captures (validates env correctness)

##### M1.4: PPO Training Pipeline
- **Status**: `not-started`
- **Description**: Train PPO with Stable-Baselines3 on the interception env. Achieve >80% capture rate against constant-velocity target.
- **Acceptance Criteria**:
  - [ ] `train_ppo.py` script with configurable hyperparameters
  - [ ] Training runs and converges (reward curve trends upward)
  - [ ] >80% capture rate against constant-velocity target after training
  - [ ] Model checkpoints saved periodically
  - [ ] `eval_policy.py` script for evaluation runs

##### M1.5: Episode Logging & Replay
- **Status**: `not-started`
- **Description**: Implement per-step trajectory logging and 2D replay plots.
- **Acceptance Criteria**:
  - [ ] Per-step JSONL log: t, drone_pos, drone_vel, target_pos, target_vel, action, reward, done
  - [ ] Per-episode summary: total reward, success/failure, capture time, min distance
  - [ ] 2D top-down trajectory plot (drone path, target path, capture/collision marker)
  - [ ] Training metrics logged: episode reward, length, success rate, capture time
  - [ ] Video export for selected episodes (every N episodes during training)

##### M1.6: Target Behavior Curriculum
- **Status**: `not-started`
- **Description**: Add configurable target behaviors beyond constant velocity. Evaluate policy generalization.
- **Acceptance Criteria**:
  - [ ] Zigzag target behavior implemented
  - [ ] Target behavior selectable via env config parameter
  - [ ] Policy trained on constant velocity still tested against zigzag (generalization baseline)
  - [ ] Optional: curriculum training across behaviors

---

## Phase 2: Tracked Target (Realistic Sensing)

**Goal:** Replace simulator truth with noisy detections + Kalman tracker. Policy must handle uncertainty.

**Status:** `not-started`

**Key changes:**
- Gaussian noise injected into target detections
- Kalman filter tracker node smooths detections
- Observation switches to tracked state + confidence
- Policy must handle track loss gracefully

**Observation schema (Phase 2):**
```python
obs = {
    "self_state": [x, y, z, vx, vy, vz],
    "tracked_target_pos": [x, y, z],
    "tracked_target_vel": [vx, vy, vz],
    "track_confidence": float,
}
```

**Milestones (to be detailed when phase begins):**
- M2.1: Noise injection and detection simulation
- M2.2: Kalman filter tracker node
- M2.3: Observation builder integration
- M2.4: Policy training and evaluation under uncertainty

---

## Phase 3: Obstacle-Aware Interception

**Goal:** Add obstacles to the environment and local obstacle perception to the observation. Policy learns safe pursuit.

**Status:** `not-started`

**Key changes:**
- Gazebo worlds with static obstacles
- Depth sensor or occupancy grid for local obstacle awareness
- Observation gains obstacle sector distances
- Reward gains near-obstacle and collision penalties

**Observation schema (Phase 3):**
```python
obs = {
    "self_state": ...,
    "tracked_target": ...,
    "local_obstacles": [d_sector_0, ..., d_sector_N],  # distances per angular sector
}
```

**Milestones (to be detailed when phase begins):**
- M3.1: Obstacle-populated Gazebo worlds
- M3.2: Local obstacle perception (depth/occupancy)
- M3.3: Observation and reward integration
- M3.4: Policy training with safe pursuit

---

## Phase 4: Prediction-Aware Interception

**Goal:** Add target prediction to enable lead pursuit / smarter interception strategies.

**Status:** `not-started`

**Key changes:**
- Predictor node (constant velocity initially, learned later)
- Observation gains predicted future target positions
- Policy can anticipate target trajectory

**Observation schema (Phase 4):**
```python
obs = {
    "self_state": ...,
    "tracked_target": ...,
    "predicted_target_t05": [x, y, z],  # 0.5s ahead
    "predicted_target_t10": [x, y, z],  # 1.0s ahead
    "local_obstacles": ...,
}
```

**Milestones (to be detailed when phase begins):**
- M4.1: Constant-velocity predictor node
- M4.2: Prediction integration in observation
- M4.3: Policy training with lead pursuit
- M4.4: Optional: learned predictor (RNN/transformer)

---

## Blueprint Reference

The full drone interception RL blueprint (`drone_interception_rl_blueprint.md`) contains detailed technical guidance for all phases:

| Blueprint Section | Relevant Phase | Content |
|-------------------|---------------|---------|
| 1-3 | All | Big picture, tech stack, agent types |
| 4 | All | Interception problem definition |
| 5-6 | Phase 2-3 | Sensor stack, detection, real-world strategy |
| 7 | All | Pipeline pieces with data contracts per layer |
| 8 | All | Staged roadmap (Phase A-C) |
| 9 | Phase 1 | Recommended stack (PX4 + Gazebo + ROS 2 + Gymnasium + SB3) |
| 10 | All | Implementation phases with observation schemas |
| 11-12 | Phase 1 | Observation/action/reward design, capture condition |
| 13 | All | Project structure |
| 14 | Phase 1 | Target behavior progression |
| 15 | Phase 1 | Algorithm recommendation (PPO first) |
| 16 | All | What NOT to do first |
| 17-18 | Phase 1 | MVP milestone and build order |
| 19-24 | All | Episode observability, logging, visualization |
| 25-26 | All | Final MVP path and summary |

The blueprint should be stored in the spec folder as `blueprint-reference.md` for direct access during implementation.
