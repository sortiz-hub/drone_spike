# SPEC025 - Delivery Strategy

## Overview

Four phases mapping to progressive environment complexity. **Phase 1 is the current focus** with detailed milestones. Phases 2-4 are outlined for planning — detail will be added when each phase begins.

Full technical reference: `drone_interception_rl_blueprint.md` (blueprint sections 7-10, 13-14, 19-24 cover the progression in detail).

---

## Phase 1: Cheated Interception (MVP)

**Goal:** Train a PPO policy that intercepts a scripted target using simulator truth state in open space.

**Status:** `in-progress`

##### M1.1: PX4 SITL + Gazebo Environment Setup
- **Status**: `implemented`
- **Description**: Bring up PX4 SITL with Gazebo, verify a drone can arm and hover. Run a ROS 2 Offboard velocity command example.
- **Implementation**: Docker container (`docker/Dockerfile`) with PX4 v1.15.2 + Gazebo Harmonic 8.11 + ROS 2 Humble + MAVROS. GUI via VcXsrv X11 forwarding. RTX 5090 GPU passthrough confirmed.
- **Acceptance Criteria**:
  - [x] PX4 SITL launches with Gazebo world (open arena, no obstacles)
  - [x] ROS 2 bridge operational (mavros or px4_ros_com)
  - [x] Drone arms, takes off, and accepts velocity setpoints via Offboard mode
  - [x] Scripted test: drone moves to a waypoint via velocity commands — `scripts/12_px4_offboard_test.py`

##### M1.2: Target Actor
- **Status**: `partial` (Python behaviors done, Gazebo integration pending)
- **Description**: Create a scripted target entity in Gazebo that moves with configurable behavior. Expose its truth state via ROS 2.
- **Acceptance Criteria**:
  - [ ] Target spawns in Gazebo with visible model
  - [x] Constant-velocity behavior implemented and configurable — `sim/target_behaviors/constant_velocity.py`
  - [x] Waypoint-following behavior implemented — `sim/target_behaviors/waypoint.py`
  - [ ] Target state published on ROS 2 topic (position, velocity)

##### M1.3: Gymnasium Environment Wrapper
- **Status**: `implemented`
- **Description**: Create a Gymnasium-compatible environment wrapping the PX4 + Gazebo + target setup with proper observation/action/reward spaces.
- **Acceptance Criteria**:
  - [x] `InterceptEnv` implements `gymnasium.Env` with correct spaces — `env/intercept_env.py`
  - [x] `reset()` resets drone + target positions, returns initial obs
  - [x] `step(action)` sends velocity command, advances sim, returns (obs, reward, terminated, truncated, info)
  - [x] Observation: 14D vector (self pos/vel + relative target pos/vel + distance + battery) — `env/observation_builder.py`
  - [x] Action: 4D continuous (vx, vy, vz, yaw_rate) with configurable bounds
  - [x] Reward: distance shaping + capture bonus (+100) + crash penalty (-100) — `env/rewards.py` (original + shaped modes)
  - [x] Termination: capture (d<1.5m, v_rel<2.0m/s), crash, timeout, out-of-bounds — `env/termination.py`
  - [ ] Scripted baseline policy achieves some captures (validates env correctness)
  - [x] Physics backend abstraction: `sim/backends/base.py` + `simplified.py` — toggleable via `physics_backend` param

##### M1.4: PPO Training Pipeline
- **Status**: `implemented`
- **Description**: Train PPO with Stable-Baselines3 on the interception env. Achieve >80% capture rate against constant-velocity target.
- **Acceptance Criteria**:
  - [x] `train_ppo.py` script with configurable hyperparameters, `--device`, `--resume`, `--reward-mode`
  - [x] Training runs and converges (reward curve trends upward)
  - [x] >80% capture rate against constant-velocity target after training
  - [x] Model checkpoints saved periodically — `training/callbacks.py`
  - [x] `eval_policy.py` script for evaluation runs with `--device`, `--all-targets`

##### M1.5: Episode Logging & Replay
- **Status**: `implemented`
- **Description**: Implement per-step trajectory logging and 2D replay plots.
- **Acceptance Criteria**:
  - [x] Per-step JSONL log — `replay/logger.py`
  - [x] Per-episode summary: total reward, success/failure, capture time, min distance — CSV output
  - [x] 2D top-down trajectory plot (drone path, target path, capture/collision marker) — `replay/plotter.py`
  - [x] Training metrics logged: episode reward, length, success rate, capture time — `training/callbacks.py`
  - [x] Video export (mp4/gif) — `replay/plotter.py:animate_episode`
  - [x] Batch episode viewer with filter/sort/compare/distributions — `scripts/11_batch_viewer.py`

##### M1.6: Target Behavior Curriculum
- **Status**: `partial`
- **Description**: Add configurable target behaviors beyond constant velocity. Evaluate policy generalization.
- **Acceptance Criteria**:
  - [x] Zigzag target behavior implemented — `sim/target_behaviors/zigzag.py`
  - [x] Target behavior selectable via env config parameter — registry in `intercept_env.py`
  - [x] Cross-evaluation across all target types — `scripts/09_train_full.py`, `scripts/10_eval.py --all-targets`
  - [ ] Optional: curriculum training across behaviors

---

## Phase 2: Tracked Target (Realistic Sensing)

**Goal:** Replace simulator truth with noisy detections + Kalman tracker. Policy must handle uncertainty.

**Status:** `implemented`

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

**Status:** `implemented`

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

**Status:** `implemented`

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

---

## Verification Checklist

_Verified: 2026-04-01 against actual source code._

### Phase 1: Cheated Interception (MVP)

#### M1.1: PX4 SITL + Gazebo Environment Setup

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1.1.1 | PX4 SITL launches with Gazebo world | ❌ Missing | No PX4/Gazebo integration exists. Simplified first-order dynamics used instead (`intercept_env.py:39-43`) |
| 1.1.2 | ROS 2 bridge operational | ❌ Missing | `ros2_nodes/__init__.py` exists but is empty; no ROS 2 bridge code |
| 1.1.3 | Drone arms, takes off, accepts velocity setpoints via Offboard | ❌ Missing | Velocity tracking is simulated via first-order lag (`intercept_env.py:185-188`) — no real PX4 Offboard |
| 1.1.4 | Scripted test: drone moves to waypoint via velocity commands | ❌ Missing | No integration test with PX4/Gazebo |

#### M1.2: Target Actor

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1.2.1 | Target spawns in Gazebo with visible model | ❌ Missing | Target is a pure-Python simulation object, not a Gazebo entity |
| 1.2.2 | Constant-velocity behavior implemented and configurable | ✅ Implemented | `sim/target_behaviors/constant_velocity.py:10-29` — speed/direction configurable |
| 1.2.3 | Waypoint-following behavior implemented | ✅ Implemented | `sim/target_behaviors/waypoint.py:14-52` |
| 1.2.4 | Target state published on ROS 2 topic | ❌ Missing | Target state accessed directly via Python objects, no ROS 2 topics |

#### M1.3: Gymnasium Environment Wrapper

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1.3.1 | `InterceptEnv` implements `gymnasium.Env` with correct spaces | ✅ Implemented | `env/intercept_env.py:38` — extends `gym.Env`, spaces at lines 115-125 |
| 1.3.2 | `reset()` returns initial obs | ✅ Implemented | `env/intercept_env.py:136-175` |
| 1.3.3 | `step(action)` returns (obs, reward, terminated, truncated, info) | ✅ Implemented | `env/intercept_env.py:177-265` |
| 1.3.4 | Observation: 14D vector (self pos/vel + relative target pos/vel + distance + battery) | ✅ Implemented | `env/observation_builder.py:14-21` — 14D vector with all specified components |
| 1.3.5 | Action: 4D continuous (vx, vy, vz, yaw_rate) with configurable bounds | ✅ Implemented | `env/intercept_env.py:121-125` — Box space with configurable max vel/yaw |
| 1.3.6 | Reward: distance shaping + capture bonus (+100) + crash penalty (-100) | ✅ Implemented | `env/rewards.py:8-39` — matches spec formula |
| 1.3.7 | Termination: capture (d<1.5m, v_rel<2.0m/s), crash, timeout, out-of-bounds | ✅ Implemented | `env/termination.py:27-60` — all four conditions present |
| 1.3.8 | Scripted baseline policy achieves some captures | ⚠️ Partial | No scripted baseline test exists; env correctness inferred from structure |

#### M1.4: PPO Training Pipeline

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1.4.1 | `train_ppo.py` script with configurable hyperparameters | ✅ Implemented | `training/train_ppo.py:49-100` — LR, batch, epochs, gamma, seed all configurable via CLI |
| 1.4.2 | Training runs and converges | ⚠️ Partial | Code is correct; no checked-in training logs proving convergence |
| 1.4.3 | >80% capture rate against constant-velocity target | ⚠️ Partial | No evaluation results checked in to confirm this threshold |
| 1.4.4 | Model checkpoints saved periodically | ✅ Implemented | `training/callbacks.py:59-63` — saves every `save_freq` steps |
| 1.4.5 | `eval_policy.py` script for evaluation runs | ✅ Implemented | `training/eval_policy.py:17-104` — full eval with metrics |

#### M1.5: Episode Logging & Replay

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1.5.1 | Per-step JSONL log: t, drone_pos, drone_vel, target_pos, target_vel, action, reward, done | ✅ Implemented | `replay/logger.py:25-35` — `StepRecord` has all fields; written as JSONL at line 106 |
| 1.5.2 | Per-episode summary: total reward, success/failure, capture time, min distance | ✅ Implemented | `replay/logger.py:38-46` — `EpisodeSummary` dataclass + CSV output at line 111 |
| 1.5.3 | 2D top-down trajectory plot (drone path, target path, capture/collision marker) | ✅ Implemented | `replay/plotter.py:14-80` — trajectory + distance + reward subplots |
| 1.5.4 | Training metrics logged: episode reward, length, success rate, capture time | ✅ Implemented | `training/callbacks.py:36-56` — logs avg reward + success rate every 50 eps |
| 1.5.5 | Video export for selected episodes | ✅ Implemented | `replay/plotter.py:83-171` — `animate_episode` supports mp4/gif export |

#### M1.6: Target Behavior Curriculum

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 1.6.1 | Zigzag target behavior implemented | ✅ Implemented | `sim/target_behaviors/zigzag.py:10-47` |
| 1.6.2 | Target behavior selectable via env config parameter | ✅ Implemented | `env/intercept_env.py:25-29,54` — registry + `target_behavior` param |
| 1.6.3 | Policy trained on constant velocity still tested against zigzag | ⚠️ Partial | `eval_policy.py` supports `--target` flag for cross-eval, but no checked-in results |
| 1.6.4 | Optional: curriculum training across behaviors | ❌ Missing | No curriculum scheduler implementation |

### Phase 2: Tracked Target (Realistic Sensing)

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 2.1 | Gaussian noise injected into target detections | ✅ Implemented | `sim/noise.py:25-40` — configurable pos/vel noise + detection dropout |
| 2.2 | Kalman filter tracker node smooths detections | ✅ Implemented | `sim/tracker.py:27-128` — full 6-state Kalman filter with predict/update |
| 2.3 | Observation switches to tracked state + confidence | ✅ Implemented | `env/observation_builder.py:23-25,100-101` — 15D obs with track_confidence |
| 2.4 | Policy must handle track loss gracefully | ✅ Implemented | `sim/tracker.py:74-77` — confidence decays on missed detections; `sim/noise.py:35` — stochastic dropout |

### Phase 3: Obstacle-Aware Interception

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 3.1 | Static obstacles in environment | ✅ Implemented | `sim/obstacles.py:40-53` — random cylindrical obstacles generated per episode |
| 3.2 | Local obstacle perception (sector distances) | ✅ Implemented | `sim/obstacles.py:56-107` — angular sector distance computation |
| 3.3 | Obstacle sector distances in observation | ✅ Implemented | `env/observation_builder.py:64-66,103-104` — appended to obs vector |
| 3.4 | Reward gains near-obstacle and collision penalties | ✅ Implemented | `env/rewards.py:30-31` — proximity penalty; `env/rewards.py:37` — crash penalty |
| 3.5 | Obstacle collision detection | ✅ Implemented | `sim/obstacles.py:110-124` — collision check with drone radius |

### Phase 4: Prediction-Aware Interception

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| 4.1 | Constant-velocity predictor node | ✅ Implemented | `sim/predictor.py:24-49` — linear extrapolation at configurable horizons |
| 4.2 | Predicted positions at t+0.5s and t+1.0s in observation | ✅ Implemented | `sim/predictor.py:21` — default horizons `(0.5, 1.0)`; `env/observation_builder.py:67-71,104-108` |
| 4.3 | Policy can train with prediction | ✅ Implemented | `training/train_ppo.py:123-126` — `--prediction` flag wires predictor_config through |
| 4.4 | Optional: learned predictor (RNN/transformer) | ❌ Missing | Only constant-velocity predictor exists |

### Cross-Cutting

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| C.1 | Unit tests | ❌ Missing | No test files found anywhere in the project |
| C.2 | `pip install -e ".[dev]"` works | ⚠️ Partial | Not verified in this check |

---

### Summary

| Category | Implemented | Partial | Missing | Total |
|----------|:-----------:|:-------:|:-------:|:-----:|
| M1.1 PX4/Gazebo Setup | 0 | 0 | 4 | 4 |
| M1.2 Target Actor | 2 | 0 | 2 | 4 |
| M1.3 Gymnasium Env | 7 | 1 | 0 | 8 |
| M1.4 PPO Training | 3 | 2 | 0 | 5 |
| M1.5 Logging & Replay | 5 | 0 | 0 | 5 |
| M1.6 Target Curriculum | 2 | 1 | 1 | 4 |
| Phase 2 Tracked Target | 4 | 0 | 0 | 4 |
| Phase 3 Obstacles | 5 | 0 | 0 | 5 |
| Phase 4 Prediction | 3 | 0 | 1 | 4 |
| Cross-Cutting | 0 | 1 | 1 | 2 |
| **Total** | **31** | **5** | **9** | **45** |

**31/45 implemented, 5 partial, 9 missing**

### Key Findings

1. **No Gazebo/PX4/ROS 2 integration** — The entire M1.1 milestone is unimplemented. The spec calls for PX4 SITL + Gazebo as the dynamics backend, but the codebase uses simplified first-order Python dynamics instead. This is a deliberate design decision (noted in code comments) to allow fast iteration, with the intent to swap in PX4 later.

2. **No tests** — Zero unit or integration tests exist in the project.

3. **Core RL loop is fully functional** — The Gymnasium env, observation/action/reward design, PPO training pipeline, episode logging, and replay are all implemented and match the spec.

4. **Phases 2-4 are implemented as simplified Python simulations** — Tracker, obstacles, and predictor all work but as lightweight Python modules, not as Gazebo plugins or ROS 2 nodes.
