# SPEC025 - Drone Interception RL

## 1. Purpose

Build a reinforcement learning environment for drone interception — training a decision policy that sends velocity setpoints to an autopilot to intercept a moving target in simulation.

This spec covers the **full progression** from a minimal "cheated" environment to prediction-aware obstacle-avoiding interception, but **Phase 1 is the priority**: get a working RL loop with simulator truth.

### 1.1 Problem Statement

Drone interception is a pursuit-evasion problem. The interceptor must close distance to a moving target efficiently. Rather than training end-to-end from sensors to motors, the RL agent operates at the **decision/guidance layer** — receiving estimated world state and outputting velocity commands that a flight controller (PX4) executes.

### 1.2 Scope

**In scope (full project):**
- Gymnasium-compatible interception environment
- PPO training with Stable-Baselines3
- Scripted target behaviors (constant velocity -> evasive)
- Episode logging and replay
- Progressive complexity: truth -> tracking -> obstacles -> prediction

**Out of scope:**
- Real hardware / sim-to-real transfer
- Multi-agent swarm interception
- End-to-end camera-to-motor RL
- Raw motor control

## 2. Stakeholders & Actors

### 2.1 Stakeholders

| ID | Stakeholder | Type | Need |
|----|-------------|------|------|
| SN-01 | RL Researcher | Direct | Train and evaluate interception policies with fast iteration |
| SN-02 | Robotics Engineer | Direct | Validate policies in realistic sim before hardware |

### 2.2 Actors

| Actor | Description |
|-------|-------------|
| RL Researcher | Configures environment, trains policies, analyzes episodes |
| Trained Policy | Autonomous agent producing velocity commands from observations |
| Target Actor | Scripted moving entity the interceptor must catch |
| PX4 Autopilot | Flight controller executing velocity setpoints |

## 3. User Stories

### US-01: Train basic interception policy (SN-01)
**As** an RL researcher, **I want** to train a PPO policy against a constant-velocity target using simulator truth state, **so that** I have a working baseline before adding complexity.

**Acceptance criteria:**
- Gymnasium env with `reset()` / `step(action)` loop
- Observation: self position/velocity + relative target position/velocity + distance
- Action: 3D velocity command + yaw rate (continuous)
- Reward: distance shaping + capture bonus + crash penalty
- Capture condition: distance < 1.5m AND relative speed < 2.0 m/s
- Episode terminates on capture, crash, timeout, or out-of-bounds
- PPO converges to >80% capture rate against constant-velocity target

### US-02: Log and replay episodes (SN-01)
**As** an RL researcher, **I want** per-step trajectory logs and episode summaries, **so that** I can debug policy behavior and compare runs.

**Acceptance criteria:**
- Per-step log: timestamp, drone pos/vel, target pos/vel, action, reward, done
- Per-episode summary: total reward, success/failure, capture time, min distance
- Replay tool: 2D top-down trajectory plot (drone path, target path, capture/collision marker)

### US-03: Progressive target difficulty (SN-01)
**As** an RL researcher, **I want** configurable target behaviors (constant velocity -> waypoint -> zigzag -> random evasive), **so that** I can curriculum-train robust policies.

**Acceptance criteria:**
- Target behavior selectable via env config
- At minimum: constant velocity, waypoint-following, zigzag

### US-04: Tracked target with noise (SN-02) [Phase 2]
**As** a robotics engineer, **I want** the environment to support noisy detections + Kalman tracking instead of truth state, **so that** I can evaluate policy robustness under realistic sensing.

### US-05: Obstacle-aware interception (SN-02) [Phase 3]
**As** a robotics engineer, **I want** obstacles in the environment and local obstacle perception in the observation, **so that** the policy learns safe pursuit.

### US-06: Prediction-aware interception (SN-02) [Phase 4]
**As** a robotics engineer, **I want** target prediction (future position estimates) in the observation, **so that** the policy can perform lead pursuit.

## 4. Observation, Action, Reward Reference

### Observation (Phase 1 - compact vector)

```python
obs = np.array([
    self_x, self_y, self_z,           # interceptor position
    self_vx, self_vy, self_vz,        # interceptor velocity
    target_dx, target_dy, target_dz,  # relative target position
    target_dvx, target_dvy, target_dvz, # relative target velocity
    distance_to_target,               # scalar distance
    battery_level,                    # normalized 0-1
])  # 14-dimensional
```

### Action (velocity setpoints)

```python
action = np.array([
    vx_cmd, vy_cmd, vz_cmd,  # desired velocity (m/s)
    yaw_rate_cmd              # desired yaw rate (rad/s)
])  # 4-dimensional, continuous
```

### Reward

```python
reward = (
    -0.1 * distance_to_target
    - 0.01 * control_effort
    - 0.05 * collision_risk
)
if captured:
    reward += 100.0
if crashed:
    reward -= 100.0
```

### Capture Condition

```python
captured = distance_to_target < 1.5 and relative_speed < 2.0
```

## 5. Blueprint Reference

The full technical blueprint is stored at:
**Source:** `drone_interception_rl_blueprint.md` (see delivery strategy for copy location)

Key sections for future phases:
- **Sections 5-6**: Sensor stack and detection methods (Phase 2-3)
- **Section 7**: Pipeline pieces with data contracts per layer (Phase 2-4)
- **Section 10**: Implementation phases with observation schemas (Phase 2-4)
- **Section 13**: Project structure reference
- **Section 14**: Target behavior progression (Phase 1 curriculum)
- **Sections 19-24**: Episode observability and visualization (all phases)
