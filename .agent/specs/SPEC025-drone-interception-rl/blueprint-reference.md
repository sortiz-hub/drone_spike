# Drone Interception & RL Blueprint

## 1. Big picture
Drone autonomy is best understood as a **pipeline of pieces**, not as “one magic RL agent that does everything”.

High-level stack:

```text
[Simulator / Real Sensors]
        ↓
[State Estimation]
        ↓
[Perception]
        ↓
[Tracking]
        ↓
[Prediction]
        ↓
[Decision / Policy]
        ↓
[Guidance / Setpoints]
        ↓
[Autopilot / Low-level Control]
        ↓
[Drone Motion]
```

Core principle:
> You are not training “a drone”. You are training **a decision policy over an estimated world state**.

---

## 2. Underlying tech for drones

Typical drone-control technology stack:

- **Autopilot / Flight controller**
  - PX4
  - ArduPilot
- **State estimation**
  - EKF / sensor fusion
- **Low-level control**
  - PID / cascaded controllers
- **Communication / middleware**
  - MAVLink
  - MAVSDK
  - ROS 2
- **Autonomy layer**
  - perception, planning, tracking, RL, etc.

Recommended practical architecture:

- Let **PX4 / ArduPilot** handle stabilization
- Put **RL above it**, producing:
  - waypoint commands
  - velocity commands
  - yaw-rate or guidance commands

Avoid starting with:
- raw motor control
- end-to-end camera-to-motor RL

---

## 3. What can you train agents for?

### 3.1 Navigation agent
Train the drone to go from A → B robustly.

Examples:
- waypoint reaching
- corridor following
- gate passing
- return home

Typical observation:
- position
- velocity
- heading
- relative target position

Typical action:
- desired velocity
- yaw rate
- waypoint offset

Typical reward:
- get closer to goal
- penalize collisions
- penalize oscillation
- bonus for success

Difficulty: **Easy–Medium**

---

### 3.2 Obstacle avoidance agent
Train the drone to avoid obstacles while still progressing.

Examples:
- indoor navigation
- cluttered spaces
- emergency rerouting

Observation:
- depth / LiDAR / occupancy
- relative goal
- velocity

Action:
- local velocity commands
- heading changes

Reward:
- progress
- collision penalty
- near-obstacle penalty
- smoothness reward

Difficulty: **Medium**

---

### 3.3 Target tracking / following agent
Keep a moving target in view and follow it.

Examples:
- follow a person
- follow a vehicle
- follow another drone

Observation:
- camera detections / bounding boxes
- relative target pose
- drone state

Action:
- velocity
- yaw
- altitude
- gimbal angle (if any)

Reward:
- target centered
- target visible
- low jitter
- penalty if target lost

Difficulty: **Medium–High**

---

### 3.4 Inspection / coverage / mapping agent
Teach the drone to inspect or cover an area efficiently.

Examples:
- solar panels
- wind turbines
- facades / roofs
- power lines
- building interiors

Observation:
- pose
- explored/unexplored area
- battery
- known or partial map

Action:
- next viewpoint
- route choice
- camera angle

Reward:
- useful new coverage
- good viewpoints
- penalty for redundant revisits
- penalty for energy use

Difficulty: **Medium–High**

---

### 3.5 Energy-efficient / battery-aware flight agent
Optimize mission success while minimizing energy.

Examples:
- efficient route choice
- exploit wind
- avoid aggressive acceleration
- adapt to low battery

Observation:
- battery
- speed
- altitude
- wind estimate
- route remaining

Action:
- speed profile
- climb/descent strategy
- path choice

Reward:
- mission completion
- low energy use
- smoothness

Difficulty: **Medium**

---

### 3.6 Multi-drone coordination / swarm agent
Train multiple drones to cooperate.

Examples:
- search and rescue
- area coverage
- formation flying
- cooperative mapping

Observation:
- own state
- nearby drone states
- assigned sectors / shared map

Action:
- movement
- formation offset
- task choice
- communication decisions

Reward:
- team coverage
- target found
- avoid collisions
- avoid duplicated work

Difficulty: **High**

---

## 4. Interception: what it is
Interception is a **pursuit–evasion** problem.

At a high level:
- one drone = interceptor
- one target = evader

State usually includes:
- relative position
- relative velocity
- own velocity
- optionally target intent / predicted path

Goal:
- minimize distance or time-to-capture
- optionally maintain safety / avoid obstacles

Interception scenarios:

### 4.1 Basic chase
- single drone vs moving target
- target can be scripted
- easiest RL entry point

### 4.2 Lead pursuit / predictive interception
- chase where the target **will be**, not where it **is**
- much better against fast targets

### 4.3 Interception with obstacles
- now it becomes a real robotics problem
- combine pursuit + obstacle avoidance

### 4.4 Multi-agent interception
- multiple drones cooperate to intercept one target

### 4.5 Adversarial interception
- train both interceptor and evader
- co-evolving strategies

Practical recommendation:
> Start with **single-drone interception against a scripted target**.

---

## 5. How drones detect the environment and other drones

This is split into two related but different problems:

1. **Environment sensing**
2. **Other-drone detection**

### 5.1 Environment sensing

#### A. Cameras (mono / stereo / fisheye)
Used for:
- obstacle detection
- visual odometry / SLAM
- semantic understanding

Pros:
- cheap
- lightweight
- rich information

Cons:
- weak in darkness, glare, rain, fog
- thin objects can be difficult

#### B. Depth cameras / ToF / stereo depth
Used for:
- short- to mid-range obstacle awareness
- local planning

Pros:
- direct distance estimate

Cons:
- sunlight / range limitations depending on sensor

#### C. LiDAR
Used for:
- 3D geometry
- mapping
- local obstacle avoidance

Pros:
- strong geometric understanding
- robust in low light

Cons:
- cost
- payload
- power

#### D. Radar
Used for:
- poor visibility conditions
- moving object detection
- airspace awareness

Pros:
- robust in fog/dust/night
- useful for moving targets

Cons:
- lower spatial resolution
- classification harder

#### E. Ultrasonic / IR
Used for:
- close-range safety
- landing
- near-surface awareness

Pros:
- cheap and simple

Cons:
- only useful at short range

---

### 5.2 Detecting other drones

#### A. Vision-based detection
Typical pipeline:
1. detect drone in image
2. track over time
3. estimate relative motion

Common models:
- YOLO-like detector
- segmentation-based methods
- transformer detectors

Pros:
- classification possible
- good in controlled settings

Cons:
- small fast drones are hard
- lighting / background / occlusion issues

#### B. Radar-based drone detection
Especially relevant for airspace and counter-UAS style awareness.

Pros:
- range
- poor visibility robustness

Cons:
- cost / complexity
- small drone detectability can be challenging

#### C. LiDAR-based drone detection
Possible for near / mid-range geometric localization.

#### D. RF-based detection
Detects the **radio emissions** of another drone.

Can reveal:
- control link presence
- telemetry presence
- possible vendor/protocol family

Pros:
- useful early warning

Cons:
- only works if the target emits RF

#### E. Cooperative detection (ADS-B / Remote ID)
If the target broadcasts its position/identity.

Pros:
- elegant and low compute

Cons:
- only works if the target is cooperative

---

## 6. Common real-world strategy
Real systems usually do **not** rely on one magic sensor.

Typical autonomy stack:

### Layer 1 — State estimation
Question answered:
> Where am I?

Output:
- position
- velocity
- orientation

### Layer 2 — Perception
Question answered:
> What is around me?

Output:
- obstacles
- free space
- detections
- map / occupancy

### Layer 3 — Tracking
Question answered:
> Which detection is the same object over time?

Output:
- smoothed target state
- target velocity
- target continuity

Typical methods:
- Kalman filter
- EKF / UKF
- particle filter
- SORT / DeepSORT-style tracking

### Layer 4 — Prediction
Question answered:
> Where will it probably be next?

Output:
- future target estimate

Typical methods:
- constant velocity
- constant acceleration
- learned predictors (RNN / transformer)

### Layer 5 — Planning / decision
Question answered:
> What should I do now?

Output:
- intercept / avoid / follow / orbit / hold / retreat

Important principle:
> Use **classical estimation + sensor fusion + tracking first**, and only then add RL where it adds leverage.

---

## 7. The pieces and what you get in each step

### Step 1 — Simulator / World
What it is:
- Gazebo / AirSim / Isaac Sim / etc.

Input:
- commands, target behavior, wind, obstacles

Output:
- world state, collisions, sensors, target state

What you get:
- the environment in which to test everything

RL here?
- No

---

### Step 2 — State Estimation
What it is:
- self-state estimation

Input:
- IMU, GPS, barometer, visual odometry, etc.

Output:
```python
{
  "position": [x, y, z],
  "velocity": [vx, vy, vz],
  "orientation": [roll, pitch, yaw],
  "angular_rates": [p, q, r]
}
```

What you get:
- reliable self-state

RL here?
- Usually no

Practical note:
> If you use PX4, you largely get this for free.

---

### Step 3 — Perception
What it is:
- environment and target sensing

Input:
- RGB, depth, LiDAR, radar, RF, simulator truth

Output examples:
```python
{
  "obstacles": [...],
  "free_space": [...],
  "depth_map": ...,
  "nearest_obstacle_distance": 2.3
}
```

Or target-related:
```python
{
  "detected": True,
  "bearing": 18.5,
  "distance": 12.4,
  "bbox": [x1, y1, x2, y2]
}
```

What you get:
- awareness of walls, obstacles, or another drone

RL here?
- Not first. Start by cheating with simulator truth.

---

### Step 4 — Tracking
What it is:
- smoothing detections over time

Input:
- target detections over time

Output:
```python
{
  "position": [x, y, z],
  "velocity": [vx, vy, vz],
  "confidence": 0.95
}
```

What you get:
- stable relative target estimate
- target velocity estimate

RL here?
- Usually no

Why it matters:
> Without tracking, the policy chases noisy ghosts.

---

### Step 5 — Prediction
What it is:
- estimate where the target will be next

Input:
```python
{
  "position": [x, y, z],
  "velocity": [vx, vy, vz]
}
```

Output:
```python
{
  "t+0.5s": [x1, y1, z1],
  "t+1.0s": [x2, y2, z2]
}
```

What you get:
- lead pursuit
- smarter interception

RL here?
- Optional, but not required first

Recommendation:
> Start with constant velocity prediction.

---

### Step 6 — Decision / Policy
What it is:
- the main “brain”

Input example:
```python
obs = {
    "self_state": drone_state,
    "target_state": tracked_target,
    "predicted_target": predicted_target,
    "local_obstacles": occupancy_grid,
    "battery": battery_level
}
```

Output example:
```python
action = {
    "desired_velocity": [vx, vy, vz],
    "desired_yaw_rate": yaw_rate
}
```

What you get:
- learned pursuit / avoidance / interception strategy

RL here?
- **Yes — this is the main RL slot**

---

### Step 7 — Guidance / Setpoint Generation
What it is:
- convert policy decisions into autopilot setpoints

Input:
```python
desired_velocity = [1.2, -0.5, 0.3]
```

Output:
- velocity setpoints
- position setpoints
- yaw-rate setpoints

What you get:
- clean bridge to PX4 / ArduPilot

RL here?
- No

---

### Step 8 — Autopilot / Low-level control
What it is:
- stabilization and actual actuation

Input:
```python
target_velocity = [vx, vy, vz]
target_yaw = yaw
```

Output:
- motor-level actuation

What you get:
- stable flight

RL here?
- Do not start here

Important recommendation:
> Reuse PX4 / ArduPilot for this layer.

---

## 8. Clean staged roadmap

### Phase A — Simplest interception
Pieces:
- simulator
- PX4
- exact target state
- RL policy

What you get:
- working interception benchmark

---

### Phase B — Realistic target sensing
Pieces:
- detector
- tracker
- RL policy

What you get:
- robust pursuit under uncertainty

---

### Phase C — Safe interception
Pieces:
- obstacle perception
- target tracking
- prediction
- RL policy

What you get:
- realistic autonomy scenario

---

## 9. Buildable implementation blueprint

### 9.1 Recommended stack
Use:
- **Linux or WSL2**
- **PX4 SITL**
- **Gazebo**
- **ROS 2**
- **Python Gymnasium wrapper**
- **Stable-Baselines3 PPO**

Why:
- practical
- realistic
- modular
- strong path to sim-to-real

---

### 9.2 Runtime architecture

```text
Gazebo world
   ↓
PX4 SITL
   ↔
ROS 2 control/perception nodes
   ↔
Gymnasium env wrapper
   ↔
SB3 policy (training) or trained policy (inference)
```

Logical flow:

```text
Sensors / simulator truth
   ↓
Perception
   ↓
Tracking
   ↓
Prediction
   ↓
RL decision policy
   ↓
Setpoint generator
   ↓
PX4 Offboard
```

---

### 9.3 What runs where

#### Inside PX4
- stabilization
- low-level control
- state estimation
- arming / mode logic

#### In ROS 2 / Python
- target simulator or target interface
- perception node
- tracking node
- predictor node
- policy inference node
- reward / episode manager
- logging / metadata

#### In Gymnasium env
- `reset()`
- `step(action)`
- reward calculation
- termination / truncation
- observation assembly

#### In SB3
- policy training
- rollouts
- checkpoints

---

## 10. Minimal phases in implementation terms

### Phase 1 — Cheated interception
Use simulator truth for target state.

Observation:
```python
obs = {
    "self_pos": [x, y, z],
    "self_vel": [vx, vy, vz],
    "target_rel_pos": [dx, dy, dz],
    "target_rel_vel": [dvx, dvy, dvz],
}
```

Action:
```python
action = {
    "desired_velocity": [vx_cmd, vy_cmd, vz_cmd],
    "yaw_rate": yaw_rate_cmd
}
```

What you get:
- fastest useful benchmark

---

### Phase 2 — Tracked target
Replace simulator truth with noisy detections + tracker.

Observation:
```python
obs = {
    "self_state": ...,
    "tracked_target_pos": ...,
    "tracked_target_vel": ...,
    "track_confidence": ...,
}
```

What you get:
- realistic pursuit under uncertainty

---

### Phase 3 — Obstacle-aware interception
Add local obstacle perception.

Observation:
```python
obs = {
    "self_state": ...,
    "tracked_target": ...,
    "local_obstacles": ...,
}
```

What you get:
- safe pursuit and avoidance

---

### Phase 4 — Prediction-aware interception
Add target prediction.

Observation:
```python
obs = {
    "self_state": ...,
    "tracked_target": ...,
    "predicted_target_t05": ...,
    "predicted_target_t10": ...,
    "local_obstacles": ...,
}
```

What you get:
- smarter lead pursuit / interception

---

## 11. Recommended observation, action, reward

### Observation
Start compact.

Example vector:
```python
obs = np.array([
    self_x, self_y, self_z,
    self_vx, self_vy, self_vz,
    target_dx, target_dy, target_dz,
    target_dvx, target_dvy, target_dvz,
    distance_to_target,
    battery_level,
])
```

Later add:
- obstacle sector distances
- target confidence
- predicted intercept points

---

### Action
Use **velocity setpoints**, not motor outputs.

Example:
```python
action = np.array([
    vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd
])
```

---

### Reward
Starter reward:

```python
reward = (
    -0.1 * distance_to_target
    -0.01 * control_effort
    -0.05 * collision_risk
)

if captured:
    reward += 100.0
if crashed:
    reward -= 100.0
```

Possible improvements:
- progress reward
- smoothness penalty
- time penalty
- safe-distance shaping

---

## 12. Capture condition
Define success clearly.

Example:
```python
captured = distance_to_target < 1.5 and relative_speed < 2.0
```

What you get:
- better reward signal
- more realistic interception metric

---

## 13. Suggested project structure

```text
drone_intercept/
  sim/
    world_config/
    target_behaviors/
  ros2_nodes/
    vehicle_state_node.py
    target_provider_node.py
    tracker_node.py
    predictor_node.py
    offboard_command_node.py
  env/
    intercept_env.py
    rewards.py
    termination.py
    observation_builder.py
  training/
    train_ppo.py
    eval_policy.py
    callbacks.py
  models/
  logs/
```

---

## 14. Target behavior progression
Start scripted before adversarial.

Recommended order:
1. constant velocity target
2. waypoint-following target
3. zig-zag target
4. random evasive target
5. obstacle-aware target

What you get:
- curriculum
- easier debugging
- better generalization

---

## 15. Best first algorithm
Start with **PPO**.

Why:
- good for continuous control
- simple setup in SB3
- practical baseline

Later compare with:
- SAC

---

## 16. What not to do first
Avoid starting with:
- raw camera → end-to-end RL
- raw motor outputs
- multi-agent swarm interception
- full sim-to-real from day one

Reason:
- slower progress
- harder debugging
- many false failures

---

## 17. Best first milestone

Recommended first milestone:

> A trained policy that sends **velocity setpoints** to PX4 Offboard in simulation and can **intercept a scripted moving target in open space**.

This is already:
- meaningful
- technically credible
- extensible

---

## 18. Exact next build order

1. Bring up **PX4 SITL + Gazebo**
2. Run a **ROS 2 Offboard** example
3. Create a **target actor** with known truth state
4. Write a **Gymnasium env** around one episode
5. Use a **scripted baseline policy** first
6. Train **PPO**
7. Add **noise**
8. Add **tracker**
9. Add **obstacles**
10. Add **predictor**

---

## 19. How to see the episodes
Seeing the episodes properly is critical. You usually want **four views**:

- **World view** → what happened physically
- **Agent view** → what the policy saw
- **Telemetry view** → numbers over time
- **Replay view** → review later

---

### 19.1 Watch live in the simulator
Use Gazebo / AirSim live view.

What you get:
- immediate sanity check
- qualitative understanding

---

### 19.2 Record videos of episodes
Save videos periodically.

Sources can be:
- simulator camera
- chase camera
- top-down camera
- env render frames

What you get:
- human-readable replay
- easy comparison across runs
- demo material

---

### 19.3 Store trajectory data and replay it
This is one of the most valuable debugging tools.

At each timestep, log:

```python
step_log = {
    "t": t,
    "drone_pos": [x, y, z],
    "drone_vel": [vx, vy, vz],
    "target_pos": [tx, ty, tz],
    "target_vel": [tvx, tvy, tvz],
    "action": [ax, ay, az, yaw_rate],
    "reward": reward,
    "done": done,
    "distance_to_target": distance,
}
```

Store in:
- JSONL
- CSV
- Parquet
- experiment tracker

What you get:
- replay later
- failure analysis
- reward debugging

---

### 19.4 Plot episode trajectories
Very useful for interception.

Plot:
- drone path
- target path
- capture point
- collision point
- reward over time

What you get:
- geometric understanding of interception behavior

---

### 19.5 Watch what the agent “saw”
Important to distinguish:
- bad policy
vs
- bad observation

Possible overlays:
- distance to target
- relative angle
- obstacle sectors
- target confidence
- RGB / depth / detection boxes / tracker markers

What you get:
- much better debugging of perception-aware RL

---

### 19.6 ROS-based visualization
If using ROS 2, publish and visualize:
- drone pose
- target pose
- predicted path
- obstacle points
- commanded velocity vectors

What you get:
- more engineering-grade robotics debugging

---

### 19.7 Training metrics dashboard
Log:
- episode reward
- episode length
- success rate
- capture time
- collision rate
- minimum distance to target

What you get:
- real learning signal over time

---

## 20. Recommended episode observability stack

### A. Live simulator
What you get:
- quick visual sanity check

### B. Per-step trajectory logs
What you get:
- structured replay and debugging

### C. 2D top-down episode plot
What you get:
- interception geometry clarity

### D. Video export for selected episodes
What you get:
- qualitative comparison and demos

### E. Training metrics dashboard
What you get:
- learning signal and trend tracking

---

## 21. Suggested logging cadence

### Every step
Log:
- state
- action
- reward
- target state
- done flag

### Every episode
Save:
- total reward
- success/failure
- capture time
- minimum distance to target
- collision flag

### Every 50 or 100 episodes
Also save:
- video
- trajectory plot

---

## 22. Five things you should absolutely visualize for interception
If you only visualize five things, use these:

1. drone path
2. target path
3. distance to target over time
4. policy action over time
5. capture / collision marker

---

## 23. Minimal replay format
One file per episode could look like this:

```json
{
  "episode_id": 42,
  "success": true,
  "capture_time": 18.4,
  "steps": [
    {
      "t": 0.0,
      "drone_pos": [0.0, 0.0, 2.0],
      "target_pos": [10.0, 3.0, 2.0],
      "action": [1.2, 0.4, 0.0, 0.1],
      "reward": -1.1
    },
    {
      "t": 0.1,
      "drone_pos": [0.12, 0.04, 2.0],
      "target_pos": [9.95, 3.02, 2.0],
      "action": [1.3, 0.4, 0.0, 0.08],
      "reward": -1.05
    }
  ]
}
```

This is enough to build:
- plots
- replay tools
- episode dashboards
- failure analysis

---

## 24. Best practical recommendation
If you only do **one serious thing** for episode visibility, do this:

> **Log every step and build a replay/plot tool.**

This is more valuable long-term than just watching Gazebo live.

Because later you will want to ask questions like:
- show me the first successful interception
- show me all collisions with obstacles
- show me episodes where reward was high but capture failed

That becomes possible once your episodes are structured and replayable.

---

## 25. Final recommended MVP path

### MVP Goal
Train a policy that:
- receives compact state observations
- outputs velocity setpoints
- runs over PX4 Offboard in simulation
- intercepts a scripted moving target
- logs and replays episodes properly

### MVP Pieces
- PX4 SITL
- Gazebo
- ROS 2 bridge
- Gymnasium env
- PPO trainer
- episode logger
- top-down replay / plotter

### Why this is the right first build
Because it gives you:
- a real robotics RL scenario
- a clean experimental loop
- something extensible toward:
  - noisy perception
  - obstacle avoidance
  - prediction
  - multi-agent extensions
  - eventually sim-to-real

---

## 26. Short blunt summary
If you want the shortest possible truth:

- **Use PX4 for flight control**
- **Use ROS 2 / Python for orchestration**
- **Use Gymnasium + SB3 for RL**
- **Train at the guidance / decision layer, not motor layer**
- **Start with simulator truth**
- **Then add tracking, obstacles, and prediction**
- **Log every episode properly**

That is the clean, realistic, and high-leverage way to build drone interception RL.

