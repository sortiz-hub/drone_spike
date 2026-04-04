# SOP: Validation Scripts

## Overview

The `scripts/` folder contains numbered validation scripts that test progressively deeper layers of the system. Run them in order — each depends on the previous layers being functional.

## Quick Reference

```bash
# Run all quick validations (01-06, no training required)
python scripts/run_all.py --quick

# Run everything including training + eval
python scripts/run_all.py
```

## Script Catalog

### Layer 1: Environment Validation (no model needed)

| Script | What it Tests | Expected Output |
|--------|--------------|-----------------|
| `01_env_smoke_test.py` | Basic `reset()` + `step()` | Obs shape (14,), reward, distance |
| `02_target_behaviors.py` | All 3 target types work | PASS for constant_velocity, waypoint, zigzag |
| `03_tracked_sensing.py` | Phase 2 — Kalman tracker | 15D obs, track_confidence value |
| `04_obstacles.py` | Phase 3 — sector distances | 22D obs (14+8), sector distance array |
| `05_prediction.py` | Phase 4 — lead pursuit | 20D obs (14+6), predicted positions |
| `06_gpu_check.py` | GPU availability | GPU name + VRAM, or CPU-only warning |

### Layer 2: Training Pipeline (creates a model)

| Script | What it Tests | Expected Output |
|--------|--------------|-----------------|
| `07_short_training.py` | PPO trains without crashing (50k steps) | Model saved to `models/` |
| `08_eval_and_plot.py` | Evaluation + plotting pipeline | Success rate, trajectory plots |
| `09_train_full.py` | Full training + cross-evaluation | Summary table: success rate per target type |

### Layer 3: Analysis Tools (requires eval logs)

| Script | What it Tests | Expected Output |
|--------|--------------|-----------------|
| `10_eval.py` | Standalone evaluation tool | Metrics + plots for any model/target combo |
| `11_batch_viewer.py` | Episode browser + analysis | Tables, distributions, comparison plots |

### Layer 4: Gazebo Integration (requires Docker container)

| Script | What it Tests | Expected Output |
|--------|--------------|-----------------|
| `12_px4_offboard_test.py` | PX4 arm, takeoff, velocity control | Drone moves in Gazebo 3D window |

---

## Detailed Usage

### 01 — Environment Smoke Test

```bash
python scripts/01_env_smoke_test.py
```

Creates an `InterceptEnv`, calls `reset()` and `step()` once, prints observation shape and reward. If this fails, the package isn't installed correctly.

### 02 — Target Behaviors

```bash
python scripts/02_target_behaviors.py
```

Runs 10 steps each with `constant_velocity`, `waypoint`, and `zigzag` targets. Validates that all target types initialize and step without errors.

### 03 — Tracked Sensing (Phase 2)

```bash
python scripts/03_tracked_sensing.py
```

Creates env with `sensing_mode="tracked"`. Validates:
- Observation is 15D (14 base + 1 track_confidence)
- Track confidence is populated after 20 steps

### 04 — Obstacles (Phase 3)

```bash
python scripts/04_obstacles.py
```

Creates env with `ObstacleConfig()`. Validates:
- Observation is 22D (14 base + 8 sector distances)
- Sector distances appear in the `info` dict

### 05 — Prediction (Phase 4)

```bash
python scripts/05_prediction.py
```

Creates env with `PredictorConfig()`. Validates:
- Observation is 20D (14 base + 2 predictions x 3D)
- Predicted positions appear in the `info` dict

### 06 — GPU Check

```bash
python scripts/06_gpu_check.py
```

Reports GPU name, CUDA version, and VRAM. Passes with a warning if no GPU detected. Training uses CPU by default, so GPU is optional.

### 07 — Short Training

```bash
python scripts/07_short_training.py
python scripts/07_short_training.py --device cuda  # force GPU (not recommended for MLP policy)
```

Runs a quick 50k-step PPO training. Validates the full training pipeline: env creation, PPO initialization, callback logging, model checkpointing, and final save.

### 08 — Eval and Plot

```bash
python scripts/08_eval_and_plot.py
```

**Requires**: A trained model at `models/ppo_intercept_final.zip` (from script 07 or 09).

Loads the model, runs 20 evaluation episodes, prints success rate, and generates trajectory plots to `logs/eval/`.

### 09 — Full Training + Cross-Evaluation

```bash
# Default: 500k steps, constant velocity, then eval vs all 3 targets
python scripts/09_train_full.py

# More training
python scripts/09_train_full.py --timesteps 1000000

# Resume from saved model (auto-detects models/ppo_intercept_final.zip)
python scripts/09_train_full.py --timesteps 500000

# Train against zigzag
python scripts/09_train_full.py --target zigzag

# Use shaped reward (default) or original
python scripts/09_train_full.py --reward-mode shaped
python scripts/09_train_full.py --reward-mode original

# Skip evaluation (training only)
python scripts/09_train_full.py --skip-eval
```

**Auto-resume**: If `models/ppo_intercept_final.zip` exists, training resumes from it automatically. No flag needed.

**Output**: After training, evaluates the model against all 3 target types (constant_velocity, waypoint, zigzag) and prints a summary table with success rate, avg reward, and capture time.

### 10 — Standalone Evaluation

```bash
# Auto-detect model, evaluate vs constant velocity
python scripts/10_eval.py

# Evaluate vs zigzag at higher speed
python scripts/10_eval.py --target zigzag --target-speed 8

# Evaluate vs all 3 targets
python scripts/10_eval.py --all-targets

# Use a specific checkpoint
python scripts/10_eval.py --model models/ppo_intercept_250000.zip

# More episodes, no plots
python scripts/10_eval.py --episodes 500 --no-plot
```

**Auto-detect**: If `--model` is not specified, uses `models/ppo_intercept_final.zip`.

### 11 — Batch Episode Viewer

```bash
# Auto-detect latest eval directory
python scripts/11_batch_viewer.py

# Browse a specific directory
python scripts/11_batch_viewer.py logs/eval_zigzag

# Filter and sort
python scripts/11_batch_viewer.py logs/eval_zigzag --filter success --sort reward --top 10
python scripts/11_batch_viewer.py logs/eval_zigzag --filter fail --sort distance

# Visualizations
python scripts/11_batch_viewer.py logs/eval_zigzag --dist                    # histograms
python scripts/11_batch_viewer.py logs/eval_zigzag --compare 0 3 7           # overlay episodes
python scripts/11_batch_viewer.py logs/eval_zigzag --plot-all --out plots/   # save all plots
```

**Features**:
- Stats summary: success rate, reward distribution, failure breakdown
- Episode table: sortable by reward, distance, steps, capture time
- Filters: `--filter success` or `--filter fail`
- Distributions: `--dist` shows histograms of reward, min distance, capture time
- Comparison: `--compare ID1 ID2 ...` overlays selected trajectories
- Batch export: `--plot-all --out DIR` saves every episode plot

### 12 — PX4 Offboard Test (Docker only)

**Prerequisites**: PX4 SITL + MAVROS running in Docker container (see [Docker Testing](#docker-testing) below).

```bash
# Inside the container (terminal 3):
source /opt/ros/humble/setup.bash
cd /workspace/drone_spike
python3 scripts/12_px4_offboard_test.py
```

**Sequence**: Pre-stream setpoints → OFFBOARD mode → Arm → Takeoff (vz=2) → Forward (vx=2) → Right (vy=2) → Hover → Land (vz=-1) → Disarm

Watch the Gazebo 3D window to see the drone fly an L-shaped path.

---

## Docker Testing

### Setup

1. **Install VcXsrv** on Windows (X11 server for Gazebo GUI)
   - Launch XLaunch: Multiple windows > Start no client > Disable access control

2. **Build and start container**:
   ```bash
   cd docker
   docker compose build        # ~15-30 min first time
   docker compose up -d
   ```

### Running the Full Stack

You need **3 terminals** inside the container:

| Terminal | Purpose | Command |
|----------|---------|---------|
| 1 | PX4 SITL + Gazebo | `cd /opt/PX4-Autopilot && make px4_sitl gz_x500` |
| 2 | MAVROS bridge | `source /opt/ros/humble/setup.bash && ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14557` |
| 3 | Your commands | `source /opt/ros/humble/setup.bash && cd /workspace/drone_spike` |

Open each with: `docker exec -it drone-sim bash`

### Verification Steps

```bash
# Terminal 3 — verify topics are flowing
ros2 topic list | grep mavros

# Check drone position
ros2 topic echo /mavros/local_position/pose --once

# Check drone state (armed? mode?)
ros2 topic echo /mavros/state --once

# Run offboard test
python3 scripts/12_px4_offboard_test.py
```

### Container Management

```bash
docker compose up -d          # Start
docker exec -it drone-sim bash  # Shell
docker compose down           # Stop
docker compose build          # Rebuild (after Dockerfile changes)
```

---

## run_all.py — Orchestrator

```bash
# Quick: runs 01-06 (no training, ~10 seconds)
python scripts/run_all.py --quick

# Full: runs 01-08 (includes training + eval, ~5-10 minutes)
python scripts/run_all.py
```

Runs scripts in numbered order. Stops on first failure with exit code 1.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: drone_intercept` | Run `pip install -e .` from project root |
| Script 08 says "no model found" | Run script 07 or 09 first to train a model |
| `torch._C._CudaDeviceProperties` error | Update script 06 (known PyTorch API change) |
| MAVROS won't connect in script 12 | Ensure `source /opt/ros/humble/setup.bash` was run |
| Gazebo window doesn't appear | Check VcXsrv is running with "Disable access control" |
| PX4 build fails with nuttx error | `sed` fix is baked into Dockerfile; rebuild if needed |
