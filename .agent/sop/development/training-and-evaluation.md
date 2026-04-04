# SOP: Training and Evaluation

## Prerequisites

- Python 3.10+
- Package installed: `pip install -e ".[dev]"`

## 1. Train a Policy

### Quick Start

```bash
python -m drone_intercept.training.train_ppo --timesteps 500000
```

This trains PPO against a constant-velocity target (5 m/s) with 4 parallel environments. Expect ~6000 FPS on a modern CPU.

### Full Options

```bash
python -m drone_intercept.training.train_ppo \
  --timesteps 500000 \
  --target constant_velocity \
  --target-speed 5.0 \
  --n-envs 4 \
  --lr 3e-4 \
  --batch-size 256 \
  --seed 42 \
  --save-dir models \
  --log-dir logs
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 500,000 | Total training steps |
| `--target` | `constant_velocity` | Target behavior: `constant_velocity`, `waypoint`, `zigzag` |
| `--target-speed` | 5.0 | Target speed in m/s |
| `--n-envs` | 4 | Parallel environments |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | Minibatch size |
| `--seed` | 42 | Random seed |
| `--save-dir` | `models` | Checkpoint directory |
| `--log-dir` | `logs` | Log directory |
| `--sensing-mode` | `truth` | `truth` (Phase 1) or `tracked` (Phase 2) |
| `--obstacles` | (flag) | Enable obstacles (Phase 3) |
| `--prediction` | (flag) | Enable target prediction (Phase 4) |
| `--device` | `cpu` | Compute device: `cpu`, `cuda`, or `auto` |
| `--reward-mode` | `original` | Reward mode: `original` or `shaped` (see `RewardConfig`) |
| `--resume` | (flag) | Auto-resume from `models/ppo_intercept_final.zip` if it exists |

### Convenience Script

```bash
# Full training run via validation script (equivalent to the CLI above)
python scripts/09_train_full.py
```

### Training Output

- **Checkpoints**: `models/ppo_intercept_{timestep}` every 50k steps
- **Final model**: `models/ppo_intercept_final.zip`
- **Console**: Episode metrics every 50 episodes (avg reward, success rate)

### Training with Tracked Sensing (Phase 2)

```bash
# Train with noisy detections + Kalman tracker (15D obs with track_confidence)
python -m drone_intercept.training.train_ppo --timesteps 500000 --sensing-mode tracked

# Evaluate Phase 2 policy
python -m drone_intercept.training.eval_policy models/ppo_intercept_final.zip --sensing-mode tracked
```

### Training with Obstacles (Phase 3)

```bash
# Train with obstacles (8 cylindrical obstacles, 8-sector perception)
python -m drone_intercept.training.train_ppo --timesteps 500000 --obstacles

# Combine with tracked sensing (Phase 2+3)
python -m drone_intercept.training.train_ppo --timesteps 500000 --sensing-mode tracked --obstacles
```

### Training with Prediction (Phase 4)

```bash
# Train with target prediction (adds predicted positions at t+0.5s, t+1.0s)
python -m drone_intercept.training.train_ppo --timesteps 500000 --prediction

# Full stack: tracked + obstacles + prediction
python -m drone_intercept.training.train_ppo --timesteps 500000 \
  --sensing-mode tracked --obstacles --prediction
```

### Curriculum Training (Progressive Difficulty)

```bash
# Stage 1: Constant velocity (easiest)
python -m drone_intercept.training.train_ppo --timesteps 500000 --target constant_velocity --save-dir models/stage1

# Stage 2: Waypoint-following
python -m drone_intercept.training.train_ppo --timesteps 500000 --target waypoint --save-dir models/stage2

# Stage 3: Zigzag (hardest)
python -m drone_intercept.training.train_ppo --timesteps 500000 --target zigzag --save-dir models/stage3
```

## 2. Evaluate a Policy

### Quick Evaluation

```bash
python -m drone_intercept.training.eval_policy models/ppo_intercept_final.zip --episodes 100
```

### Full Options

```bash
python -m drone_intercept.training.eval_policy \
  models/ppo_intercept_final.zip \
  --episodes 100 \
  --target constant_velocity \
  --target-speed 5.0 \
  --log-dir logs/eval \
  --seed 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_path` | (required) | Path to saved `.zip` model |
| `--episodes` | 100 | Number of evaluation episodes |
| `--target` | `constant_velocity` | Target behavior |
| `--target-speed` | 5.0 | Target speed in m/s |
| `--log-dir` | `logs/eval` | Output directory |
| `--no-plot` | (flag) | Skip trajectory plot generation |
| `--seed` | 0 | Random seed |
| `--device` | `cpu` | Compute device: `cpu`, `cuda`, or `auto` |

### Evaluation Output

- **Console**: Success rate, avg reward, avg min distance, avg capture time
- **Trajectory plots**: `logs/eval/plot_episode_XXXXX.png` (first 5 episodes)
- **Episode logs**: `logs/eval/episode_XXXXX.jsonl` (all episodes)
- **Summary CSV**: `logs/eval/episode_summaries.csv`

### Convenience Script

```bash
# Evaluate across all target types
python scripts/10_eval.py --all-targets
```

### Cross-Target Generalization Test

```bash
# Train on constant velocity
python -m drone_intercept.training.train_ppo --target constant_velocity --save-dir models/cv

# Evaluate on zigzag to test generalization
python -m drone_intercept.training.eval_policy models/cv/ppo_intercept_final.zip \
  --target zigzag --log-dir logs/eval_generalization
```

## 3. View Episode Logs and Plots

### View Trajectory Plots (Saved PNGs)

Evaluation automatically saves plots for the first 5 episodes to `logs/eval/plot_episode_XXXXX.png`. Each plot contains 3 panels:

1. **Top-down trajectory** — Drone path (blue), target path (red), start/end markers
2. **Distance over time** — Range to target with capture threshold line
3. **Reward over time** — Per-step reward signal

### Generate Plots from JSONL Logs

```python
from drone_intercept.replay.plotter import plot_episode_from_file

# Display interactively
plot_episode_from_file("logs/eval/episode_00000.jsonl")

# Save to file without displaying
plot_episode_from_file(
    "logs/eval/episode_00000.jsonl",
    save_path="my_plot.png",
    show=False,
)
```

### Batch-Plot Multiple Episodes

```python
from pathlib import Path
from drone_intercept.replay.plotter import plot_episode_from_file

log_dir = Path("logs/eval")
for jsonl in sorted(log_dir.glob("episode_*.jsonl"))[:10]:
    plot_episode_from_file(
        jsonl,
        save_path=log_dir / f"plot_{jsonl.stem}.png",
        show=False,
    )
```

### Batch Viewer Script

```bash
# Interactive batch viewer for episode plots
python scripts/11_batch_viewer.py
```

### Analyze Episode Summaries

```python
import csv
from pathlib import Path

with open("logs/eval/episode_summaries.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["success"] == "True":
            print(f"Episode {row['episode_id']}: captured in {row['capture_time']}s, "
                  f"min_dist={row['min_distance']}m")
```

### Inspect Raw Step Data

```python
from drone_intercept.replay.logger import EpisodeLogger

steps = EpisodeLogger.load_episode("logs/eval/episode_00000.jsonl")
for s in steps:
    print(f"t={s.t:3d}  dist={s.distance:.2f}m  reward={s.reward:+.2f}  "
          f"action=[{s.action[0]:.1f}, {s.action[1]:.1f}, {s.action[2]:.1f}]")
```

## 4. Success Criteria

Phase 1 target: **>80% capture rate** against a constant-velocity target (5 m/s) after 500k training steps.
