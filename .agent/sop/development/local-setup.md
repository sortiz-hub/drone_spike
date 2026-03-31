# SOP: Local Development Setup

## Prerequisites

- Python 3.10 or newer
- pip (recent version)
- Git

## Installation

### 1. Clone and Enter Repo

```bash
cd /c/dev/rl-energyplus
git clone <repo-url> drone_spike  # or navigate to existing clone
cd drone_spike
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/WSL/macOS
# or
venv\Scripts\activate           # Windows CMD
# or
source venv/Scripts/activate    # Windows Git Bash
```

### 3. Install Package

```bash
# Core dependencies (gymnasium, stable-baselines3, numpy, matplotlib)
pip install -e .

# With dev tools (pytest, ruff)
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
python -c "
from drone_intercept.env.intercept_env import InterceptEnv
env = InterceptEnv()
obs, info = env.reset(seed=42)
print(f'Observation shape: {obs.shape}')
print('Installation OK')
"
```

Expected output:
```
Observation shape: (14,)
Installation OK
```

## Quick Smoke Test

Run a naive pursuit policy to verify the environment works end-to-end:

```bash
python -c "
import numpy as np
from drone_intercept.env.intercept_env import InterceptEnv

env = InterceptEnv(target_behavior='constant_velocity', target_speed=3.0)
captures = 0
for ep in range(10):
    obs, _ = env.reset(seed=ep)
    for _ in range(1000):
        rel_pos = obs[6:9]
        dist = np.linalg.norm(rel_pos)
        vel_cmd = rel_pos / max(dist, 0.01) * 8.0
        action = np.append(vel_cmd, 0.0).astype(np.float32)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            if info.get('captured'):
                captures += 1
            break
print(f'Captures: {captures}/10 (expect 10/10)')
"
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| gymnasium | >=0.29 | RL environment interface |
| stable-baselines3 | >=2.1 | PPO training |
| numpy | >=1.24 | Numerical computing |
| matplotlib | >=3.7 | Trajectory plots |
| pytest | >=7.0 | Testing (dev) |
| ruff | >=0.1 | Linting (dev) |
