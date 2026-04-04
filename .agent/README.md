# drone_spike Documentation

Drone interception RL spike — training pursuit-evasion policies with Gymnasium + SB3.

**Last Updated:** 2026-04-03

---

## Documentation Structure

```text
.agent/
├── README.md                          # This file – documentation index
├── system/
│   └── project_architecture.md        # Architecture, obs/action/reward, project structure
├── sop/
│   ├── development/
│   │   ├── local-setup.md             # Installation, virtual env, smoke test
│   │   └── training-and-evaluation.md # Train, eval, view logs/plots, CLI reference
│   └── testing/
│       └── validation-scripts.md      # Scripts 01–12 reference and usage
└── specs/
    ├── SPEC025-drone-interception-rl/  # Interception RL spec (4 phases)
    │   ├── requirements.md            # Requirements & user stories
    │   ├── architecture.md            # Technical architecture
    │   ├── delivery-strategy.md       # Phases & milestones
    │   ├── additional-diagrams.md     # Mermaid diagrams
    │   └── blueprint-reference.md     # Full drone interception RL blueprint
    └── _NEXT-SPEC026/                 # Next spec number placeholder
```

---

## Quick Start

1. **Install**: `pip install -e ".[dev]"` — see [sop/development/local-setup.md](sop/development/local-setup.md)
2. **Validate**: `python scripts/run_all.py --quick` — runs all validation scripts
3. **Train**: `python -m drone_intercept.training.train_ppo --timesteps 500000`
4. **Evaluate**: `python -m drone_intercept.training.eval_policy models/ppo_intercept_final.zip`
5. **View plots**: Open `logs/eval/plot_episode_XXXXX.png` or use `plot_episode_from_file()`
6. **Docker (PX4 + Gazebo)**: `cd docker && docker compose up -d` — see `docker/` for details

Full training/eval reference: [sop/development/training-and-evaluation.md](sop/development/training-and-evaluation.md)

## Architecture & Design

- [system/project_architecture.md](system/project_architecture.md) — Runtime architecture, project structure, observation/action/reward reference, design decisions

## Current Status

**All four phases** are implemented with simplified dynamics. Docker-based PX4 + Gazebo + ROS 2 Humble infrastructure is operational for Phase 2+ integration.

- Phase 2: noisy detections + Kalman filter (`--sensing-mode tracked`)
- Phase 3: obstacles with sector-distance perception (`--obstacles`)
- Phase 4: target prediction for lead pursuit (`--prediction`)
- Physics backend abstraction (`sim/backends/`) enables toggling between simplified and future Gazebo dynamics
- Validation scripts (`scripts/01–12`) cover end-to-end verification

## Parent Repository

This repo is part of the **rl-platform-root** monorepo ecosystem at `sortiz-hub/rl-platform-root`. The parent repo contains SPEC025 as a cross-cutting reference.
