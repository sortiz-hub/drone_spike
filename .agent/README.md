# drone_spike Documentation

Drone interception RL spike — training pursuit-evasion policies with Gymnasium + SB3.

**Last Updated:** 2026-03-31

---

## Documentation Structure

```text
.agent/
├── README.md                          # This file – documentation index
├── system/
│   └── project_architecture.md        # Architecture, obs/action/reward, project structure
├── sop/
│   └── development/
│       ├── local-setup.md             # Installation, virtual env, smoke test
│       └── training-and-evaluation.md # Train, eval, view logs/plots, CLI reference
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
2. **Train**: `python -m drone_intercept.training.train_ppo --timesteps 500000`
3. **Evaluate**: `python -m drone_intercept.training.eval_policy models/ppo_intercept_final.zip`
4. **View plots**: Open `logs/eval/plot_episode_XXXXX.png` or use `plot_episode_from_file()`

Full training/eval reference: [sop/development/training-and-evaluation.md](sop/development/training-and-evaluation.md)

## Architecture & Design

- [system/project_architecture.md](system/project_architecture.md) — Runtime architecture, project structure, observation/action/reward reference, design decisions

## Current Status

**Phases 1–3** are implemented with simplified dynamics (no PX4/Gazebo dependency). Phase 2 adds noisy detections + Kalman filter (`--sensing-mode tracked`). Phase 3 adds obstacles with sector-distance perception (`--obstacles`).

## Parent Repository

This repo is part of the **rl-platform-root** monorepo ecosystem at `sortiz-hub/rl-platform-root`. The parent repo contains SPEC025 as a cross-cutting reference.
