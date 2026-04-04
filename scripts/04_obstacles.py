"""Step 4: Phase 3 — obstacles with sector-distance perception."""

from drone_intercept.env.intercept_env import InterceptEnv
from drone_intercept.sim.obstacles import ObstacleConfig

cfg = ObstacleConfig()
env = InterceptEnv(obstacle_config=cfg)
obs, _ = env.reset(seed=42)
expected_dim = 14 + cfg.n_sectors
print(f"Obs shape: {obs.shape} (expect {expected_dim}D)")
assert obs.shape[0] == expected_dim, f"Expected {expected_dim}D, got {obs.shape[0]}D"

for _ in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())

print(f"Sector distances: {info.get('sector_distances', 'N/A')}")
env.close()
print("PASS")
