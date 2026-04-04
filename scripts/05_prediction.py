"""Step 5: Phase 4 — target prediction for lead pursuit."""

from drone_intercept.env.intercept_env import InterceptEnv
from drone_intercept.sim.predictor import PredictorConfig

cfg = PredictorConfig()
env = InterceptEnv(predictor_config=cfg)
obs, _ = env.reset(seed=42)
expected_dim = 14 + len(cfg.horizons) * 3
print(f"Obs shape: {obs.shape} (expect {expected_dim}D)")
assert obs.shape[0] == expected_dim, f"Expected {expected_dim}D, got {obs.shape[0]}D"

for _ in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())

print(f"Predictions: {info.get('predicted_positions', 'N/A')}")
env.close()
print("PASS")
