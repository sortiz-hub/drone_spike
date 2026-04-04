"""Step 3: Phase 2 — tracked sensing with noise + Kalman filter."""

from drone_intercept.env.intercept_env import InterceptEnv

env = InterceptEnv(sensing_mode="tracked")
obs, info = env.reset(seed=42)
print(f"Obs shape: {obs.shape} (expect 15D)")
assert obs.shape[0] == 15, f"Expected 15D, got {obs.shape[0]}D"

for _ in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())

print(f"Track confidence: {info['track_confidence']:.2f}")
env.close()
print("PASS")
