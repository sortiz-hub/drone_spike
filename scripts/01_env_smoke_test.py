"""Step 1: Environment smoke test — does it even run?"""

from drone_intercept.env.intercept_env import InterceptEnv

env = InterceptEnv()
obs, info = env.reset(seed=42)
print(f"Obs shape: {obs.shape}, first 5: {obs[:5]}")

action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(action)
print(f"Reward: {reward:.2f}, Distance: {info['distance']:.1f}m")

env.close()
print("PASS")
