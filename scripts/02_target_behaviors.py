"""Step 2: Target behaviors — do all three work?"""

from drone_intercept.env.intercept_env import InterceptEnv

for target in ["constant_velocity", "waypoint", "zigzag"]:
    env = InterceptEnv(target_behavior=target)
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, r, *_ = env.step(env.action_space.sample())
    env.close()
    print(f"{target}: PASS")

print("ALL PASS")
