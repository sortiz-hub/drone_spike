"""Step 13: Test the PX4 Gazebo physics backend with InterceptEnv.

Run INSIDE the Docker container (with PX4 SITL + MAVROS running):
    source /opt/ros/humble/setup.bash
    cd /workspace/drone_spike
    python3 scripts/13_px4_backend_test.py

This creates an InterceptEnv with physics_backend="px4_gazebo" and runs
a few episodes with random actions — proving the full RL loop works
against the real PX4 simulator.
"""

import sys
import numpy as np

try:
    import rclpy
except ImportError:
    print("ERROR: ROS 2 not available. Run inside the Docker container.")
    print("  source /opt/ros/humble/setup.bash")
    sys.exit(1)

from drone_intercept.env.intercept_env import InterceptEnv
from drone_intercept.env.termination import TerminationConfig


def main():
    # Shorter episodes for testing
    cfg = TerminationConfig(max_steps=100)

    print("Creating InterceptEnv with physics_backend='px4_gazebo'...")
    env = InterceptEnv(
        physics_backend="px4_gazebo",
        target_behavior="constant_velocity",
        target_speed=3.0,
        dt=0.1,
        termination=cfg,
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")

    for ep in range(3):
        print(f"\n--- Episode {ep} ---")
        obs, info = env.reset(seed=ep)
        print(f"Reset done. Drone pos: {info['drone_pos']}, Target pos: {info['target_pos']}")

        total_reward = 0.0
        for step in range(100):
            # Simple pursuit: move toward relative target position
            rel_pos = obs[6:9]
            dist = np.linalg.norm(rel_pos)
            if dist > 0.01:
                vel_cmd = rel_pos / dist * 3.0  # chase at 3 m/s
            else:
                vel_cmd = np.zeros(3)
            action = np.append(vel_cmd, 0.0).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if step % 20 == 0:
                print(f"  step={step:3d}  dist={info['distance']:.1f}m  reward={reward:.2f}  pos={info['drone_pos']}")

            if terminated or truncated:
                print(f"  DONE: {info['reason']} at step {step}")
                break

        print(f"  Total reward: {total_reward:.1f}")

    env.close()
    print("\nPASS — PX4 Gazebo backend works with InterceptEnv!")


if __name__ == "__main__":
    main()
