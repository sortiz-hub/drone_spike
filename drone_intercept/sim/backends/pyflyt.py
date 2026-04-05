"""PyFlyt (PyBullet) physics backend — validated quadrotor dynamics.

Faster than PX4+Gazebo (~5-20k steps/sec) with realistic CrazyFlie-based
quadrotor physics. Uses PyFlyt mode 6: ground-frame velocity control.

Install: pip install PyFlyt

Usage:
    InterceptEnv(physics_backend="pyflyt")
"""

from __future__ import annotations

import numpy as np

from drone_intercept.sim.backends.base import PhysicsBackend

_MAX_VEL = 5.0        # m/s per axis
_MAX_YAW_RATE = 2.0   # rad/s


class PyFlytBackend(PhysicsBackend):
    """PyBullet-based quadrotor dynamics via PyFlyt.

    Uses the Aviary + QuadX in mode 6 (ground-frame velocity control).
    Setpoint mapping: our [vx, vy, vz, yaw_rate] -> PyFlyt [vx, vy, yaw_rate, vz].
    """

    def __init__(self, physics_hz: int = 240, control_hz: int = 120) -> None:
        super().__init__()
        self._physics_hz = physics_hz
        self._control_hz = control_hz
        self._aviary = None

    def reset(self, rng: np.random.Generator) -> None:
        from PyFlyt.core import Aviary

        # Random initial position near origin
        start_pos = np.array([[
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(2.0, 4.0),
        ]])
        start_orn = np.array([[0.0, 0.0, 0.0]])

        # Tear down old sim if exists
        if self._aviary is not None:
            try:
                self._aviary.disconnect()
            except Exception:
                pass

        self._aviary = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="quadx",
            render=False,
            physics_hz=self._physics_hz,
        )
        # Mode 6 = ground-frame velocity: [vx, vy, yaw_rate, vz]
        self._aviary.set_mode(6)

        # Let the sim stabilize
        for _ in range(10):
            self._aviary.set_setpoint(0, np.array([0.0, 0.0, 0.0, 0.0]))
            self._aviary.step()

        self._sync_state()
        self.battery = 1.0

    def step(self, action: np.ndarray, dt: float) -> None:
        # Our action: [vx, vy, vz, yaw_rate]
        # PyFlyt mode 6: [vx, vy, yaw_rate, vz]
        vx, vy, vz, yaw_rate = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        setpoint = np.array([vx, vy, yaw_rate, vz])
        self._aviary.set_setpoint(0, setpoint)

        # Step enough times to cover dt
        n_steps = max(1, int(round(dt * self._control_hz)))
        for _ in range(n_steps):
            self._aviary.step()

        self._sync_state()
        self.battery = max(0.0, self.battery - 0.0002)

    def _sync_state(self) -> None:
        """Pull position, velocity, yaw from PyFlyt into our state attrs."""
        state = self._aviary.state(0)  # (4, 3)
        # state[3] = ground-frame position (x, y, z)
        self.position = state[3].astype(np.float32)
        # state[1] = Euler angles (roll, pitch, yaw)
        self.yaw = float(state[1][2])
        # state[2] = body-frame velocity; rotate to world frame
        roll, pitch, yaw = state[1]
        quat = self._aviary.getQuaternionFromEuler([roll, pitch, yaw])
        rot = np.array(self._aviary.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.velocity = (rot @ state[2]).astype(np.float32)

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.array([-_MAX_VEL, -_MAX_VEL, -_MAX_VEL, -_MAX_YAW_RATE], dtype=np.float32)
        high = np.array([_MAX_VEL, _MAX_VEL, _MAX_VEL, _MAX_YAW_RATE], dtype=np.float32)
        return low, high

    def close(self) -> None:
        if self._aviary is not None:
            try:
                self._aviary.disconnect()
            except Exception:
                pass
            self._aviary = None
