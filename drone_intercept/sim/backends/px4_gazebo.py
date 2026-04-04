"""PX4 + Gazebo backend via MAVROS/ROS 2.

Requires: Docker container with PX4 SITL + Gazebo + MAVROS running.
See docker/README.md for setup instructions.

Usage:
    InterceptEnv(physics_backend="px4_gazebo")
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from drone_intercept.sim.backends.base import PhysicsBackend

# PX4 velocity limits (conservative for SITL safety)
_MAX_VEL = 5.0        # m/s per axis
_MAX_YAW_RATE = 1.0   # rad/s
_HOVER_ALT = 3.0       # meters — takeoff target altitude
_SETTLE_TIME = 1.0     # seconds to wait after mode/arm transitions


@dataclass
class PX4GazeboConfig:
    """Configuration for the PX4/Gazebo backend.

    Attributes:
        hover_alt: Target altitude for takeoff in meters.
        pre_stream_duration: Seconds to stream setpoints before Offboard switch.
        step_rate: Rate (Hz) for publishing setpoints during step().
        reset_timeout: Max seconds to wait for takeoff during reset.
    """

    hover_alt: float = 3.0
    pre_stream_duration: float = 2.0
    step_rate: float = 20.0
    reset_timeout: float = 15.0


class PX4GazeboBackend(PhysicsBackend):
    """Physics backend that delegates to PX4 SITL via MAVROS.

    On step(): publishes velocity setpoint, spins ROS 2, reads back state.
    On reset(): disarms, resets Gazebo world, re-arms, takes off to hover.

    Must run inside the Docker container with PX4 SITL + MAVROS active.
    """

    def __init__(self, cfg: PX4GazeboConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PX4GazeboConfig()
        self._node = None
        self._spin_thread = None
        self._connected = False
        self._init_ros()

    def _init_ros(self) -> None:
        """Initialize ROS 2 node and MAVROS subscriptions/publishers."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor
            from geometry_msgs.msg import TwistStamped, PoseStamped, TwistStamped as Twist
            from mavros_msgs.msg import State
            from mavros_msgs.srv import CommandBool, SetMode, CommandLong
        except ImportError as e:
            raise RuntimeError(
                "ROS 2 / MAVROS not available. This backend must run inside "
                "the Docker container. See docker/README.md"
            ) from e

        if not rclpy.ok():
            rclpy.init()

        self._rclpy = rclpy
        self._Node = Node
        self._TwistStamped = TwistStamped
        self._PoseStamped = PoseStamped
        self._State = State
        self._CommandBool = CommandBool
        self._SetMode = SetMode
        self._CommandLong = CommandLong

        self._node = Node("px4_gazebo_backend")

        # Subscribers
        self._mavros_state = State()
        self._pose_msg = None
        self._vel_msg = None

        self._node.create_subscription(
            State, "/mavros/state", self._on_state, 10
        )
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self._node.create_subscription(
            PoseStamped, "/mavros/local_position/pose",
            self._on_pose, sensor_qos,
        )
        self._node.create_subscription(
            TwistStamped, "/mavros/local_position/velocity_local",
            self._on_velocity, sensor_qos,
        )

        # Publisher
        self._vel_pub = self._node.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )

        # Service clients
        self._arming_client = self._node.create_client(
            CommandBool, "/mavros/cmd/arming"
        )
        self._set_mode_client = self._node.create_client(
            SetMode, "/mavros/set_mode"
        )

        # Spin in background thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True
        )
        self._spin_thread.start()

        # Wait for connection
        deadline = time.time() + 10.0
        while time.time() < deadline and not self._mavros_state.connected:
            time.sleep(0.1)
        if not self._mavros_state.connected:
            raise RuntimeError("Timeout waiting for MAVROS connection")
        self._connected = True

    def _spin_loop(self) -> None:
        """Background ROS 2 spin."""
        try:
            while self._rclpy.ok():
                self._executor.spin_once(timeout_sec=0.01)
        except Exception:
            pass

    # -- ROS 2 callbacks --

    def _on_state(self, msg) -> None:
        self._mavros_state = msg

    def _on_pose(self, msg) -> None:
        self._pose_msg = msg
        p = msg.pose.position
        q = msg.pose.orientation
        # PX4 uses NED internally but MAVROS local_position is in ENU
        self.position = np.array([p.x, p.y, p.z], dtype=np.float32)
        # Extract yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = float(math.atan2(siny_cosp, cosy_cosp))

    def _on_velocity(self, msg) -> None:
        self._vel_msg = msg
        v = msg.twist.linear
        self.velocity = np.array([v.x, v.y, v.z], dtype=np.float32)

    # -- Service helpers --

    def _call_service(self, client, request, timeout: float = 5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(request)
        # Wait synchronously (background spin handles callbacks)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        return future.result() if future.done() else None

    def _set_mode(self, mode: str) -> bool:
        req = self._SetMode.Request()
        req.custom_mode = mode
        result = self._call_service(self._set_mode_client, req)
        return result is not None and result.mode_sent

    def _arm(self, value: bool = True) -> bool:
        req = self._CommandBool.Request()
        req.value = value
        result = self._call_service(self._arming_client, req)
        return result is not None and result.success

    def _publish_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0) -> None:
        msg = self._TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.z = float(yaw_rate)
        self._vel_pub.publish(msg)

    def _stream_setpoints(self, vx, vy, vz, duration: float, yaw_rate=0.0) -> None:
        """Publish velocity at step_rate for a duration."""
        interval = 1.0 / self.cfg.step_rate
        steps = int(duration * self.cfg.step_rate)
        for _ in range(steps):
            self._publish_velocity(vx, vy, vz, yaw_rate)
            time.sleep(interval)

    # -- PhysicsBackend interface --

    def reset(self, rng: np.random.Generator) -> None:
        """Reset PX4: land if airborne, disarm, re-arm, takeoff to hover altitude."""
        # If armed, land first
        if self._mavros_state.armed:
            self._stream_setpoints(0.0, 0.0, -1.5, duration=5.0)
            self._arm(False)
            time.sleep(1.0)

        # Pre-stream setpoints (PX4 requires this before Offboard)
        self._stream_setpoints(0.0, 0.0, 0.0, duration=self.cfg.pre_stream_duration)

        # Switch to OFFBOARD
        self._set_mode("OFFBOARD")
        self._stream_setpoints(0.0, 0.0, 0.0, duration=0.5)

        # Arm
        self._arm(True)
        time.sleep(0.5)

        # Takeoff — climb to hover altitude
        deadline = time.time() + self.cfg.reset_timeout
        while time.time() < deadline:
            alt = float(self.position[2]) if self._pose_msg else 0.0
            if alt >= self.cfg.hover_alt * 0.9:
                break
            self._publish_velocity(0.0, 0.0, 2.0)
            time.sleep(1.0 / self.cfg.step_rate)

        # Stabilize at hover
        self._stream_setpoints(0.0, 0.0, 0.0, duration=_SETTLE_TIME)

        # Battery is not simulated in PX4 SITL — keep at 1.0
        self.battery = 1.0

    def step(self, action: np.ndarray, dt: float) -> None:
        """Send velocity command to PX4 and wait for dt.

        The background spin thread continuously updates self.position,
        self.velocity, and self.yaw from MAVROS topics.
        """
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        yaw_rate = float(action[3])

        # Publish at step_rate for dt duration
        interval = 1.0 / self.cfg.step_rate
        n_publishes = max(1, int(dt * self.cfg.step_rate))
        for _ in range(n_publishes):
            self._publish_velocity(vx, vy, vz, yaw_rate)
            time.sleep(interval)

        # State is updated by background callbacks — nothing else to do

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.array([-_MAX_VEL, -_MAX_VEL, -_MAX_VEL, -_MAX_YAW_RATE], dtype=np.float32)
        high = np.array([_MAX_VEL, _MAX_VEL, _MAX_VEL, _MAX_YAW_RATE], dtype=np.float32)
        return low, high

    def close(self) -> None:
        """Clean shutdown: land and disarm."""
        try:
            if self._mavros_state.armed:
                self._stream_setpoints(0.0, 0.0, -1.0, duration=5.0)
                self._arm(False)
        except Exception:
            pass
        try:
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
        except Exception:
            pass
        try:
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass
