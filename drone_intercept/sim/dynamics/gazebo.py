"""Gazebo + PX4 SITL dynamics backend via ROS 2.

Requires:
  - ROS 2 (Humble or later) with rclpy
  - PX4 SITL running with Gazebo
  - px4_msgs package
  - MAVROS or px4_ros_com bridge

This backend sends velocity setpoints to PX4 Offboard mode and reads
drone state from PX4 state estimation topics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_intercept.sim.dynamics.base import DynamicsBackend, DynamicsState


@dataclass
class GazeboConfig:
    """Configuration for the Gazebo + PX4 backend.

    Attributes:
        world_file: Path to the Gazebo .sdf world file (None = default empty world).
        vehicle_model: PX4 vehicle model (e.g. "x500").
        offboard_topic: ROS 2 topic for velocity setpoints.
        state_topic: ROS 2 topic for vehicle state.
        arm_timeout: Seconds to wait for arming.
        step_timeout: Seconds to wait for a step to complete.
    """

    world_file: str | None = None
    vehicle_model: str = "x500"
    offboard_topic: str = "/fmu/in/trajectory_setpoint"
    state_topic: str = "/fmu/out/vehicle_local_position"
    arm_timeout: float = 10.0
    step_timeout: float = 2.0


class GazeboDynamics(DynamicsBackend):
    """Dynamics backend that drives PX4 SITL via ROS 2.

    Lifecycle:
        1. __init__: Imports rclpy, creates ROS node, sets up pub/sub.
        2. reset(): Arms the vehicle, enters offboard mode, hovers.
        3. step(): Publishes velocity setpoint, waits for state update.
        4. close(): Disarms, shuts down node.

    If rclpy or px4_msgs are not installed, __init__ raises ImportError
    with a helpful message.
    """

    def __init__(self, config: GazeboConfig | None = None) -> None:
        self._cfg = config or GazeboConfig()
        self._state = DynamicsState()

        # Lazy ROS 2 imports — fail fast with clear message
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import (
                QoSProfile,
                ReliabilityPolicy,
                HistoryPolicy,
                DurabilityPolicy,
            )
        except ImportError as e:
            raise ImportError(
                "GazeboDynamics requires ROS 2 (rclpy). "
                "Install ROS 2 Humble+ and source the workspace.\n"
                f"Missing: {e}"
            ) from e

        try:
            from px4_msgs.msg import (
                OffboardControlMode,
                TrajectorySetpoint,
                VehicleCommand,
                VehicleLocalPosition,
                VehicleStatus,
            )
        except ImportError as e:
            raise ImportError(
                "GazeboDynamics requires px4_msgs. "
                "Install with: pip install px4-msgs or build from source.\n"
                f"Missing: {e}"
            ) from e

        # Store message types for use in methods
        self._msg_types = {
            "OffboardControlMode": OffboardControlMode,
            "TrajectorySetpoint": TrajectorySetpoint,
            "VehicleCommand": VehicleCommand,
            "VehicleLocalPosition": VehicleLocalPosition,
            "VehicleStatus": VehicleStatus,
        }

        # QoS compatible with PX4 micro-XRCE-DDS bridge
        self._qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Init ROS 2
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("intercept_dynamics")

        # Publishers
        self._offboard_mode_pub = self._node.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10,
        )
        self._setpoint_pub = self._node.create_publisher(
            TrajectorySetpoint, self._cfg.offboard_topic, 10,
        )
        self._cmd_pub = self._node.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10,
        )

        # Subscribers
        self._vehicle_pos: VehicleLocalPosition | None = None
        self._vehicle_status: VehicleStatus | None = None

        self._pos_sub = self._node.create_subscription(
            VehicleLocalPosition, self._cfg.state_topic,
            self._on_vehicle_pos, self._qos,
        )
        self._status_sub = self._node.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self._on_vehicle_status, self._qos,
        )

        self._step_count = 0

    # ── ROS callbacks ──────────────────────────────────────────

    def _on_vehicle_pos(self, msg: Any) -> None:
        self._vehicle_pos = msg

    def _on_vehicle_status(self, msg: Any) -> None:
        self._vehicle_status = msg

    # ── DynamicsBackend interface ──────────────────────────────

    def reset(self, rng: np.random.Generator) -> DynamicsState:
        VehicleCommand = self._msg_types["VehicleCommand"]
        OffboardControlMode = self._msg_types["OffboardControlMode"]
        TrajectorySetpoint = self._msg_types["TrajectorySetpoint"]

        # Send a few offboard heartbeats before arming
        for _ in range(10):
            self._publish_offboard_mode()
            self._publish_hover_setpoint()
            self._spin_once(0.1)

        # Arm
        cmd = VehicleCommand()
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0  # arm
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self._cmd_pub.publish(cmd)

        # Switch to offboard mode
        cmd = VehicleCommand()
        cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0  # custom mode
        cmd.param2 = 6.0  # offboard
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self._cmd_pub.publish(cmd)

        # Wait for state
        deadline = time.monotonic() + self._cfg.arm_timeout
        while self._vehicle_pos is None and time.monotonic() < deadline:
            self._publish_offboard_mode()
            self._publish_hover_setpoint()
            self._spin_once(0.1)

        self._step_count = 0
        return self._read_state()

    def step(
        self,
        vel_cmd: np.ndarray,
        yaw_rate_cmd: float,
        dt: float,
    ) -> DynamicsState:
        TrajectorySetpoint = self._msg_types["TrajectorySetpoint"]

        # Publish offboard heartbeat + velocity setpoint
        self._publish_offboard_mode()

        sp = TrajectorySetpoint()
        # PX4 uses NED, our env uses ENU — convert
        sp.velocity[0] = float(vel_cmd[0])   # North = X
        sp.velocity[1] = -float(vel_cmd[1])  # East = -Y (NED)
        sp.velocity[2] = -float(vel_cmd[2])  # Down = -Z (NED)
        sp.yawspeed = float(yaw_rate_cmd)
        self._setpoint_pub.publish(sp)

        # Spin for dt to let the sim advance
        self._spin_once(dt)
        self._step_count += 1

        return self._read_state()

    def close(self) -> None:
        # Disarm
        try:
            VehicleCommand = self._msg_types["VehicleCommand"]
            cmd = VehicleCommand()
            cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
            cmd.param1 = 0.0  # disarm
            cmd.target_system = 1
            cmd.target_component = 1
            cmd.source_system = 1
            cmd.source_component = 1
            cmd.from_external = True
            self._cmd_pub.publish(cmd)
        except Exception:
            pass

        try:
            self._node.destroy_node()
        except Exception:
            pass

    # ── Internal helpers ───────────────────────────────────────

    def _spin_once(self, timeout_sec: float) -> None:
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def _publish_offboard_mode(self) -> None:
        OffboardControlMode = self._msg_types["OffboardControlMode"]
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self._offboard_mode_pub.publish(msg)

    def _publish_hover_setpoint(self) -> None:
        TrajectorySetpoint = self._msg_types["TrajectorySetpoint"]
        sp = TrajectorySetpoint()
        sp.velocity[0] = 0.0
        sp.velocity[1] = 0.0
        sp.velocity[2] = 0.0
        self._setpoint_pub.publish(sp)

    def _read_state(self) -> DynamicsState:
        """Convert latest PX4 local position to DynamicsState."""
        if self._vehicle_pos is None:
            return self._state

        msg = self._vehicle_pos
        # PX4 NED → our ENU
        self._state = DynamicsState(
            position=np.array(
                [msg.x, -msg.y, -msg.z], dtype=np.float32,
            ),
            velocity=np.array(
                [msg.vx, -msg.vy, -msg.vz], dtype=np.float32,
            ),
            yaw=float(-msg.heading),  # NED→ENU yaw
            battery=max(0.0, 1.0 - self._step_count * 0.0002),
        )
        return self._state
