"""ROS 2 node that sends velocity setpoints to PX4 via Offboard mode.

Publishes OffboardControlMode and TrajectorySetpoint messages at the
rate required by PX4 to maintain offboard control. If the heartbeat
stops, PX4 will revert to its failsafe mode.

Requires: rclpy, px4_msgs
"""

from __future__ import annotations

from typing import Any

import numpy as np


class OffboardCommandNode:
    """Sends velocity commands to PX4 Offboard mode via ROS 2."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from px4_msgs.msg import (
            OffboardControlMode,
            TrajectorySetpoint,
            VehicleCommand,
        )

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("offboard_commander")

        self._OffboardControlMode = OffboardControlMode
        self._TrajectorySetpoint = TrajectorySetpoint
        self._VehicleCommand = VehicleCommand

        self._mode_pub = self._node.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10,
        )
        self._setpoint_pub = self._node.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", 10,
        )
        self._cmd_pub = self._node.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10,
        )

    def send_velocity(
        self, vel_cmd: np.ndarray, yaw_rate: float = 0.0,
    ) -> None:
        """Publish an offboard velocity setpoint.

        Args:
            vel_cmd: [vx, vy, vz] in ENU frame (m/s). Converted to NED for PX4.
            yaw_rate: Yaw rate in rad/s.
        """
        # Offboard heartbeat
        mode = self._OffboardControlMode()
        mode.position = False
        mode.velocity = True
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        self._mode_pub.publish(mode)

        # Velocity setpoint (ENU → NED)
        sp = self._TrajectorySetpoint()
        sp.velocity[0] = float(vel_cmd[0])    # North = X
        sp.velocity[1] = -float(vel_cmd[1])   # East → -Y
        sp.velocity[2] = -float(vel_cmd[2])   # Up → -Z (down)
        sp.yawspeed = float(yaw_rate)
        self._setpoint_pub.publish(sp)

    def arm(self) -> None:
        """Send arm command to PX4."""
        cmd = self._VehicleCommand()
        cmd.command = self._VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self._cmd_pub.publish(cmd)

    def disarm(self) -> None:
        """Send disarm command to PX4."""
        cmd = self._VehicleCommand()
        cmd.command = self._VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 0.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self._cmd_pub.publish(cmd)

    def set_offboard_mode(self) -> None:
        """Send command to switch PX4 to offboard mode."""
        cmd = self._VehicleCommand()
        cmd.command = self._VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0   # custom mode
        cmd.param2 = 6.0   # offboard
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self._cmd_pub.publish(cmd)

    def destroy(self) -> None:
        self._node.destroy_node()
