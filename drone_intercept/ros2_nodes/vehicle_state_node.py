"""ROS 2 node that subscribes to PX4 vehicle state estimation.

Reads VehicleLocalPosition and VehicleAttitude topics from PX4's
micro-XRCE-DDS bridge and exposes the latest state for the Gymnasium
environment to query.

Requires: rclpy, px4_msgs
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VehicleStateNode:
    """Reads drone state from PX4 via ROS 2.

    Publishes nothing — only subscribes and caches the latest state.
    The environment polls .position / .velocity / .yaw each step.
    """

    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
        )
        from px4_msgs.msg import VehicleLocalPosition

        self._qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("vehicle_state_reader")

        self._position = np.zeros(3, dtype=np.float32)
        self._velocity = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0

        self._sub = self._node.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._on_msg,
            self._qos,
        )

    def _on_msg(self, msg: Any) -> None:
        # PX4 NED → ENU
        self._position = np.array([msg.x, -msg.y, -msg.z], dtype=np.float32)
        self._velocity = np.array([msg.vx, -msg.vy, -msg.vz], dtype=np.float32)
        self._yaw = float(-msg.heading)

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def yaw(self) -> float:
        return self._yaw

    def spin_once(self, timeout_sec: float = 0.01) -> None:
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def destroy(self) -> None:
        self._node.destroy_node()
