"""ROS 2 node that provides target state.

In Gazebo mode, the target is a separate model in the simulator. This node
subscribes to its ground-truth pose (or publishes scripted motion on a topic)
so the environment can read the target state each step.

For Phase 1/2 (truth or tracked), this publishes the target's actual state.
The noise injection and Kalman tracking happen in the environment layer.

Requires: rclpy, geometry_msgs
"""

from __future__ import annotations

from typing import Any

import numpy as np


class TargetProviderNode:
    """Provides target state from a Gazebo model or scripted behavior.

    Two modes:
      - "gazebo": Subscribes to a Gazebo model's ground-truth pose topic.
      - "scripted": Uses the existing TargetBehavior classes and publishes
        their state on a ROS topic for other nodes to consume.
    """

    def __init__(self, mode: str = "scripted") -> None:
        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self._node = Node("target_provider")
        self._mode = mode

        self._position = np.zeros(3, dtype=np.float32)
        self._velocity = np.zeros(3, dtype=np.float32)

        if mode == "gazebo":
            from geometry_msgs.msg import PoseStamped, TwistStamped

            self._pose_sub = self._node.create_subscription(
                PoseStamped, "/target/pose", self._on_pose, 10,
            )
            self._twist_sub = self._node.create_subscription(
                TwistStamped, "/target/twist", self._on_twist, 10,
            )

    def _on_pose(self, msg: Any) -> None:
        p = msg.pose.position
        self._position = np.array([p.x, p.y, p.z], dtype=np.float32)

    def _on_twist(self, msg: Any) -> None:
        v = msg.twist.linear
        self._velocity = np.array([v.x, v.y, v.z], dtype=np.float32)

    def update_from_behavior(
        self, position: np.ndarray, velocity: np.ndarray,
    ) -> None:
        """Directly set target state from a scripted TargetBehavior."""
        self._position = position.copy()
        self._velocity = velocity.copy()

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def spin_once(self, timeout_sec: float = 0.01) -> None:
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def destroy(self) -> None:
        self._node.destroy_node()
