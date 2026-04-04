"""Gazebo visual for the target — spawns and moves a drone model.

Used by PX4GazeboBackend to show the target in the 3D simulator.
Uses /world/default/set_pose service to move the static model each step.
"""

from __future__ import annotations

import numpy as np


class GzTargetVisual:
    """Manages a visual target entity in Gazebo Harmonic."""

    def __init__(self, name: str = "target_drone", model: str = "x500") -> None:
        from gz.transport13 import Node
        from gz.msgs10.entity_factory_pb2 import EntityFactory
        from gz.msgs10.pose_pb2 import Pose as GzPose
        from gz.msgs10.vector3d_pb2 import Vector3d
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose

        self._EntityFactory = EntityFactory
        self._GzPose = GzPose
        self._Vector3d = Vector3d
        self._Boolean = Boolean
        self._Pose = Pose

        self._node = Node()
        self._name = name
        self._model = model
        self._spawned = False

    def spawn(self, x: float, y: float, z: float) -> bool:
        """Spawn the target model at the given position."""
        if self._spawned:
            self.update(x, y, z)
            return True

        req = self._EntityFactory()
        req.sdf_filename = self._model
        req.name = self._name
        req.pose.CopyFrom(self._GzPose(
            position=self._Vector3d(x=x, y=y, z=z)
        ))

        ok, _ = self._node.request(
            "/world/default/create",
            req, self._EntityFactory, self._Boolean, 5000,
        )
        self._spawned = ok
        if ok:
            import time
            time.sleep(0.3)
        return ok

    def update(self, x: float, y: float, z: float) -> None:
        """Move the target to a new position via set_pose service."""
        req = self._Pose()
        req.name = self._name
        req.position.CopyFrom(self._Vector3d(x=float(x), y=float(y), z=float(z)))

        self._node.request(
            "/world/default/set_pose",
            req, self._Pose, self._Boolean, 100,
        )

    def update_from_array(self, pos: np.ndarray) -> None:
        """Move the target using a numpy position array."""
        self.update(float(pos[0]), float(pos[1]), float(pos[2]))

    def remove(self) -> None:
        """Remove the target from Gazebo."""
        if not self._spawned:
            return
        try:
            from gz.msgs10.entity_pb2 import Entity
            req = Entity()
            req.name = self._name
            req.type = 2  # MODEL
            self._node.request(
                "/world/default/remove",
                req, Entity, self._Boolean, 3000,
            )
        except Exception:
            pass
        self._spawned = False
