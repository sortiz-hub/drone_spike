"""Dynamics backend abstraction.

Provides a pluggable interface for drone physics so the RL environment can
run with either simplified dynamics (fast, no dependencies) or a full
Gazebo + PX4 SITL stack (realistic, requires ROS 2).

Usage:
    from drone_intercept.sim.dynamics import SimplifiedDynamics, get_dynamics

    # Explicit
    dyn = SimplifiedDynamics()

    # By name (feature flag)
    dyn = get_dynamics("simplified")   # or "gazebo"
"""

from drone_intercept.sim.dynamics.base import DynamicsBackend, DynamicsState
from drone_intercept.sim.dynamics.simplified import SimplifiedDynamics


def get_dynamics(backend: str = "simplified", **kwargs) -> DynamicsBackend:
    """Factory: create a dynamics backend by name.

    Args:
        backend: "simplified" or "gazebo".
        **kwargs: Forwarded to the backend constructor.

    Raises:
        ValueError: Unknown backend name.
        ImportError: Gazebo backend requested but ROS 2 deps not installed.
    """
    if backend == "simplified":
        return SimplifiedDynamics(**kwargs)
    if backend == "gazebo":
        try:
            from drone_intercept.sim.dynamics.gazebo import GazeboDynamics
        except ImportError as e:
            raise ImportError(
                "Gazebo dynamics requires ROS 2 + PX4 dependencies. "
                "Install with: pip install -e '.[gazebo]'\n"
                f"Missing: {e}"
            ) from e
        return GazeboDynamics(**kwargs)
    raise ValueError(
        f"Unknown dynamics backend '{backend}'. Choose from: simplified, gazebo"
    )


__all__ = [
    "DynamicsBackend",
    "DynamicsState",
    "SimplifiedDynamics",
    "get_dynamics",
]
