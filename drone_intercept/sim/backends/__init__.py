from drone_intercept.sim.backends.base import PhysicsBackend
from drone_intercept.sim.backends.simplified import SimplifiedBackend

__all__ = ["PhysicsBackend", "SimplifiedBackend"]

def get_backend_registry() -> dict[str, type[PhysicsBackend]]:
    """Return all available backends. PX4 backend is lazy-loaded."""
    registry: dict[str, type[PhysicsBackend]] = {
        "simplified": SimplifiedBackend,
    }
    try:
        from drone_intercept.sim.backends.px4_gazebo import PX4GazeboBackend
        registry["px4_gazebo"] = PX4GazeboBackend
    except ImportError:
        pass  # ROS 2 not available — px4_gazebo backend unavailable
    return registry
