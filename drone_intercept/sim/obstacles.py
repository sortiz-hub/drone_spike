"""Obstacle generation and local obstacle perception via sector distances."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObstacleConfig:
    """Configuration for obstacle placement and perception.

    Attributes:
        n_obstacles: Number of cylindrical obstacles to place.
        radius_range: (min, max) obstacle radius in meters.
        height_range: (min, max) obstacle height in meters.
        placement_range: (min, max) distance from origin for obstacle centers.
        n_sectors: Number of angular sectors for local perception.
        perception_range: Maximum sensing range in meters.
    """

    n_obstacles: int = 8
    radius_range: tuple[float, float] = (1.0, 3.0)
    height_range: tuple[float, float] = (5.0, 20.0)
    placement_range: tuple[float, float] = (10.0, 80.0)
    n_sectors: int = 8
    perception_range: float = 20.0


@dataclass
class Obstacle:
    """A cylindrical obstacle (vertical column)."""

    center: np.ndarray  # [x, y] center position
    radius: float
    height: float


def generate_obstacles(
    rng: np.random.Generator,
    cfg: ObstacleConfig,
) -> list[Obstacle]:
    """Generate random cylindrical obstacles in the arena."""
    obstacles = []
    for _ in range(cfg.n_obstacles):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(*cfg.placement_range)
        center = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)
        radius = rng.uniform(*cfg.radius_range)
        height = rng.uniform(*cfg.height_range)
        obstacles.append(Obstacle(center=center, radius=radius, height=height))
    return obstacles


def compute_sector_distances(
    drone_pos: np.ndarray,
    obstacles: list[Obstacle],
    cfg: ObstacleConfig,
) -> np.ndarray:
    """Compute distance to nearest obstacle in each angular sector.

    Returns an array of shape (n_sectors,) where each element is the
    distance to the nearest obstacle surface in that sector, capped at
    perception_range. Uses the XY plane for sector angles.
    """
    n = cfg.n_sectors
    distances = np.full(n, cfg.perception_range, dtype=np.float32)
    sector_width = 2 * np.pi / n
    drone_xy = drone_pos[:2]
    drone_z = float(drone_pos[2])

    for obs in obstacles:
        # Skip if drone is above obstacle
        if drone_z > obs.height:
            continue

        to_obs = obs.center - drone_xy
        dist_to_center = float(np.linalg.norm(to_obs))
        # Distance to obstacle surface
        dist_to_surface = max(0.0, dist_to_center - obs.radius)

        if dist_to_surface >= cfg.perception_range:
            continue

        # Angle to obstacle center
        angle = float(np.arctan2(to_obs[1], to_obs[0]))
        if angle < 0:
            angle += 2 * np.pi

        # Angular extent of obstacle from drone's perspective
        if dist_to_center > obs.radius:
            half_angle = float(np.arcsin(min(1.0, obs.radius / dist_to_center)))
        else:
            # Drone is inside obstacle radius — affects all sectors
            half_angle = np.pi

        # Mark sectors covered by this obstacle
        for i in range(n):
            sector_center = i * sector_width
            # Angular distance between sector center and obstacle center
            diff = abs(sector_center - angle)
            diff = min(diff, 2 * np.pi - diff)
            if diff <= half_angle + sector_width / 2:
                distances[i] = min(distances[i], dist_to_surface)

    return distances


def check_obstacle_collision(
    drone_pos: np.ndarray,
    obstacles: list[Obstacle],
    drone_radius: float = 0.3,
) -> bool:
    """Check if the drone collides with any obstacle."""
    drone_xy = drone_pos[:2]
    drone_z = float(drone_pos[2])
    for obs in obstacles:
        if drone_z > obs.height:
            continue
        dist = float(np.linalg.norm(drone_xy - obs.center))
        if dist < obs.radius + drone_radius:
            return True
    return False
