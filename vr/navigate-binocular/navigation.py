"""Sensor-independent local walking decisions. Coordinates are metres, x right, z forward."""

from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass
class LocalMap:
    """Observed floor and hazards on a metric x/z grid; unobserved is unknown."""

    x_edges: np.ndarray
    z_edges: np.ndarray
    observed: np.ndarray
    blocked: np.ndarray
    timestamp: float

    def __post_init__(self):
        shape = (len(self.z_edges) - 1, len(self.x_edges) - 1)
        if self.observed.shape != shape or self.blocked.shape != shape:
            raise ValueError("LocalMap arrays must match their metric edges")


def fuse_maps(maps, *, max_age=0.5, now=None):
    """Combine sensor evidence; blocked wins and stale evidence cannot clear space."""
    maps = list(maps)
    if not maps:
        raise ValueError("at least one map is required")
    now = time.monotonic() if now is None else now
    reference = maps[0]
    observed = np.zeros_like(reference.observed)
    blocked = np.zeros_like(reference.blocked)
    for scene in maps:
        if not (np.array_equal(scene.x_edges, reference.x_edges)
                and np.array_equal(scene.z_edges, reference.z_edges)):
            raise ValueError("sensor maps must use the same metric grid")
        if 0 <= now - scene.timestamp <= max_age:
            observed |= scene.observed
            blocked |= scene.blocked
    return LocalMap(reference.x_edges, reference.z_edges, observed, blocked, now)


@dataclass
class Decision:
    action: str  # "direction", "stop", or "unknown"
    direction: int | None
    near: np.ndarray
    ahead: np.ndarray
    unknown: np.ndarray


class Navigator:
    def __init__(self, *, path_width=0.65, near_limit=1.8,
                 min_observed_fraction=0.12, switch_frames=3):
        self.path_width = path_width
        self.near_limit = near_limit
        self.min_observed_fraction = min_observed_fraction
        self.switch_frames = switch_frames
        self.previous = None
        self.pending = None
        self.pending_count = 0

    def decide(self, scene: LocalMap) -> Decision:
        xs = (scene.x_edges[:-1] + scene.x_edges[1:]) / 2
        zs = (scene.z_edges[:-1] + scene.z_edges[1:]) / 2
        x, z = np.meshgrid(xs, zs)
        near_zone = z < self.near_limit
        ahead_zone = ~near_zone
        near = np.zeros(5, dtype=bool)
        ahead = np.zeros(5, dtype=bool)
        unknown = np.zeros(5, dtype=bool)

        # A column describes a person-wide candidate corridor, not an image pixel.
        for column, degrees in enumerate((-12, -6, 0, 6, 12)):
            corridor = np.abs(x - math.tan(math.radians(degrees)) * z) <= self.path_width / 2
            near[column] = np.any(scene.blocked & corridor & near_zone)
            ahead[column] = np.any(scene.blocked & corridor & ahead_zone)
            for zone in (near_zone, ahead_zone):
                region = corridor & zone
                if not np.any(region) or np.mean(scene.observed[region] | scene.blocked[region]) < self.min_observed_fraction:
                    unknown[column] = True

        # An imminent hazard on the current walking line overrides steering advice.
        if near[2]:
            self.previous = None
            return Decision("stop", None, near, ahead, unknown)

        viable = ~(near | ahead | unknown)
        if not np.any(viable):
            self.previous = None
            return Decision("unknown" if np.any(unknown) else "stop", None, near, ahead, unknown)

        preferred = min(np.flatnonzero(viable), key=lambda i: (abs(i - 2), abs(i - (self.previous if self.previous is not None else 2))))
        if self.previous is not None and viable[self.previous] and preferred != self.previous:
            if self.pending != preferred:
                self.pending, self.pending_count = preferred, 1
            else:
                self.pending_count += 1
            if self.pending_count < self.switch_frames:
                preferred = self.previous
        else:
            self.pending, self.pending_count = None, 0
        self.previous = int(preferred)
        return Decision("direction", self.previous, near, ahead, unknown)
