"""Sensor-independent local walking decisions. Coordinates are metres, x right, z forward."""

from dataclasses import dataclass
import math
import time

import numpy as np

from config import (
    DEBOUNCE_ADD_OBSTACLE_MS, DEBOUNCE_REMOVE_OBSTACLE_MS,
    DEBOUNCE_ADD_UNKNOWN_MS, DEBOUNCE_REMOVE_UNKNOWN_MS,
    DEBOUNCE_ADD_PATH_MS, DEBOUNCE_PATH_CHOICE_MS,
    DEBOUNCE_ADD_STOP_MS, DEBOUNCE_RELEASE_STOP_MS,
)


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
    near_coverage: np.ndarray
    ahead_coverage: np.ndarray
    reason: str


class _TimedBits:
    """Accept a bit change only after the same observation persists in time."""

    def __init__(self, initial=None):
        self.value = None if initial is None else np.asarray(initial, dtype=bool).copy()
        self.pending = None if initial is None else np.zeros(self.value.shape, dtype=bool)
        self.since = None if initial is None else np.zeros(self.value.shape, dtype=float)

    def update(self, observed, now, *, add_ms, remove_ms):
        observed = np.asarray(observed, dtype=bool)
        if self.value is None:
            self.value = observed.copy()
            self.pending = np.zeros(observed.shape, dtype=bool)
            self.since = np.full(observed.shape, now, dtype=float)
            return self.value.copy()
        for index in np.ndindex(observed.shape):
            target = bool(observed[index])
            if target == self.value[index]:
                self.pending[index] = False
            else:
                if not self.pending[index]:
                    self.pending[index] = True
                    self.since[index] = now
                delay_ms = add_ms if target else remove_ms
                if (now - self.since[index]) * 1000 >= delay_ms:
                    self.value[index] = target
                    self.pending[index] = False
        return self.value.copy()


class Navigator:
    def __init__(self, *, path_width=0.65, stop_distance=1.8,
                 near_limit=2.0, ahead_limit=3.0,
                 min_observed_fraction=0.12,
                 debounce_add_obstacle_ms=DEBOUNCE_ADD_OBSTACLE_MS,
                 debounce_remove_obstacle_ms=DEBOUNCE_REMOVE_OBSTACLE_MS,
                 debounce_add_unknown_ms=DEBOUNCE_ADD_UNKNOWN_MS,
                 debounce_remove_unknown_ms=DEBOUNCE_REMOVE_UNKNOWN_MS,
                 debounce_add_path_ms=DEBOUNCE_ADD_PATH_MS,
                 debounce_path_choice_ms=DEBOUNCE_PATH_CHOICE_MS,
                 debounce_add_stop_ms=DEBOUNCE_ADD_STOP_MS,
                 debounce_release_stop_ms=DEBOUNCE_RELEASE_STOP_MS):
        if not 0 < stop_distance < near_limit < ahead_limit:
            raise ValueError("distances must satisfy 0 < stop < near < ahead")
        delays = (debounce_add_obstacle_ms, debounce_remove_obstacle_ms,
                  debounce_add_unknown_ms, debounce_remove_unknown_ms,
                  debounce_add_path_ms, debounce_path_choice_ms,
                  debounce_add_stop_ms,
                  debounce_release_stop_ms)
        if any(not math.isfinite(delay) or delay < 0 for delay in delays):
            raise ValueError("debounce delays must be finite nonnegative milliseconds")
        self.path_width = path_width
        self.stop_distance = stop_distance
        self.near_limit = near_limit
        self.ahead_limit = ahead_limit
        self.min_observed_fraction = min_observed_fraction
        self.debounce_add_obstacle_ms = debounce_add_obstacle_ms
        self.debounce_remove_obstacle_ms = debounce_remove_obstacle_ms
        self.debounce_add_unknown_ms = debounce_add_unknown_ms
        self.debounce_remove_unknown_ms = debounce_remove_unknown_ms
        self.debounce_add_path_ms = debounce_add_path_ms
        self.debounce_path_choice_ms = debounce_path_choice_ms
        self.debounce_add_stop_ms = debounce_add_stop_ms
        self.debounce_release_stop_ms = debounce_release_stop_ms
        self._near = _TimedBits()
        self._ahead = _TimedBits()
        self._unknown = _TimedBits()
        self._paths = _TimedBits()
        self._stop = _TimedBits(initial=[False])
        self.previous = None
        self.pending = None
        self.pending_since = None

    def decide(self, scene: LocalMap, *, now=None) -> Decision:
        now = time.monotonic() if now is None else now
        xs = (scene.x_edges[:-1] + scene.x_edges[1:]) / 2
        zs = (scene.z_edges[:-1] + scene.z_edges[1:]) / 2
        x, z = np.meshgrid(xs, zs)
        near_zone = z < self.near_limit
        ahead_zone = (z >= self.near_limit) & (z < self.ahead_limit)
        stop_zone = z < self.stop_distance
        near = np.zeros(5, dtype=bool)
        ahead = np.zeros(5, dtype=bool)
        unknown = np.zeros(5, dtype=bool)
        near_coverage = np.zeros(5, dtype=np.float32)
        ahead_coverage = np.zeros(5, dtype=np.float32)
        center_stop = False

        # A column describes a person-wide candidate corridor, not an image pixel.
        for column, degrees in enumerate((-12, -6, 0, 6, 12)):
            corridor = np.abs(x - math.tan(math.radians(degrees)) * z) <= self.path_width / 2
            near[column] = np.any(scene.blocked & corridor & near_zone)
            ahead[column] = np.any(scene.blocked & corridor & ahead_zone)
            if column == 2:
                center_stop = bool(np.any(scene.blocked & corridor & stop_zone))
            for zone, coverage in ((near_zone, near_coverage), (ahead_zone, ahead_coverage)):
                region = corridor & zone
                coverage[column] = np.mean(scene.observed[region] | scene.blocked[region]) if np.any(region) else 0
                if coverage[column] < self.min_observed_fraction:
                    unknown[column] = True

        # Smooth each displayed cell independently.
        near = self._near.update(near, now, add_ms=self.debounce_add_obstacle_ms,
                                 remove_ms=self.debounce_remove_obstacle_ms)
        ahead = self._ahead.update(ahead, now, add_ms=self.debounce_add_obstacle_ms,
                                   remove_ms=self.debounce_remove_obstacle_ms)
        unknown = self._unknown.update(unknown, now, add_ms=self.debounce_add_unknown_ms,
                                       remove_ms=self.debounce_remove_unknown_ms)
        clear_corridors = ~(near | ahead | unknown)
        viable = self._paths.update(clear_corridors, now,
                                    add_ms=self.debounce_add_path_ms, remove_ms=0)

        if center_stop:
            candidate_action = "stop"
            reason = "hazard within stop distance"
        elif not np.any(viable):
            candidate_action = "unknown" if np.any(unknown | clear_corridors) else "stop"
            reason = ("waiting for a clear corridor to settle" if np.any(clear_corridors)
                      else "no verified clear corridor" if candidate_action == "unknown"
                      else "all corridors blocked")
        else:
            candidate_action = "direction"

        stop = self._stop.update(np.array([candidate_action == "stop"]), now,
                                 add_ms=self.debounce_add_stop_ms,
                                 remove_ms=self.debounce_release_stop_ms)[0]
        if stop or candidate_action != "direction":
            self.previous = None
            self.pending = self.pending_since = None
            if stop:
                return Decision("stop", None, near, ahead, unknown,
                                near_coverage, ahead_coverage,
                                reason if candidate_action == "stop" else "waiting for stop condition to clear")
            if candidate_action == "stop":
                reason = "checking possible stop condition"
            return Decision("unknown", None, near, ahead, unknown,
                            near_coverage, ahead_coverage, reason)

        preferred = min(np.flatnonzero(viable), key=lambda i: (abs(i - 2), abs(i - (self.previous if self.previous is not None else 2))))
        if self.previous is not None and viable[self.previous] and preferred != self.previous:
            if self.pending != preferred:
                self.pending, self.pending_since = preferred, now
            if (now - self.pending_since) * 1000 < self.debounce_path_choice_ms:
                preferred = self.previous
            else:
                self.pending = None
        else:
            self.pending = self.pending_since = None
        self.previous = int(preferred)
        return Decision("direction", self.previous, near, ahead, unknown,
                        near_coverage, ahead_coverage, "clear corridor selected")
