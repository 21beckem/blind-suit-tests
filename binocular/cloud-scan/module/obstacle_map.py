"""Bounded, gravity-aligned 2.5D obstacle evidence grid."""

import math

import numpy as np

from .map_geometry import camera_to_gravity, gravity_to_world_xz, sample_points


class ObstacleMap:
    """World X right, Z forward, Y up; unknown is the default cell state.

    Each pixel is first reduced to a per-frame cell observation. Three obstacle
    observations among five integration frames promote persistent evidence;
    subsequent observations strengthen it. Bare-floor evidence accrues more
    slowly and cannot erase an obstacle in one frame. No unseen ray is carved.
    """

    UNKNOWN, FREE, OBSTACLE, DANGER = range(4)

    def __init__(self, config):
        self.settings = config["map"]
        s = self.settings
        self.cell_m = float(s["cell_m"])
        self.side = int(round(s["size_m"] / self.cell_m))
        if self.side < 2 or self.cell_m <= 0:
            raise ValueError("map size and cell size must be positive")
        self.origin_x_m = -self.side * self.cell_m / 2
        self.origin_z_m = -self.side * self.cell_m / 2
        shape = (self.side, self.side)
        self.max_height = np.full(shape, -np.inf, np.float32)
        self.min_height = np.full(shape, np.inf, np.float32)
        self.hits = np.zeros(shape, np.float32)
        self.floor_hits = np.zeros(shape, np.float32)
        self.danger_hits = np.zeros(shape, np.float32)
        self.last_seen = np.full(shape, -np.inf, np.float64)
        self._recent_obstacles = np.zeros((s["persistence_frames"], *shape), np.uint8)
        self._history_index = 0
        self._last_update_s = None
        self._now_s = 0.0
        self.pose = None
        self.tracking_ok = False
        self.integrated_frames = 0

    def _shift(self, array, dx, dz, fill):
        shifted = np.roll(array, (-dz, -dx), axis=(-2, -1))
        if abs(dx) >= self.side or abs(dz) >= self.side:
            shifted[...] = fill
            return shifted
        if dx > 0:
            shifted[..., -dx:] = fill
        elif dx < 0:
            shifted[..., :-dx] = fill
        if dz > 0:
            shifted[..., -dz:, :] = fill
        elif dz < 0:
            shifted[..., :-dz, :] = fill
        return shifted

    def _recenter(self, pose):
        margin = self.settings["recenter_margin_m"]
        cx = (pose.x_m - self.origin_x_m) / self.cell_m
        cz = (pose.z_m - self.origin_z_m) / self.cell_m
        limit = margin / self.cell_m
        dx = int(round(cx - self.side / 2)) if cx < limit or cx > self.side - limit else 0
        dz = int(round(cz - self.side / 2)) if cz < limit or cz > self.side - limit else 0
        if dx == 0 and dz == 0:
            return
        for name, fill in (("max_height", -np.inf), ("min_height", np.inf),
                           ("hits", 0), ("floor_hits", 0), ("danger_hits", 0),
                           ("last_seen", -np.inf), ("_recent_obstacles", 0)):
            setattr(self, name, self._shift(getattr(self, name), dx, dz, fill))
        self.origin_x_m += dx * self.cell_m
        self.origin_z_m += dz * self.cell_m

    def _decay(self, timestamp_s):
        if self._last_update_s is not None:
            dt = timestamp_s - self._last_update_s
            if dt < -1e-6:
                raise ValueError("map timestamps must be monotonic")
            factor = math.exp(-max(dt, 0) / self.settings["evidence_decay_s"])
            self.hits *= factor
            self.floor_hits *= factor
            self.danger_hits *= factor
        self._last_update_s = timestamp_s
        self._now_s = timestamp_s

    def integrate(self, result, floor, pose, timestamp_s):
        """Integrate one observation, or only advance decay when pose/floor is unsafe."""
        self._decay(timestamp_s)
        self.pose = pose
        self.tracking_ok = bool(pose is not None and pose.tracking_ok)
        if not floor.usable or not self.tracking_ok:
            return {"integrated": False, "obstacle_cells": 0, "floor_cells": 0,
                    "danger_cells": 0, "far_hints": 0}
        self._recenter(pose)
        s = self.settings
        p_cam, _, _ = sample_points(result, s["sample_stride"], s["min_range_m"],
                                     s["far_hint_range_m"])
        self._recent_obstacles[self._history_index] = 0
        self._history_index = (self._history_index + 1) % len(self._recent_obstacles)
        self.integrated_frames += 1
        if len(p_cam) == 0:
            return {"integrated": True, "obstacle_cells": 0, "floor_cells": 0,
                    "danger_cells": 0, "far_hints": 0}
        p_grav = camera_to_gravity(p_cam, floor.rotation_grav_cam, floor.height_m)
        h = p_grav[:, 1]
        f_times_b = float(result.Q[2, 3] / result.Q[3, 2])
        sigma_z = s["disparity_noise_px"] * p_cam[:, 2] ** 2 / f_times_b
        min_h = np.maximum(s["obstacle_min_height_m"],
                           s["obstacle_sigma_multiplier"] * sigma_z)
        trusted = p_cam[:, 2] <= s["max_range_m"]
        obstacle = trusted & (h >= min_h) & (h <= s["obstacle_max_height_m"])
        floor_point = trusted & (np.abs(h) <= s["floor_height_tol_m"])
        danger_threshold = np.maximum(s["hole_min_below_m"],
                                      s["danger_sigma_multiplier"] * sigma_z)
        danger = trusted & (h <= -danger_threshold)
        far_hints = int(np.count_nonzero(~trusted & (h >= min_h) &
                                          (h <= s["obstacle_max_height_m"])))
        x_world, z_world = gravity_to_world_xz(p_grav, pose)
        ix = np.floor((x_world - self.origin_x_m) / self.cell_m).astype(np.int32)
        iz = np.floor((z_world - self.origin_z_m) / self.cell_m).astype(np.int32)
        inside = (ix >= 0) & (ix < self.side) & (iz >= 0) & (iz < self.side)
        flat = iz * self.side + ix
        obs_flat = np.unique(flat[inside & obstacle])
        floor_flat = np.unique(flat[inside & floor_point])
        danger_flat = np.unique(flat[inside & danger])
        floor_flat = np.setdiff1d(floor_flat, np.union1d(obs_flat, danger_flat),
                                  assume_unique=True)
        if len(obs_flat):
            raw = flat[inside & obstacle]
            height = h[inside & obstacle]
            np.maximum.at(self.max_height.ravel(), raw, height)
            np.minimum.at(self.min_height.ravel(), raw, height)
            slot = (self._history_index - 1) % len(self._recent_obstacles)
            self._recent_obstacles[slot].ravel()[obs_flat] = 1
            recent = self._recent_obstacles.reshape(len(self._recent_obstacles), -1)[:, obs_flat].sum(axis=0)
            confirmed = recent >= s["persistence_required"]
            old_hits = self.hits.ravel()[obs_flat]
            self.hits.ravel()[obs_flat] = np.where(
                confirmed,
                s["obstacle_evidence_cap"], old_hits)
            self.floor_hits.ravel()[obs_flat] *= 0.5
        if len(floor_flat):
            self.floor_hits.ravel()[floor_flat] += s["floor_evidence_per_frame"]
            self.hits.ravel()[floor_flat] = np.maximum(
                0, self.hits.ravel()[floor_flat] - s["free_clears_obstacle_per_frame"])
            self.danger_hits.ravel()[floor_flat] *= 0.5
        if len(danger_flat):
            self.danger_hits.ravel()[danger_flat] += 1
            self.floor_hits.ravel()[danger_flat] = 0
        np.minimum(self.floor_hits, s["floor_evidence_cap"], out=self.floor_hits)
        np.minimum(self.hits, s["obstacle_evidence_cap"], out=self.hits)
        for indices in (obs_flat, floor_flat, danger_flat):
            self.last_seen.ravel()[indices] = timestamp_s
        return {"integrated": True, "obstacle_cells": len(obs_flat),
                "floor_cells": len(floor_flat), "danger_cells": len(danger_flat),
                "far_hints": far_hints}

    def state_grid(self):
        """0 unknown, 1 free, 2 obstacle; 3 is an unknown drop-off warning."""
        s = self.settings
        recent = self._recent_obstacles.sum(axis=0)
        age_ok = self._now_s - self.last_seen <= s["evidence_ttl_s"]
        states = np.zeros(self.hits.shape, np.uint8)
        danger = age_ok & (self.danger_hits >= s["danger_evidence_required"])
        obstacle = age_ok & (self.hits >= s["obstacle_evidence_required"])
        free = (age_ok & (self.floor_hits >= s["floor_evidence_required"]) &
                (self.hits < 0.5) & (recent == 0) & ~danger)
        states[free] = self.FREE
        states[obstacle] = self.OBSTACLE
        states[danger] = self.DANGER
        return states

    def cell_state(self, x_m, z_m):
        ix = int(math.floor((x_m - self.origin_x_m) / self.cell_m))
        iz = int(math.floor((z_m - self.origin_z_m) / self.cell_m))
        if not (0 <= ix < self.side and 0 <= iz < self.side):
            return "unknown"
        return ("unknown", "free", "obstacle", "unknown")[self.state_grid()[iz, ix]]

    def nearest_obstacle(self, direction_deg=0.0, width_deg=20.0):
        """Distance to known obstacle; None means no known hit, not clear space."""
        if self.pose is None:
            return None
        iz, ix = np.nonzero(self.state_grid() == self.OBSTACLE)
        if len(ix) == 0:
            return None
        dx = self.origin_x_m + (ix + 0.5) * self.cell_m - self.pose.x_m
        dz = self.origin_z_m + (iz + 0.5) * self.cell_m - self.pose.z_m
        distance = np.hypot(dx, dz)
        bearing = np.rad2deg(np.arctan2(dx, dz) - self.pose.yaw_rad)
        bearing = (bearing + 180) % 360 - 180
        delta = (bearing - direction_deg + 180) % 360 - 180
        selected = np.abs(delta) <= width_deg / 2
        return float(distance[selected].min()) if np.any(selected) else None

    def clearance_sectors(self):
        """Known-obstacle distances in 0.8 m wide left/front/right corridors."""
        if self.pose is None:
            return {name: None for name in ("left", "front", "right")}
        iz, ix = np.nonzero(self.state_grid() == self.OBSTACLE)
        if not len(ix):
            return {name: None for name in ("left", "front", "right")}
        dx = self.origin_x_m + (ix + 0.5) * self.cell_m - self.pose.x_m
        dz = self.origin_z_m + (iz + 0.5) * self.cell_m - self.pose.z_m
        values = {}
        for name, degrees in (("left", -45), ("front", 0), ("right", 45)):
            angle = self.pose.yaw_rad + math.radians(degrees)
            forward = dx * math.sin(angle) + dz * math.cos(angle)
            lateral = dx * math.cos(angle) - dz * math.sin(angle)
            selected = (forward > 0) & (np.abs(lateral) <=
                                        self.settings["clearance_width_m"] / 2)
            values[name] = float(forward[selected].min()) if np.any(selected) else None
        return values

    def blind_zone(self):
        return {"near_m": self.settings["min_range_m"],
                "below_frame_unknown": True, "tracking_ok": self.tracking_ok}
