"""Bounded, gravity-aligned 2.5D obstacle evidence grid."""

import math

import cv2
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
        self.last_floor_seen = np.full(shape, -np.inf, np.float64)
        self.last_danger_seen = np.full(shape, -np.inf, np.float64)
        self.last_traversed = np.full(shape, -np.inf, np.float64)
        self._recent_obstacles = np.zeros((s["persistence_frames"], *shape), np.uint8)
        self._recent_floor = np.zeros_like(self._recent_obstacles)
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
                           ("last_seen", -np.inf), ("last_floor_seen", -np.inf),
                           ("last_danger_seen", -np.inf),
                           ("last_traversed", -np.inf),
                           ("_recent_obstacles", 0),
                           ("_recent_floor", 0)):
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
            self.floor_hits *= math.exp(
                -max(dt, 0) / self.settings["floor_evidence_decay_s"])
            self.danger_hits *= math.exp(-max(dt, 0) / self.settings["danger_decay_s"])
        self._last_update_s = timestamp_s
        self._now_s = timestamp_s

    def _carve_obstacle_rays(self, p_grav, floor_mask, cols, image_width,
                             pose, camera_height_m, current_obstacles):
        """Reduce old obstacle evidence only where solid floor rays cross its height.

        A floor hit proves that its 3D camera-to-hit ray was clear. It does not
        prove the ground beneath the ray was bare, so this never adds free
        evidence. Height overlap prevents an overhead ray from erasing a low
        box or tabletop that it could have passed above.
        """
        s = self.settings
        candidate = np.flatnonzero(floor_mask)
        if len(candidate) == 0:
            return 0
        bins = s["ray_bearing_bins"]
        bearing_bin = np.minimum(cols[candidate] * bins // image_width, bins - 1)
        chosen = []
        for bearing in np.unique(bearing_bin):
            members = candidate[bearing_bin == bearing]
            chosen.append(members[np.argmax(p_grav[members, 2])])
        endpoints = p_grav[chosen]
        end_x, end_z = gravity_to_world_xz(endpoints, pose)
        dx, dz = end_x - pose.x_m, end_z - pose.z_m
        lengths = np.hypot(dx, dz)
        step = np.arange(1, int(np.ceil(s["max_range_m"] / self.cell_m)) + 1,
                         dtype=np.float32) * self.cell_m
        fraction = step[None, :] / np.maximum(lengths[:, None], self.cell_m)
        valid = fraction < 1 - s["ray_endpoint_margin_fraction"]
        x = pose.x_m + fraction * dx[:, None]
        z = pose.z_m + fraction * dz[:, None]
        ix = np.floor((x - self.origin_x_m) / self.cell_m).astype(np.int32)
        iz = np.floor((z - self.origin_z_m) / self.cell_m).astype(np.int32)
        valid &= (ix >= 0) & (ix < self.side) & (iz >= 0) & (iz < self.side)
        if not np.any(valid):
            return 0
        flat = iz[valid] * self.side + ix[valid]
        ray_h = (1 - fraction) * camera_height_m + fraction * endpoints[:, 1, None]
        ray_h = ray_h[valid]
        height_margin = s["ray_height_margin_m"]
        min_h = self.min_height.ravel()[flat]
        max_h = self.max_height.ravel()[flat]
        current_mask = np.zeros(self.side * self.side, bool)
        current_mask[current_obstacles] = True
        clear = ((self.hits.ravel()[flat] > 0) &
                 (ray_h >= min_h - height_margin) &
                 (ray_h <= max_h + height_margin) &
                 ~current_mask[flat])
        cells = np.unique(flat[clear])
        if len(cells):
            self.hits.ravel()[cells] = np.maximum(
                0, self.hits.ravel()[cells] - s["ray_clear_obstacle_per_frame"])
            cleared = cells[self.hits.ravel()[cells] == 0]
            self.max_height.ravel()[cleared] = -np.inf
            self.min_height.ravel()[cleared] = np.inf
        return len(cells)

    def _clear_traversed_to_unknown(self, pose, timestamp_s, current_obstacles):
        """A traversed body footprint rules out an old obstacle, not unseen floor."""
        radius = self.settings["traversal_clear_radius_m"]
        center_x = int((pose.x_m - self.origin_x_m) / self.cell_m)
        center_z = int((pose.z_m - self.origin_z_m) / self.cell_m)
        cells = int(math.ceil(radius / self.cell_m))
        x0, x1 = max(0, center_x - cells), min(self.side, center_x + cells + 1)
        z0, z1 = max(0, center_z - cells), min(self.side, center_z + cells + 1)
        iz, ix = np.mgrid[z0:z1, x0:x1]
        x = self.origin_x_m + (ix + 0.5) * self.cell_m
        z = self.origin_z_m + (iz + 0.5) * self.cell_m
        mask = (x - pose.x_m) ** 2 + (z - pose.z_m) ** 2 <= radius ** 2
        flat = (iz * self.side + ix)[mask]
        if len(current_obstacles):
            flat = np.setdiff1d(flat, current_obstacles, assume_unique=False)
        self.hits.ravel()[flat] = 0
        self.danger_hits.ravel()[flat] = 0
        self.max_height.ravel()[flat] = -np.inf
        self.min_height.ravel()[flat] = np.inf
        self.last_traversed.ravel()[flat] = timestamp_s
        return len(flat)

    def _suppress_flat_obstacle_fragments(self, obstacle, heights_m, rows, cols,
                                          image_shape):
        """Reject tiny horizontal stereo aliases on patterned floors to unknown.

        Apply this to image-space components before temporal fusion. The sampled
        pixels remain neither floor nor free; only obstacle evidence is removed.
        A tall thin component or a high horizontal tabletop remains eligible.
        """
        if not np.any(obstacle):
            return obstacle
        s = self.settings
        stride = s["sample_stride"]
        shape = tuple((dimension + stride - 1) // stride for dimension in image_shape)
        sampled = np.zeros(shape, np.uint8)
        sampled[rows[obstacle] // stride, cols[obstacle] // stride] = 1
        count, labels, stats, _ = cv2.connectedComponentsWithStats(sampled, 8)
        if count <= 1:
            return obstacle
        component = labels[rows[obstacle] // stride, cols[obstacle] // stride]
        component_height = np.full(count, -np.inf, np.float32)
        np.maximum.at(component_height, component, heights_m[obstacle])
        low_flat = ((stats[:, cv2.CC_STAT_AREA] <
                     s["artifact_max_component_samples"]) &
                    (stats[:, cv2.CC_STAT_HEIGHT] <=
                     s["artifact_max_component_rows"]) &
                    (component_height < s["artifact_max_height_m"]))
        low_flat[0] = False
        filtered = obstacle.copy()
        filtered[np.flatnonzero(obstacle)] &= ~low_flat[component]
        return filtered

    def _retain_supported_dropoffs(self, danger, rows, cols, image_shape):
        """Tiny below-floor stereo islands remain unknown, not drop-off alarms."""
        if not np.any(danger):
            return danger
        stride = self.settings["sample_stride"]
        shape = tuple((dimension + stride - 1) // stride for dimension in image_shape)
        sampled = np.zeros(shape, np.uint8)
        sampled[rows[danger] // stride, cols[danger] // stride] = 1
        _, labels, stats, _ = cv2.connectedComponentsWithStats(sampled, 8)
        component = labels[rows[danger] // stride, cols[danger] // stride]
        supported = stats[:, cv2.CC_STAT_AREA] >= self.settings[
            "danger_min_component_samples"]
        supported[0] = False
        filtered = danger.copy()
        filtered[np.flatnonzero(danger)] &= supported[component]
        return filtered

    def integrate(self, result, floor, pose, timestamp_s):
        """Integrate one observation, or only advance decay when pose/floor is unsafe."""
        self._decay(timestamp_s)
        self.pose = pose
        self.tracking_ok = bool(pose is not None and pose.tracking_ok)
        if not floor.usable or not self.tracking_ok:
            return {"integrated": False, "obstacle_cells": 0, "floor_cells": 0,
                    "danger_cells": 0, "far_hints": 0, "carved_cells": 0,
                    "traversed_cells": 0}
        self._recenter(pose)
        s = self.settings
        p_cam, sample_rows, cols = sample_points(result, s["sample_stride"], s["min_range_m"],
                                       s["far_hint_range_m"])
        self._recent_obstacles[self._history_index] = 0
        self._recent_floor[self._history_index] = 0
        self._history_index = (self._history_index + 1) % len(self._recent_obstacles)
        self.integrated_frames += 1
        if len(p_cam) == 0:
            traversed = self._clear_traversed_to_unknown(
                pose, timestamp_s, np.empty(0, np.int64))
            return {"integrated": True, "obstacle_cells": 0, "floor_cells": 0,
                    "danger_cells": 0, "far_hints": 0, "carved_cells": 0,
                    "traversed_cells": traversed}
        p_grav = camera_to_gravity(p_cam, floor.rotation_grav_cam, floor.height_m)
        h = p_grav[:, 1]
        f_times_b = float(result.Q[2, 3] / result.Q[3, 2])
        sigma_z = s["disparity_noise_px"] * p_cam[:, 2] ** 2 / f_times_b
        min_h = np.maximum(s["obstacle_min_height_m"],
                           s["obstacle_sigma_multiplier"] * sigma_z)
        trusted = p_cam[:, 2] <= s["max_range_m"]
        obstacle = trusted & (h >= min_h) & (h <= s["obstacle_max_height_m"])
        obstacle = self._suppress_flat_obstacle_fragments(
            obstacle, h, sample_rows, cols, result.disparity.shape)
        floor_point = trusted & (np.abs(h) <= s["floor_height_tol_m"])
        danger_threshold = np.maximum(s["hole_min_below_m"],
                                      s["danger_sigma_multiplier"] * sigma_z)
        danger = trusted & (h <= -danger_threshold)
        danger = self._retain_supported_dropoffs(
            danger, sample_rows, cols, result.disparity.shape)
        far_hints = int(np.count_nonzero(~trusted & (h >= min_h) &
                                          (h <= s["obstacle_max_height_m"])))
        x_world, z_world = gravity_to_world_xz(p_grav, pose)
        ix = np.floor((x_world - self.origin_x_m) / self.cell_m).astype(np.int32)
        iz = np.floor((z_world - self.origin_z_m) / self.cell_m).astype(np.int32)
        inside = (ix >= 0) & (ix < self.side) & (iz >= 0) & (iz < self.side)
        flat = iz * self.side + ix
        obs_flat = np.unique(flat[inside & obstacle])
        # Several independent solid pixels in the same cell constitute a
        # stronger floor observation than a single boundary/noise pixel.
        floor_counts = np.bincount(flat[inside & floor_point],
                                   minlength=self.side * self.side)
        floor_flat = np.flatnonzero(floor_counts)
        danger_flat = np.unique(flat[inside & danger])
        floor_flat = np.setdiff1d(floor_flat, np.union1d(obs_flat, danger_flat),
                                  assume_unique=True)
        if not floor.fresh:
            floor_flat = floor_flat[:0]
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
            dense = floor_counts[floor_flat] >= s["floor_dense_min_samples"]
            self.floor_hits.ravel()[floor_flat] += np.where(
                dense, s["floor_evidence_per_frame"],
                s["floor_sparse_evidence_per_frame"])
            self.last_floor_seen.ravel()[floor_flat] = timestamp_s
            self.hits.ravel()[floor_flat] = np.maximum(
                0, self.hits.ravel()[floor_flat] - s["free_clears_obstacle_per_frame"])
            self.danger_hits.ravel()[floor_flat] *= 0.5
            slot = (self._history_index - 1) % len(self._recent_floor)
            self._recent_floor[slot].ravel()[floor_flat[dense]] = 1
            recent_floor = self._recent_floor.reshape(len(self._recent_floor), -1)[:, floor_flat].sum(axis=0)
            recent_obstacles = self._recent_obstacles.reshape(len(self._recent_obstacles), -1)[:, floor_flat].sum(axis=0)
            reconciled = (recent_floor >= s["floor_override_required"]) & (recent_obstacles == 0)
            self.hits.ravel()[floor_flat[reconciled]] = 0
            self.danger_hits.ravel()[floor_flat[reconciled]] = 0
            self.max_height.ravel()[floor_flat[reconciled]] = -np.inf
            self.min_height.ravel()[floor_flat[reconciled]] = np.inf
        if len(danger_flat):
            self.danger_hits.ravel()[danger_flat] += 1
            self.floor_hits.ravel()[danger_flat] = 0
            self.last_danger_seen.ravel()[danger_flat] = timestamp_s
        np.minimum(self.floor_hits, s["floor_evidence_cap"], out=self.floor_hits)
        np.minimum(self.hits, s["obstacle_evidence_cap"], out=self.hits)
        for indices in (obs_flat, floor_flat, danger_flat):
            self.last_seen.ravel()[indices] = timestamp_s
        carved_cells = (self._carve_obstacle_rays(
            p_grav, floor_point, cols, result.disparity.shape[1], pose,
            floor.height_m, obs_flat)
                        if floor.fresh else 0)
        traversed_cells = self._clear_traversed_to_unknown(pose, timestamp_s, obs_flat)
        return {"integrated": True, "obstacle_cells": len(obs_flat),
                "floor_cells": len(floor_flat), "danger_cells": len(danger_flat),
                "far_hints": far_hints, "carved_cells": carved_cells,
                "traversed_cells": traversed_cells}

    def state_grid(self):
        """0 unknown, 1 free, 2 obstacle; 3 is an unknown drop-off warning."""
        s = self.settings
        recent = self._recent_obstacles.sum(axis=0)
        age_ok = self._now_s - self.last_seen <= s["evidence_ttl_s"]
        floor_age_ok = self._now_s - self.last_floor_seen <= s["floor_evidence_ttl_s"]
        states = np.zeros(self.hits.shape, np.uint8)
        danger = ((self._now_s - self.last_danger_seen <= s["danger_ttl_s"]) &
                  (self.danger_hits >= s["danger_evidence_required"]))
        height_supported = ((self.max_height >= s["obstacle_high_surface_m"]) |
                            (self.max_height - self.min_height >=
                             s["obstacle_min_vertical_span_m"]))
        obstacle = (age_ok & (self.hits >= s["obstacle_evidence_required"]) &
                    height_supported)
        free = (floor_age_ok & (self.floor_hits >= s["floor_evidence_required"]) &
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

    def walked_path_mask(self):
        """Recent tracked body footprints, separate from FREE cell evidence."""
        return self._now_s - self.last_traversed <= self.settings[
            "traversal_display_ttl_s"]

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

    def forward_coverage(self, length_m, width_m):
        """Count observed and unknown cells in a camera-relative corridor."""
        if self.pose is None:
            return None
        iz, ix = np.indices(self.hits.shape)
        dx = self.origin_x_m + (ix + 0.5) * self.cell_m - self.pose.x_m
        dz = self.origin_z_m + (iz + 0.5) * self.cell_m - self.pose.z_m
        c, s = math.cos(self.pose.yaw_rad), math.sin(self.pose.yaw_rad)
        forward = s * dx + c * dz
        lateral = c * dx - s * dz
        region = (forward >= 0) & (forward < length_m) & (
            np.abs(lateral) < width_m / 2)
        states = self.state_grid()
        measured_floor = region & (self.floor_hits > 0)
        recent_obstacle = self._recent_obstacles.sum(axis=0) > 0
        return {"cells": int(np.count_nonzero(region)),
                "floor_evidence_cells": int(np.count_nonzero(measured_floor)),
                "floor_with_obstacle_evidence": int(np.count_nonzero(
                    measured_floor & (self.hits >= 0.5))),
                "floor_with_recent_obstacle": int(np.count_nonzero(
                    measured_floor & recent_obstacle)),
                "floor_with_danger": int(np.count_nonzero(
                    measured_floor & (states == self.DANGER))),
                "free": int(np.count_nonzero(region & (states == self.FREE))),
                "obstacle": int(np.count_nonzero(region & (states == self.OBSTACLE))),
                "danger_unknown": int(np.count_nonzero(region & (states == self.DANGER))),
                "unknown": int(np.count_nonzero(region & (states == self.UNKNOWN)))}
