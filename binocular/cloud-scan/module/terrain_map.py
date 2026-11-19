"""Height-aware rolling terrain map built from independently evidenced surfaces.

The old obstacle grid is retained as the evidence engine for one horizontal
surface. A terrain map owns one such grid per observed elevation and a small
graph of supported step transitions. A horizontal surface is not reachable
merely because it was seen: only confirmed links connect it to the wearer.
"""

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .map_geometry import camera_to_gravity, gravity_to_world_xz, sample_points
from .obstacle_map import ObstacleMap


@dataclass
class SurfaceLevel:
    id: int
    height_m: float
    grid: ObstacleMap
    observations: int = 0
    last_seen_s: float = -math.inf
    # Retained only as tentative geometry after a live-grid reset. It never
    # contributes free cells or confirms a step without fresh observations.
    support_memory: OrderedDict = field(default_factory=OrderedDict)


@dataclass
class StepLink:
    lower_id: int
    upper_id: int
    observations: int = 0
    last_seen_s: float = -math.inf


class TerrainMap:
    """Rolling, level-aware map with the ObstacleMap query surface.

    World Y is anchored at the first supported ground plane in each tracking
    epoch. Evidence at distinct heights uses distinct grids, so floor and ray
    clearing cannot erase another level. The two step states in the projected
    view describe a *transition*, not automatically safe passage.
    """

    UNKNOWN, FREE, OBSTACLE, DANGER, REMEMBERED = range(5)
    STEP_UP, STEP_DOWN = 5, 6

    def __init__(self, config):
        self.settings = config["map"]
        self._config = config
        self._epochs = {0: ({}, {})}
        self._epoch_offsets = {}
        self._epoch = 0
        self._levels, self._links = self._epochs[0]
        self._next_id = 0
        self._adjacent_pairs = set()
        self._active_id = None
        self._base_id = None
        self._initial_camera_offset_m = None
        self._pending_id = None
        self._pending_count = 0
        self._pending_surfaces = []
        self._pose_history = deque()
        self.pose = None
        self.tracking_ok = False
        self.current_level_known = False
        self.integrated_frames = 0
        self.last_candidates = 0
        self.last_confirmed_links = 0
        self.last_rejection = None
        self.vertical_correction_m = 0.0
        self.phase_ms = {}
        self._now_s = 0.0
        self._memory_evictions = 0
        self._base = ObstacleMap(config)

    @property
    def _active(self):
        return (self._levels[self._active_id].grid
                if self._active_id in self._levels else self._base)

    @property
    def cell_m(self):
        return self._active.cell_m

    @property
    def side(self):
        return self._active.side

    @property
    def origin_x_m(self):
        return self._active.origin_x_m

    @property
    def origin_z_m(self):
        return self._active.origin_z_m

    @property
    def floor_hits(self):
        return self._active.floor_hits

    @property
    def remembered_cells(self):
        return sum(level.grid.remembered_cells for level in self._levels.values())

    @property
    def session_memory_cells(self):
        return sum(level.grid.session_memory_cells for levels, _ in self._epochs.values()
                   for level in levels.values())

    @property
    def memory_evictions(self):
        return self._memory_evictions + sum(
            level.grid.memory_evictions for levels, _ in self._epochs.values()
            for level in levels.values())

    @property
    def surface_count(self):
        return len(self._levels)

    @property
    def current_level_height_m(self):
        return (self._levels[self._active_id].height_m
                if self._active_id in self._levels else None)

    @property
    def confirmed_step_count(self):
        return sum(self._link_confirmed(link)
                   for link in self._links.values())

    def _link_confirmed(self, link):
        if link.observations < self.settings["stair_link_frames"]:
            return False
        rise = (self._levels[link.upper_id].height_m -
                self._levels[link.lower_id].height_m)
        return ((link.lower_id, link.upper_id) in self._adjacent_pairs and
                self.settings["stair_min_rise_m"] <= rise <=
                self.settings["stair_max_rise_m"])

    def _refresh_adjacency(self):
        ordered = sorted(self._levels.values(), key=lambda level: level.height_m)
        self._adjacent_pairs = {(low.id, high.id)
                                for low, high in zip(ordered, ordered[1:])}

    def reset_local(self, epoch=None):
        """Select a tracking epoch and clear only its short-lived grid evidence."""
        self._epoch = self._epoch if epoch is None else epoch
        if self._epoch not in self._epochs:
            self._epochs[self._epoch] = ({}, {})
        self._levels, self._links = self._epochs[self._epoch]
        self._refresh_adjacency()
        for level in self._levels.values():
            level.grid.reset_local(self._epoch)
        self._base = ObstacleMap(self._config)
        self._active_id = None
        self._base_id = min(self._levels) if self._levels else None
        self._initial_camera_offset_m = self._epoch_offsets.get(self._epoch)
        self._pending_id = None
        self._pending_count = 0
        self._pending_surfaces = []
        self._pose_history.clear()
        self.current_level_known = False

    def _new_level(self, height_m):
        level = SurfaceLevel(self._next_id, float(height_m),
                             ObstacleMap(self._config))
        self._levels[level.id] = level
        self._next_id += 1
        return level

    @staticmethod
    def _footprint_ok(x, z, width_m, depth_m, min_points):
        if len(x) < min_points:
            return False
        centered = np.column_stack((x - np.mean(x), z - np.mean(z)))
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ basis.T
        spans = np.sort(np.ptp(projected, axis=0))
        return spans[1] >= width_m and spans[0] >= depth_m

    def _surface_fill(self, x, z):
        """Occupied fraction of a horizontal patch's oriented footprint."""
        centered = np.column_stack((x - np.mean(x), z - np.mean(z)))
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ basis.T
        cell = self.cell_m
        indices = np.floor((projected - np.min(projected, axis=0)) /
                           cell).astype(np.int32)
        extents = np.max(indices, axis=0) + 1
        occupied = np.unique(indices[:, 0] * extents[1] + indices[:, 1]).size
        return occupied / max(1, int(extents[0] * extents[1]))

    def _supported_component(self, mode, x_world, z_world, depth_m):
        """Keep a filled tread patch instead of combining scattered aliases."""
        s = self.settings
        ix = np.floor((x_world - self.origin_x_m) / self.cell_m).astype(np.int32)
        iz = np.floor((z_world - self.origin_z_m) / self.cell_m).astype(np.int32)
        inside = (mode & (ix >= 0) & (ix < self.side) &
                  (iz >= 0) & (iz < self.side))
        if np.count_nonzero(inside) < s["stair_surface_min_points"]:
            return None
        occupied = np.zeros((self.side, self.side), np.uint8)
        occupied[iz[inside], ix[inside]] = 1
        connected = cv2.dilate(occupied, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            connected, connectivity=8)
        for component in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1:
            selected = inside & (labels[np.clip(iz, 0, self.side - 1),
                                        np.clip(ix, 0, self.side - 1)] == component)
            x, z = x_world[selected], z_world[selected]
            if not self._footprint_ok(
                    x, z, s["stair_min_tread_width_m"], depth_m,
                    s["stair_surface_min_points"]):
                continue
            if self._surface_fill(x, z) >= s[
                    "stair_surface_min_fill_fraction"]:
                return selected
        return None

    def _surface_candidates(self, heights, x_world, z_world, solid,
                            foot_height=None):
        """Find repeated horizontal height modes with a tread-sized footprint."""
        s = self.settings
        trusted = solid & np.isfinite(heights)
        if foot_height is not None:
            trusted &= np.abs(heights - foot_height) <= s[
                "stair_vertical_window_m"]
        if np.count_nonzero(trusted) < s["stair_surface_min_points"]:
            return []
        quantized = np.rint(heights[trusted] / s[
            "stair_height_histogram_m"]).astype(np.int32)
        bins, counts = np.unique(quantized, return_counts=True)
        order = np.argsort(counts)[::-1]
        candidates = []
        for index in order:
            if len(candidates) >= s["stair_max_candidates"]:
                break
            center = bins[index] * s["stair_height_histogram_m"]
            if any(abs(center - value[0]) < s["stair_level_separation_m"]
                   for value in candidates):
                continue
            mode = trusted & (np.abs(heights - center) <= s[
                "stair_surface_tolerance_m"])
            if (np.count_nonzero(mode) < s["stair_surface_min_points"] or
                    self._surface_fill(x_world[mode], z_world[mode]) < s[
                        "stair_surface_min_fill_fraction"]):
                continue
            depth_m = (s["stair_min_down_tread_depth_m"]
                       if foot_height is not None and
                       center < foot_height - s["stair_min_rise_m"]
                       else s["stair_min_tread_depth_m"])
            mask = self._supported_component(mode, x_world, z_world, depth_m)
            if mask is None:
                continue
            candidates.append((float(np.median(heights[mask])), mask))
        return candidates

    @staticmethod
    def _candidate_cells(level, mask, x_world, z_world):
        grid = level.grid
        ix = np.floor((x_world[mask] - grid.origin_x_m) /
                      grid.cell_m).astype(np.int32)
        iz = np.floor((z_world[mask] - grid.origin_z_m) /
                      grid.cell_m).astype(np.int32)
        inside = ((ix >= 0) & (ix < grid.side) &
                  (iz >= 0) & (iz < grid.side))
        return np.unique(iz[inside] * grid.side + ix[inside])

    def _spatial_overlap(self, level, mask, x_world, z_world):
        cells = self._candidate_cells(level, mask, x_world, z_world)
        if not len(cells):
            return 0, 0.0
        support = self._support_mask(level).ravel()[cells]
        return int(np.count_nonzero(support)), float(np.mean(support))

    def refine_pose(self, result, floor, pose):
        """Anchor visual Y to a previously observed *same* surface patch.

        Spatial overlap is essential: an adjacent tread is a true height
        change, whereas a reobserved patch at a shifted Y is odometry drift.
        """
        s = self.settings
        self.vertical_correction_m = 0.0
        if self.integrated_frames % s["stair_vertical_refine_interval_frames"]:
            return pose
        if (not self._levels or pose is None or not pose.tracking_ok or
                not floor.fresh or self._initial_camera_offset_m is None):
            return pose
        points, rows, cols = sample_points(
            result, s["sample_stride"], s["min_range_m"],
            s["max_range_m"], min_confidence=2)
        if not len(points):
            return pose
        grav = camera_to_gravity(points, floor.rotation_grav_cam, pose.y_m)
        x_world, z_world = gravity_to_world_xz(grav, pose)
        foot = pose.y_m - self._initial_camera_offset_m
        candidates = self._surface_candidates(
            grav[:, 1], x_world, z_world,
            np.ones(len(points), bool), foot)
        best = None
        for height, mask in candidates:
            for level in self._levels.values():
                error = height - level.height_m
                if abs(error) > s["stair_vertical_correction_max_error_m"]:
                    continue
                overlap, fraction = self._spatial_overlap(
                    level, mask, x_world, z_world)
                if (overlap < s["stair_vertical_correction_min_cells"] or
                        fraction < s["stair_vertical_correction_min_fraction"]):
                    continue
                score = overlap * fraction
                if best is None or score > best[0]:
                    best = score, error
        if best is None:
            return pose
        correction = np.clip(best[1] * s["stair_vertical_correction_gain"],
                             -s["stair_vertical_correction_max_frame_m"],
                             s["stair_vertical_correction_max_frame_m"])
        if abs(correction) < 0.002:
            return pose
        self.vertical_correction_m = float(correction)
        return replace(pose, y_m=float(pose.y_m - correction))

    def _match_candidates(self, candidates, x_world, z_world, timestamp_s):
        s = self.settings
        matched = set()
        self._pending_surfaces = [item for item in self._pending_surfaces
                                  if timestamp_s - item["last_s"] <=
                                  s["stair_pending_ttl_s"]]
        for height, mask in candidates:
            overlap_match = False
            nearest = min(self._levels.values(),
                          key=lambda level: abs(level.height_m - height))
            if abs(nearest.height_m - height) > s["stair_level_separation_m"]:
                overlap_matches = []
                for level in self._levels.values():
                    if abs(level.height_m - height) > s[
                            "stair_vertical_correction_max_error_m"]:
                        continue
                    overlap, fraction = self._spatial_overlap(
                        level, mask, x_world, z_world)
                    if (overlap >= s["stair_vertical_correction_min_cells"] and
                            fraction >= s["stair_vertical_correction_min_fraction"]):
                        overlap_matches.append((overlap * fraction, level))
                if overlap_matches:
                    nearest = max(overlap_matches, key=lambda item: item[0])[1]
                    overlap_match = True
                else:
                    pending = next((item for item in self._pending_surfaces
                                    if abs(item["height"] - height) <= s[
                                        "stair_level_separation_m"]), None)
                    if pending is None:
                        pending = {"height": height, "count": 0,
                                   "last_s": timestamp_s}
                        self._pending_surfaces.append(pending)
                    pending["count"] += 1
                    pending["last_s"] = timestamp_s
                    pending["height"] += 0.2 * (height - pending["height"])
                    if pending["count"] < s["stair_surface_frames"]:
                        continue
                    if len(self._levels) >= s["stair_max_levels"]:
                        self.last_rejection = "level_limit"
                        continue
                    nearest = self._new_level(pending["height"])
                    self._pending_surfaces.remove(pending)
            if nearest.id in matched:
                continue
            matched.add(nearest.id)
            nearest.observations += 1
            nearest.last_seen_s = timestamp_s
            # Keep the initial world datum fixed. Other levels may converge
            # slowly, but observations at a neighboring step cannot merge.
            if (nearest.id != self._base_id and not overlap_match and
                    abs(height - nearest.height_m) < 0.05):
                nearest.height_m += 0.15 * (height - nearest.height_m)
        return matched

    def _aligned_mask(self, level, mask):
        grid = level.grid
        active = self._active
        if grid is active:
            return mask
        dx = int(round((active.origin_x_m - grid.origin_x_m) / self.cell_m))
        dz = int(round((active.origin_z_m - grid.origin_z_m) / self.cell_m))
        return grid._shift(mask, dx, dz, False)

    def _support_mask(self, level, aligned=False):
        grid = level.grid
        mask = ((grid.floor_hits >= self.settings["floor_evidence_required"]) &
                (self._now_s - grid.last_floor_seen <= self.settings[
                    "floor_evidence_ttl_s"]))
        return self._aligned_mask(level, mask) if aligned else mask

    def _riser_supported(self, low_height, high_height, contact,
                         heights, x_world, z_world, solid):
        s = self.settings
        ix = np.floor((x_world - self.origin_x_m) / self.cell_m).astype(np.int32)
        iz = np.floor((z_world - self.origin_z_m) / self.cell_m).astype(np.int32)
        inside = ((ix >= 0) & (ix < self.side) &
                  (iz >= 0) & (iz < self.side))
        flat = np.clip(iz, 0, self.side - 1) * self.side + np.clip(
            ix, 0, self.side - 1)
        boundary = cv2.dilate(contact.astype(np.uint8),
                              np.ones((3, 3), np.uint8)) > 0
        face_points = (solid & inside & boundary.ravel()[flat] &
                       (heights >= low_height + s["stair_riser_margin_m"]) &
                       (heights <= high_height - s["stair_riser_margin_m"]))
        if np.count_nonzero(face_points) < s["stair_riser_min_points"]:
            return False
        face_cells = flat[face_points]
        low_by_cell = np.full(self.side * self.side, np.inf, np.float32)
        high_by_cell = np.full(self.side * self.side, -np.inf, np.float32)
        np.minimum.at(low_by_cell, face_cells, heights[face_points])
        np.maximum.at(high_by_cell, face_cells, heights[face_points])
        vertical_cells = np.flatnonzero(
            high_by_cell - low_by_cell >= s["stair_riser_vertical_span_m"])
        if len(vertical_cells) < s["stair_riser_min_cells"]:
            return False
        vz, vx = np.divmod(vertical_cells, self.side)
        return self._footprint_ok(
            self.origin_x_m + (vx + 0.5) * self.cell_m,
            self.origin_z_m + (vz + 0.5) * self.cell_m,
            s["stair_contact_width_m"], 0.0,
            s["stair_riser_min_cells"])

    def _update_links(self, seen_ids, heights, x_world, z_world, solid):
        s = self.settings
        if len(seen_ids) < 2:
            return
        # A step joins neighboring observed heights. Connecting every pair
        # creates shortcuts across treads and turns periodic stereo aliases
        # into a dense, misleading graph.
        levels = sorted((self._levels[index] for index in seen_ids),
                        key=lambda level: level.height_m)
        radius = max(1, int(math.ceil(s["stair_max_tread_gap_m"] / self.cell_m)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (radius * 2 + 1, radius * 2 + 1))
        masks = {level.id: self._support_mask(level, aligned=True)
                 for level in levels}
        dilated = {level.id: cv2.dilate(masks[level.id].astype(np.uint8),
                                      kernel) > 0 for level in levels[:-1]}
        for low, high in zip(levels, levels[1:]):
            if (low.id, high.id) not in self._adjacent_pairs:
                continue
            rise = high.height_m - low.height_m
            if not s["stair_min_rise_m"] <= rise <= s["stair_max_rise_m"]:
                continue
            low_mask, high_mask = masks[low.id], masks[high.id]
            contact = high_mask & ~low_mask & dilated[low.id]
            iz, ix = np.nonzero(contact)
            if len(ix) < s["stair_contact_min_cells"]:
                continue
            x = self.origin_x_m + (ix + 0.5) * self.cell_m
            z = self.origin_z_m + (iz + 0.5) * self.cell_m
            if not self._footprint_ok(x, z, s["stair_contact_width_m"],
                                      0.0, s["stair_contact_min_cells"]):
                continue
            if low.support_memory and high.support_memory:
                shared = len(low.support_memory.keys() & high.support_memory.keys())
                if shared / min(len(low.support_memory),
                                len(high.support_memory)) > s[
                                    "stair_max_footprint_overlap_fraction"]:
                    continue
            # A tread next to a floor can also be a box top. Require a
            # measured vertical face at the shared boundary before
            # promoting a link. The face remains a collision hazard.
            if not self._riser_supported(low.height_m, high.height_m,
                                         contact, heights, x_world, z_world,
                                         solid):
                continue
            key = (low.id, high.id)
            link = self._links.setdefault(key, StepLink(*key))
            link.observations += 1
            link.last_seen_s = self._now_s

    def _connected_ids(self):
        if not self._levels:
            return set()
        connected = {self._base_id}
        changed = True
        while changed:
            changed = False
            for link in self._links.values():
                if not self._link_confirmed(link):
                    continue
                if link.lower_id in connected and link.upper_id not in connected:
                    connected.add(link.upper_id)
                    changed = True
                elif link.upper_id in connected and link.lower_id not in connected:
                    connected.add(link.lower_id)
                    changed = True
        return connected

    def _select_current_level(self):
        if self.pose is None or self._initial_camera_offset_m is None:
            self.current_level_known = False
            return
        connected = self._connected_ids()
        if not connected:
            self.current_level_known = False
            return
        foot_height = self.pose.y_m - self._initial_camera_offset_m
        self._pose_history.append((self._now_s, self.pose.y_m))
        while (self._pose_history and self._now_s - self._pose_history[0][0] >
               self.settings["stair_switch_motion_window_s"]):
            self._pose_history.popleft()
        nearest = min(connected, key=lambda index: abs(
            self._levels[index].height_m - foot_height))
        gap = abs(self._levels[nearest].height_m - foot_height)
        if gap > self.settings["stair_current_level_tolerance_m"]:
            self.current_level_known = False
            return
        if nearest != self._active_id:
            adjacent = any(
                self._link_confirmed(link) and
                {link.lower_id, link.upper_id} == {self._active_id, nearest}
                for link in self._links.values())
            if not adjacent:
                self.current_level_known = False
                return
            target = self._levels[nearest]
            radius = int(math.ceil(self.settings[
                "stair_switch_surface_radius_m"] / self.cell_m))
            gx = int(math.floor(self.pose.x_m / self.cell_m))
            gz = int(math.floor(self.pose.z_m / self.cell_m))
            if not any(abs(cell_x - gx) <= radius and
                       abs(cell_z - gz) <= radius
                       for cell_x, cell_z in target.support_memory):
                self.current_level_known = False
                return
            if len(self._pose_history) < 2:
                self.current_level_known = False
                return
            delta_y = self.pose.y_m - self._pose_history[0][1]
            rise = target.height_m - self._levels[self._active_id].height_m
            if delta_y * math.copysign(1, rise) < self.settings[
                    "stair_switch_vertical_motion_m"]:
                self.current_level_known = False
                return
            if self._pending_id != nearest:
                self._pending_id, self._pending_count = nearest, 1
            else:
                self._pending_count += 1
            if self._pending_count >= self.settings["stair_level_switch_frames"]:
                self._active_id = nearest
                self._pending_id, self._pending_count = None, 0
        else:
            self._pending_id, self._pending_count = None, 0
        self.current_level_known = self._active_id in connected

    def _retain_support_memory(self, level):
        if level.observations < self.settings["stair_link_frames"]:
            return
        grid = level.grid
        iz, ix = np.nonzero(self._support_mask(level) &
                            (grid.last_floor_seen == self._now_s))
        for row, col in zip(iz, ix):
            key = grid._memory_key(col, row)
            if key in level.support_memory:
                continue
            level.support_memory[key] = self._now_s
        limit = self.settings["stair_surface_memory_max_cells"]
        all_levels = [item for levels, _ in self._epochs.values()
                      for item in levels.values()]
        while sum(len(item.support_memory) for item in all_levels) > limit:
            oldest = min((item for item in all_levels if item.support_memory),
                         key=lambda item: next(iter(item.support_memory.values())))
            oldest.support_memory.popitem(last=False)

    def _cap_obstacle_memory(self):
        limit = self.settings["memory_max_cells"]
        while self.session_memory_cells > limit:
            candidates = [(next(iter(memory.values()))[2], memory)
                          for levels, _ in self._epochs.values()
                          for level in levels.values()
                          for memory in level.grid._memory_epochs.values() if memory]
            if not candidates:
                break
            _, oldest = min(candidates, key=lambda item: item[0])
            oldest.popitem(last=False)
            self._memory_evictions += 1

    def _integrate_support_only(self, level, heights, x_world, z_world,
                                solid, pose, timestamp_s):
        """Fuse another level's measured tread cells without obstacle work."""
        grid = level.grid
        grid._decay(timestamp_s)
        grid._recenter(pose)
        grid.pose = pose
        grid.tracking_ok = True
        near = solid & (np.abs(heights - level.height_m) <= self.settings[
            "stair_surface_tolerance_m"])
        ix = np.floor((x_world[near] - grid.origin_x_m) /
                      grid.cell_m).astype(np.int32)
        iz = np.floor((z_world[near] - grid.origin_z_m) /
                      grid.cell_m).astype(np.int32)
        inside = ((ix >= 0) & (ix < grid.side) &
                  (iz >= 0) & (iz < grid.side))
        if not np.any(inside):
            return 0
        counts = np.bincount(iz[inside] * grid.side + ix[inside],
                             minlength=grid.side * grid.side)
        cells = np.flatnonzero(counts)
        dense = counts[cells] >= self.settings["floor_dense_min_samples"]
        grid.floor_hits.ravel()[cells] += np.where(
            dense, self.settings["floor_evidence_per_frame"],
            self.settings["floor_sparse_evidence_per_frame"])
        np.minimum(grid.floor_hits, self.settings["floor_evidence_cap"],
                   out=grid.floor_hits)
        grid.last_floor_seen.ravel()[cells] = timestamp_s
        grid.last_seen.ravel()[cells] = timestamp_s
        grid._recent_floor[grid._history_index] = 0
        grid._recent_obstacles[grid._history_index] = 0
        grid._recent_floor[grid._history_index].ravel()[cells[dense]] = 1
        grid._history_index = (grid._history_index + 1) % len(grid._recent_floor)
        grid.integrated_frames += 1
        return len(cells)

    def integrate(self, result, floor, pose, timestamp_s):
        phase_start = time.perf_counter()
        self.phase_ms = {}
        self._now_s = timestamp_s
        self.pose = pose
        self.tracking_ok = bool(pose is not None and pose.tracking_ok)
        self.last_candidates = 0
        self.last_rejection = None
        if not self.tracking_ok or not floor.usable:
            self.current_level_known = False
            update = None
            for level in self._levels.values():
                result_row = level.grid.integrate(result, floor, pose, timestamp_s,
                                                   clear_traversed=False)
                if level.id == self._active_id:
                    update = result_row
            return update if update is not None else self._base.integrate(
                result, floor, pose, timestamp_s)
        if not self._levels:
            base = self._new_level(0.0)
            self._base_id = base.id
            self._active_id = base.id
            self._initial_camera_offset_m = float(floor.height_m)
            self._epoch_offsets[self._epoch] = self._initial_camera_offset_m
        elif self._initial_camera_offset_m is None:
            # Only old maps from an earlier version can lack this datum.
            self._initial_camera_offset_m = float(floor.height_m)
            self._epoch_offsets[self._epoch] = self._initial_camera_offset_m
        if self._active_id is None:
            foot = pose.y_m - self._initial_camera_offset_m
            self._active_id = min(self._levels, key=lambda index: abs(
                self._levels[index].height_m - foot))
        sampled = sample_points(result, self.settings["sample_stride"],
                                self.settings["min_range_m"],
                                self.settings["far_hint_range_m"],
                                min_confidence=1)
        self.phase_ms["sample"] = (time.perf_counter() - phase_start) * 1000
        phase_start = time.perf_counter()
        p_cam, rows, cols = sampled
        if len(p_cam) and floor.fresh:
            p_grav = camera_to_gravity(p_cam, floor.rotation_grav_cam, pose.y_m)
            heights = p_grav[:, 1]
            x_world, z_world = gravity_to_world_xz(p_grav, pose)
            solid = (result.confidence[rows, cols] == 2) & (
                p_cam[:, 2] <= self.settings["max_range_m"])
            foot = pose.y_m - self._initial_camera_offset_m
            candidates = self._surface_candidates(
                heights, x_world, z_world, solid, foot)
            self.last_candidates = len(candidates)
            self.phase_ms["candidate"] = (time.perf_counter() - phase_start) * 1000
            phase_start = time.perf_counter()
            seen_ids = self._match_candidates(
                candidates, x_world, z_world, timestamp_s)
            self._refresh_adjacency()
        else:
            heights = np.empty(0, np.float32)
            x_world = z_world = heights
            solid = np.empty(0, bool)
            seen_ids = set()
        self.phase_ms["match"] = (time.perf_counter() - phase_start) * 1000
        phase_start = time.perf_counter()
        empty = (p_cam[:0], rows[:0], cols[:0])
        stats = None
        for level in self._levels.values():
            grid = level.grid
            if level.id not in seen_ids and level.id != self._active_id:
                grid._decay(timestamp_s)
                continue
            if level.id != self._active_id:
                self._integrate_support_only(
                    level, heights, x_world, z_world, solid, pose, timestamp_s)
                continue
            grid._recenter(pose)
            if len(p_cam) and floor.fresh:
                relative = heights - level.height_m
                near = np.abs(relative) <= self.settings["stair_surface_tolerance_m"]
                higher = ((relative >= self.settings["obstacle_min_height_m"]) &
                          (relative <= self.settings["obstacle_max_height_m"]))
                lower = (relative <= -self.settings["hole_min_below_m"])
                # Lower observed ground is a possible edge warning on the
                # wearer's level; it is also independent support on its own.
                selected = near | higher | (lower & (level.id == self._active_id))
                subset = (p_cam[selected], rows[selected], cols[selected])
            else:
                subset = empty
            local_floor = replace(floor, height_m=pose.y_m - level.height_m)
            update = grid.integrate(result, local_floor, pose, timestamp_s,
                                    sampled=subset,
                                    clear_traversed=level.id == self._active_id)
            if level.id == self._active_id:
                stats = update
        self.phase_ms["layers"] = (time.perf_counter() - phase_start) * 1000
        phase_start = time.perf_counter()
        if floor.fresh:
            self.integrated_frames += 1
            self._update_links(seen_ids, heights, x_world, z_world, solid)
            self.phase_ms["links"] = (time.perf_counter() - phase_start) * 1000
            phase_start = time.perf_counter()
            for level in self._levels.values():
                if level.id in seen_ids:
                    self._retain_support_memory(level)
            self._select_current_level()
        self._cap_obstacle_memory()
        self.phase_ms["memory"] = (time.perf_counter() - phase_start) * 1000
        self.last_confirmed_links = self.confirmed_step_count
        return stats if stats is not None else self._active.integrate(
            result, floor, pose, timestamp_s, sampled=empty)

    def transitions(self):
        """Observed step links; unobserved continuation remains unknown."""
        output = []
        for link in self._links.values():
            lower = self._levels[link.lower_id]
            upper = self._levels[link.upper_id]
            confirmed = self._link_confirmed(link)
            degrees = {level.id: sum(
                self._link_confirmed(other) and
                level.id in (other.lower_id, other.upper_id)
                for other in self._links.values()) for level in (lower, upper)}
            open_ids = [level.id for level in (lower, upper)
                        if confirmed and level.id != self._base_id and
                        degrees[level.id] == 1 and
                        len(level.support_memory) * self.cell_m ** 2 <
                        self.settings["stair_landing_area_m2"]]
            output.append({"lower_id": lower.id, "upper_id": upper.id,
                           "rise_m": upper.height_m - lower.height_m,
                           "confirmed": confirmed, "observations": link.observations,
                           "open_end": bool(open_ids),
                           "open_level_ids": open_ids})
        return output

    def state_grid(self):
        states = self._active.state_grid().copy()
        if not self.current_level_known:
            states[states == self.FREE] = self.UNKNOWN
            return states
        radius = max(1, int(math.ceil(
            self.settings["stair_max_tread_gap_m"] / self.cell_m)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (radius * 2 + 1, radius * 2 + 1))
        active_edge = cv2.dilate(self._support_mask(
            self._levels[self._active_id]).astype(np.uint8), kernel) > 0
        for link in self._links.values():
            if not self._link_confirmed(link):
                continue
            if self._active_id == link.lower_id:
                other, code = self._levels[link.upper_id], self.STEP_UP
            elif self._active_id == link.upper_id:
                other, code = self._levels[link.lower_id], self.STEP_DOWN
            else:
                continue
            mask = (self._support_mask(other, aligned=True) & active_edge &
                    (states != self.FREE))
            states[mask] = code
        return states

    def other_surface_mask(self):
        """Fresh support at other elevations, including disconnected patches."""
        mask = np.zeros((self.side, self.side), bool)
        for level in self._levels.values():
            if level.id != self._active_id:
                mask |= self._support_mask(level, aligned=True)
        return mask

    def surface_state(self, x_m, z_m, height_m):
        if not self._levels:
            return "unknown"
        level = min(self._levels.values(), key=lambda item: abs(
            item.height_m - height_m))
        if abs(level.height_m - height_m) > self.settings["stair_level_separation_m"]:
            return "unknown"
        return level.grid.cell_state(x_m, z_m)

    def cell_state(self, x_m, z_m):
        ix = int(math.floor((x_m - self.origin_x_m) / self.cell_m))
        iz = int(math.floor((z_m - self.origin_z_m) / self.cell_m))
        if not (0 <= ix < self.side and 0 <= iz < self.side):
            return "unknown"
        return ("unknown", "free", "obstacle", "unknown",
                "remembered_obstacle", "step_up", "step_down")[
                    self.state_grid()[iz, ix]]

    def walked_path_mask(self):
        return self._active.walked_path_mask()

    def _hazard_points(self):
        states = self.state_grid()
        return np.nonzero((states == self.OBSTACLE) |
                          (states == self.STEP_UP) | (states == self.STEP_DOWN))

    def nearest_obstacle(self, direction_deg=0.0, width_deg=20.0):
        if self.pose is None:
            return None
        iz, ix = self._hazard_points()
        if not len(ix):
            return None
        dx = self.origin_x_m + (ix + 0.5) * self.cell_m - self.pose.x_m
        dz = self.origin_z_m + (iz + 0.5) * self.cell_m - self.pose.z_m
        distance = np.hypot(dx, dz)
        bearing = np.rad2deg(np.arctan2(dx, dz) - self.pose.yaw_rad)
        delta = (bearing - direction_deg + 180) % 360 - 180
        selected = np.abs(delta) <= width_deg / 2
        return float(distance[selected].min()) if np.any(selected) else None

    def clearance_sectors(self):
        if self.pose is None:
            return {name: None for name in ("left", "front", "right")}
        iz, ix = self._hazard_points()
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

    def nearest_step(self, direction_deg=0.0, width_deg=60.0):
        if self.pose is None:
            return None
        states = self.state_grid()
        iz, ix = np.nonzero((states == self.STEP_UP) | (states == self.STEP_DOWN))
        if not len(ix):
            return None
        dx = self.origin_x_m + (ix + 0.5) * self.cell_m - self.pose.x_m
        dz = self.origin_z_m + (iz + 0.5) * self.cell_m - self.pose.z_m
        distance = np.hypot(dx, dz)
        bearing = np.rad2deg(np.arctan2(dx, dz) - self.pose.yaw_rad)
        delta = (bearing - direction_deg + 180) % 360 - 180
        selected = np.flatnonzero(np.abs(delta) <= width_deg / 2)
        if not len(selected):
            return None
        closest = selected[np.argmin(distance[selected])]
        direction = ("up" if states[iz[closest], ix[closest]] ==
                     self.STEP_UP else "down")
        destination_known = False
        for link in self._links.values():
            if not self._link_confirmed(link):
                continue
            target_id = (link.upper_id if direction == "up" and
                         link.lower_id == self._active_id else
                         link.lower_id if direction == "down" and
                         link.upper_id == self._active_id else None)
            if target_id is None:
                continue
            target = self._levels[target_id]
            if self._support_mask(target, aligned=True)[iz[closest], ix[closest]]:
                destination_known = (len(target.support_memory) * self.cell_m ** 2 >=
                                     self.settings["stair_landing_area_m2"])
                break
        return {"distance_m": float(distance[closest]),
                "direction": direction,
                "destination_known": destination_known}

    def blind_zone(self):
        return self._active.blind_zone()

    def forward_coverage(self, length_m, width_m):
        coverage = self._active.forward_coverage(length_m, width_m)
        if coverage is None:
            return None
        states = self.state_grid()
        iz, ix = np.indices(states.shape)
        dx = self.origin_x_m + (ix + 0.5) * self.cell_m - self.pose.x_m
        dz = self.origin_z_m + (iz + 0.5) * self.cell_m - self.pose.z_m
        c, s = math.cos(self.pose.yaw_rad), math.sin(self.pose.yaw_rad)
        forward = s * dx + c * dz
        lateral = c * dx - s * dz
        region = ((forward >= 0) & (forward < length_m) &
                  (np.abs(lateral) < width_m / 2))
        coverage["steps"] = int(np.count_nonzero(region &
                                  ((states == self.STEP_UP) |
                                   (states == self.STEP_DOWN))))
        return coverage
