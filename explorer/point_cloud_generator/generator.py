"""Synthetic high-quality room scan generator."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config import GeneratorConfig, resolve_config
from .geometry import GeometryPart, build_geometry
from .models import ScanResult, SensorInfo, SyntheticScene


def _mesh_arrays(part: GeometryPart):
    vertices = np.asarray(part.mesh.vertices, dtype=np.float64)
    triangles = np.asarray(part.mesh.triangles, dtype=np.int32)
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    good = double_area > 1e-12
    normals = np.zeros_like(cross)
    normals[good] = cross[good] / double_area[good, None]
    return tri[good], double_area[good] * 0.5, normals[good]


def _sample_part(part: GeometryPart, count: int, rng: np.random.Generator, plane_area_m2: float):
    tri, areas, tri_normals = _mesh_arrays(part)
    chosen = rng.choice(len(tri), size=count, p=areas / areas.sum())
    selected = tri[chosen]
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    points = selected[:, 0] + uv[:, :1] * (selected[:, 1] - selected[:, 0]) + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    normals = tri_normals[chosen]
    colors = np.repeat(np.asarray(part.color, dtype=np.float64)[None, :], count, axis=0)
    ids = np.full(count, part.object_id, dtype=np.int32)
    plane = np.full(count, part.large_plane, dtype=bool)
    if part.object_id > 0:
        plane |= areas[chosen] >= plane_area_m2
    if part.large_plane:
        vertices = np.asarray(part.mesh.vertices, dtype=np.float64)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        varying = np.flatnonzero((high - low) > 1e-8)
        edge_distance = np.min(
            np.column_stack([points[:, varying] - low[varying], high[varying] - points[:, varying]]),
            axis=1,
        )
    else:
        edge_distance = np.full(count, np.inf, dtype=np.float64)
        object_plane = plane & (part.object_id > 0)
        if np.any(object_plane):
            # Object meshes contain individual planar faces. Measuring distance to
            # the selected triangle's edges gives face boundaries the same clean
            # transition as room-plane boundaries.
            selected_plane = selected[object_plane]
            p = points[object_plane]
            distances = []
            for edge_start, edge_end in ((0, 1), (1, 2), (2, 0)):
                start = selected_plane[:, edge_start]
                edge = selected_plane[:, edge_end] - start
                t = np.sum((p - start) * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1e-12)
                t = np.clip(t, 0.0, 1.0)
                distances.append(np.linalg.norm(p - (start + t[:, None] * edge), axis=1))
            edge_distance[object_plane] = np.min(distances, axis=0)
    return points, colors, normals, ids, plane, edge_distance


def _combined_mesh(parts: list[GeometryPart]):
    import open3d as o3d

    combined = o3d.geometry.TriangleMesh()
    for part in parts:
        combined += part.mesh
    return combined


def _visible_mask(points: np.ndarray, mesh, origin: np.ndarray, epsilon: float) -> np.ndarray:
    """Retain candidates that are the first surface reached from the scanner."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    offsets = points - origin
    distance = np.linalg.norm(offsets, axis=1)
    directions = offsets / np.maximum(distance[:, None], 1e-12)
    keep = np.zeros(len(points), dtype=bool)
    chunk = 250_000
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        origins = np.repeat(origin[None, :], stop - start, axis=0).astype(np.float32)
        rays = np.concatenate([origins, directions[start:stop].astype(np.float32)], axis=1)
        hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        keep[start:stop] = np.isfinite(hits) & (hits >= distance[start:stop] - epsilon)
    return keep


def _voxel_first(points, colors, normals, ids, voxel: float):
    if voxel <= 0 or len(points) == 0:
        return points, colors, normals, ids
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return points[first], colors[first], normals[first], ids[first]


def _distance_to_other_surfaces(
    points: np.ndarray,
    plane_ids: np.ndarray,
    surface_points: np.ndarray,
    surface_ids: np.ndarray,
) -> np.ndarray:
    """Distance to actual nearby object surfaces, excluding the same object.

    Using sampled surfaces instead of an object's enclosing box is important for
    furniture: the empty floor under the centre of a table is not close to the
    tabletop merely because it lies inside the table's bounding volume.
    """
    from scipy.spatial import cKDTree

    closest = np.full(len(points), np.inf, dtype=np.float64)
    if len(surface_points) == 0:
        return closest
    for object_id in np.unique(plane_ids):
        target = np.flatnonzero(plane_ids == object_id)
        if object_id == 0:
            candidates = surface_ids > 0
        else:
            candidates = (surface_ids > 0) & (surface_ids != object_id)
        if np.any(candidates):
            closest[target] = cKDTree(surface_points[candidates]).query(points[target])[0]
    return closest


def _degrade_large_planes(
    points,
    normals,
    plane_mask,
    edge_distance,
    plane_ids,
    proximity_points,
    proximity_ids,
    scan_cfg,
    rng,
):
    """Drop and warp plane centres while preserving edges and object-adjacent regions."""
    if not scan_cfg.flat_plane_degradation or not np.any(plane_mask):
        return np.ones(len(points), dtype=bool)
    indices = np.flatnonzero(plane_mask)
    edge_clarity = np.clip(
        1.0 - edge_distance[indices] / scan_cfg.flat_plane_edge_width_m, 0.0, 1.0
    )
    if len(proximity_points):
        object_distance = _distance_to_other_surfaces(
            points[indices], plane_ids[indices], proximity_points, proximity_ids
        )
        object_clarity = np.clip(
            1.0 - object_distance / scan_cfg.flat_plane_object_radius_m, 0.0, 1.0
        )
        clarity = np.maximum(edge_clarity, object_clarity)
    else:
        clarity = edge_clarity
    severity = np.power(1.0 - clarity, scan_cfg.flat_plane_clarity_power)
    keep = np.ones(len(points), dtype=bool)
    keep[indices] = rng.random(len(indices)) >= scan_cfg.flat_plane_dropout * severity
    sigma = scan_cfg.flat_plane_edge_noise_m + (
        scan_cfg.flat_plane_center_noise_m - scan_cfg.flat_plane_edge_noise_m
    ) * severity
    points[indices] += normals[indices] * rng.normal(0.0, sigma)[:, None]
    return keep


def _rotation_y(yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _single_view_mask(points: np.ndarray, origin: np.ndarray, scan_cfg) -> np.ndarray:
    """Keep points inside the scanner's forward pinhole-camera frustum."""
    local = (points - origin) @ _rotation_y(scan_cfg.sensor_yaw_deg)
    horizontal = np.abs(np.degrees(np.arctan2(local[:, 0], -local[:, 2])))
    planar_range = np.hypot(local[:, 0], local[:, 2])
    vertical = np.abs(np.degrees(np.arctan2(local[:, 1], planar_range)))
    return (
        (local[:, 2] < 0)
        & (horizontal <= scan_cfg.single_view_horizontal_fov_deg / 2)
        & (vertical <= scan_cfg.single_view_vertical_fov_deg / 2)
    )


def generate_scene(
    config: GeneratorConfig | Mapping[str, Any] | None = None,
) -> SyntheticScene:
    """Create a persistent randomized room which can be scanned more than once."""
    cfg = resolve_config(config)
    rng = np.random.default_rng(cfg.seed)
    parts, objects = build_geometry(cfg, rng)
    return SyntheticScene(tuple(parts), tuple(objects), cfg.to_dict())


def scan_scene(
    scene: SyntheticScene,
    *,
    scan_overrides: Mapping[str, Any] | None = None,
    sensor_position_m: tuple[float, float, float] | None = None,
    sensor_yaw_deg: float | None = None,
    omniscient_scan: bool | None = None,
) -> ScanResult:
    """Scan an existing scene, optionally from a new world-space pose.

    The returned points are always transformed into the detector's scanner-local
    coordinate frame. Reusing ``scene`` preserves the room and every object.
    """
    raw = dict(scene.config)
    raw["room"] = dict(raw["room"])
    raw["objects"] = dict(raw["objects"])
    raw["scan"] = dict(raw["scan"])
    if scan_overrides:
        raw["scan"].update(scan_overrides)
    if sensor_position_m is not None:
        raw["scan"]["sensor_position_m"] = sensor_position_m
    if sensor_yaw_deg is not None:
        raw["scan"]["sensor_yaw_deg"] = sensor_yaw_deg
    if omniscient_scan is not None:
        raw["scan"]["omniscient_scan"] = omniscient_scan
    cfg = resolve_config(raw)
    rng = np.random.default_rng(cfg.seed)
    parts = list(scene.parts)
    objects = scene.objects


    areas = np.asarray([_mesh_arrays(part)[1].sum() for part in parts], dtype=np.float64)
    desired = np.maximum(1, np.rint(areas * cfg.scan.points_per_m2).astype(np.int64))
    total = int(desired.sum())
    if total > cfg.scan.max_points:
        desired = np.maximum(1, np.floor(desired * (cfg.scan.max_points / total)).astype(np.int64))

    sampled = [
        _sample_part(part, int(count), rng, cfg.scan.flat_plane_min_area_m2)
        for part, count in zip(parts, desired)
    ]
    points = np.concatenate([item[0] for item in sampled])
    colors = np.concatenate([item[1] for item in sampled])
    normals = np.concatenate([item[2] for item in sampled])
    object_ids = np.concatenate([item[3] for item in sampled])
    plane_mask = np.concatenate([item[4] for item in sampled])
    edge_distance = np.concatenate([item[5] for item in sampled])

    sensor_position = np.asarray(cfg.scan.sensor_position_m, dtype=np.float64)
    if not cfg.scan.omniscient_scan:
        in_view = _single_view_mask(points, sensor_position, cfg.scan)
        visible = _visible_mask(
            points, _combined_mesh(parts), sensor_position, cfg.scan.visibility_epsilon_m
        )
        in_range = np.linalg.norm(points - sensor_position, axis=1) <= cfg.scan.max_range_m
        keep = in_view & visible & in_range
        points, colors, normals, object_ids, plane_mask, edge_distance = (
            points[keep], colors[keep], normals[keep], object_ids[keep], plane_mask[keep], edge_distance[keep]
        )

    object_surface_mask = object_ids > 0
    plane_keep = _degrade_large_planes(
        points,
        normals,
        plane_mask,
        edge_distance,
        object_ids,
        points[object_surface_mask],
        object_ids[object_surface_mask],
        cfg.scan,
        rng,
    )
    points, colors, normals, object_ids = (
        points[plane_keep], colors[plane_keep], normals[plane_keep], object_ids[plane_keep]
    )

    if cfg.scan.dropout:
        keep = rng.random(len(points)) >= cfg.scan.dropout
        points, colors, normals, object_ids = points[keep], colors[keep], normals[keep], object_ids[keep]
    if cfg.scan.noise_sigma_m:
        points = points + rng.normal(0.0, cfg.scan.noise_sigma_m, points.shape)

    if cfg.scan.outlier_fraction and len(points):
        count = max(1, int(len(points) * cfg.scan.outlier_fraction))
        room = cfg.room
        outliers = np.column_stack(
            [
                rng.uniform(-room.width_m / 2, room.width_m / 2, count),
                rng.uniform(-room.sensor_height_m, room.height_m - room.sensor_height_m, count),
                rng.uniform(-room.depth_m, -room.front_offset_m, count),
            ]
        )
        points = np.vstack([points, outliers])
        colors = np.vstack([colors, np.repeat([[1.0, 0.0, 1.0]], count, axis=0)])
        normals = np.vstack([normals, np.zeros((count, 3))])
        object_ids = np.concatenate([object_ids, np.full(count, -1, dtype=np.int32)])

    # Transform the stable world into the requested scanner-local detector frame.
    rotation = _rotation_y(cfg.scan.sensor_yaw_deg)
    points = (points - sensor_position) @ rotation
    normals = normals @ rotation
    # Present normals consistently toward the scanner.
    flip = np.einsum("ij,ij->i", normals, -points) < 0
    normals[flip] *= -1
    points, colors, normals, object_ids = _voxel_first(
        points, colors, normals, object_ids, cfg.scan.voxel_m
    )

    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(np.clip(colors, 0.0, 1.0), dtype=np.float64)
    normals = np.ascontiguousarray(normals, dtype=np.float64)
    object_ids = np.ascontiguousarray(object_ids, dtype=np.int32)
    classes = {0: "room", -1: "outlier", **{item.object_id: item.kind for item in objects}}
    return ScanResult(
        points=points,
        colors=colors,
        normals=normals,
        object_ids=object_ids,
        object_classes=classes,
        objects=tuple(objects),
        sensor=SensorInfo(
            world_position_m=tuple(float(value) for value in sensor_position),
            world_yaw_deg=float(cfg.scan.sensor_yaw_deg),
            omniscient_scan=bool(cfg.scan.omniscient_scan),
        ),
        config=cfg.to_dict(),
    )


def generate_scan(
    config: GeneratorConfig | Mapping[str, Any] | None = None,
) -> ScanResult:
    """Generate a new scene and scan it using its configured scanner pose."""
    scene = generate_scene(config)
    return scan_scene(scene)


def generate_point_cloud(
    config: GeneratorConfig | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return the detector-compatible ``(N, 3)`` point array in metres."""
    return generate_scan(config).points

