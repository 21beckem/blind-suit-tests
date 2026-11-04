"""Procedural room and furniture mesh construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import GeneratorConfig
from .models import SceneObject


@dataclass(slots=True)
class GeometryPart:
    mesh: object
    object_id: int
    kind: str
    color: tuple[float, float, float]
    large_plane: bool = False


OBJECT_COLORS = {
    "box": (0.72, 0.45, 0.22),
    "cabinet": (0.40, 0.28, 0.18),
    "table": (0.58, 0.36, 0.20),
    "chair": (0.18, 0.42, 0.68),
    "cylinder": (0.70, 0.22, 0.18),
}


def _o3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - clearer than an import-time failure
        raise RuntimeError("Point-cloud generation requires Open3D.") from exc
    return o3d


def _quad(vertices: list[tuple[float, float, float]], color: tuple[float, float, float]):
    o3d = _o3d()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh.paint_uniform_color(color)
    mesh.compute_triangle_normals()
    return mesh


def _box(size: tuple[float, float, float], center: np.ndarray, color, yaw: float = 0.0):
    o3d = _o3d()
    width, depth, height = size
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    mesh.translate((-width / 2, -height / 2, -depth / 2))
    if yaw:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0.0, yaw, 0.0)), center=(0, 0, 0))
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    mesh.compute_triangle_normals()
    return mesh


def _transform_local(mesh, center: np.ndarray, yaw: float):
    o3d = _o3d()
    if yaw:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0.0, yaw, 0.0)), center=(0, 0, 0))
    mesh.translate(center)
    return mesh


def room_parts(config: GeneratorConfig) -> list[GeometryPart]:
    room = config.room
    x0, x1 = -room.width_m / 2, room.width_m / 2
    y0, y1 = -room.sensor_height_m, room.height_m - room.sensor_height_m
    z0, z1 = -room.depth_m, -room.front_offset_m
    floor = (0.56, 0.54, 0.50)
    wall = (0.76, 0.78, 0.82)
    result = [
        GeometryPart(_quad([(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)], floor), 0, "floor", floor, True),
        GeometryPart(_quad([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], wall), 0, "back_wall", wall, True),
        GeometryPart(_quad([(x1, y0, z1), (x0, y0, z1), (x0, y1, z1), (x1, y1, z1)], wall), 0, "front_wall", wall, True),
        GeometryPart(_quad([(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)], wall), 0, "left_wall", wall, True),
        GeometryPart(_quad([(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)], wall), 0, "right_wall", wall, True),
    ]
    if room.include_ceiling:
        result.append(
            GeometryPart(_quad([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], wall), 0, "ceiling", wall, True)
        )
    return result


def random_size(kind: str, rng: np.random.Generator) -> tuple[float, float, float]:
    ranges = {
        "box": ((0.30, 0.95), (0.30, 0.80), (0.20, 0.95)),
        "cabinet": ((0.60, 1.20), (0.35, 0.65), (1.00, 1.90)),
        "table": ((0.85, 1.55), (0.60, 1.00), (0.72, 0.86)),
        "chair": ((0.42, 0.62), (0.42, 0.62), (0.78, 1.05)),
        "cylinder": ((0.30, 0.70), (0.30, 0.70), (0.35, 1.00)),
    }
    return tuple(float(rng.uniform(*bounds)) for bounds in ranges[kind])


def footprint_extents(size: tuple[float, float, float], yaw: float) -> tuple[float, float]:
    width, depth, _ = size
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return 0.5 * (c * width + s * depth), 0.5 * (s * width + c * depth)


def footprints_overlap(
    left: SceneObject,
    right: SceneObject,
    margin_m: float = 0.0,
) -> bool:
    """Test two oriented rectangular footprints using the separating-axis theorem."""
    def description(item):
        yaw = math.radians(item.yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        axes = (np.array([c, -s]), np.array([s, c]))
        half = (item.size_m[0] / 2, item.size_m[1] / 2)
        center = np.array([item.center_m[0], item.center_m[2]])
        return center, axes, half

    lc, laxes, lhalf = description(left)
    rc, raxes, rhalf = description(right)
    delta = rc - lc
    for axis in (*laxes, *raxes):
        lr = sum(lhalf[i] * abs(float(np.dot(laxes[i], axis))) for i in range(2))
        rr = sum(rhalf[i] * abs(float(np.dot(raxes[i], axis))) for i in range(2))
        if abs(float(np.dot(delta, axis))) >= lr + rr + margin_m:
            return False
    return True


def place_objects(config: GeneratorConfig, rng: np.random.Generator) -> list[SceneObject]:
    room, spec = config.room, config.objects
    weights = np.asarray([spec.weights.get(name, 1.0) for name in spec.allowed], dtype=np.float64)
    if not np.any(weights > 0):
        raise ValueError("At least one allowed object must have positive weight.")
    weights /= weights.sum()
    kinds = [str(rng.choice(spec.allowed, p=weights)) for _ in range(spec.count)]
    # A chair without a table cannot satisfy the requested relationship. Promote one
    # chair to a table while preserving the configured object count.
    if "chair" in kinds and "table" not in kinds:
        kinds[kinds.index("chair")] = "table"
    while kinds.count("chair") > 4 * kinds.count("table"):
        kinds[kinds.index("chair")] = "table"
    sizes = [random_size(kind, rng) for kind in kinds]
    table_indices = [i for i, kind in enumerate(kinds) if kind == "table"]
    chair_count = kinds.count("chair")
    seating_table_count = min(len(table_indices), math.ceil(chair_count / 4))
    seating_table_ids = {index + 1 for index in table_indices[:seating_table_count]}
    # Tables and their seating are reserved before unrelated floor clutter. Cabinets
    # follow because their wall-only candidates rarely compete with this central zone.
    order = sorted(
        range(spec.count),
        key=lambda i: (
            0 if i + 1 in seating_table_ids
            else 1 if kinds[i] == "chair"
            else 2 if kinds[i] == "cabinet"
            else 3
        ),
    )
    placed: list[tuple[SceneObject, float, float]] = []

    def collides(candidate):
        return any(
            footprints_overlap(candidate, item, spec.object_margin_m)
            for item, _, _ in placed
        )

    for index in order:
        object_id, kind, size = index + 1, kinds[index], sizes[index]
        found = None
        for _ in range(spec.placement_attempts):
            if kind == "cabinet":
                wall = int(rng.integers(0, 3))
                if wall == 0:  # back wall; cabinet front faces the scanner
                    yaw = 0.0
                    ex, ez = footprint_extents(size, yaw)
                    x_limit = room.width_m / 2 - spec.wall_margin_m - ex
                    if x_limit <= 0:
                        break
                    x = float(rng.uniform(-x_limit, x_limit))
                    z = -room.depth_m + spec.wall_margin_m + ez
                else:  # side walls; shallow cabinet depth is normal to the wall
                    yaw = math.pi / 2
                    ex, ez = footprint_extents(size, yaw)
                    x = (-1 if wall == 1 else 1) * (room.width_m / 2 - spec.wall_margin_m - ex)
                    z_near = -(room.front_offset_m + spec.front_clearance_m + ez)
                    z_far = -(room.depth_m - spec.wall_margin_m - ez)
                    if z_far >= z_near:
                        break
                    z = float(rng.uniform(z_far, z_near))
            elif kind == "chair":
                tables = [entry[0] for entry in placed if entry[0].object_id in seating_table_ids]
                if not tables:
                    break
                table = tables[int(rng.integers(0, len(tables)))]
                side = int(rng.integers(0, 4))
                table_yaw = math.radians(table.yaw_deg)
                tw, td, _ = table.size_m
                gap = float(rng.uniform(0.10, 0.28))
                if side < 2:
                    local = np.array([(-1 if side == 0 else 1) * (tw / 2 + size[1] / 2 + gap), 0.0])
                else:
                    local = np.array([0.0, (-1 if side == 2 else 1) * (td / 2 + size[1] / 2 + gap)])
                c, s = math.cos(table_yaw), math.sin(table_yaw)
                x = float(table.center_m[0] + c * local[0] + s * local[1])
                z = float(table.center_m[2] - s * local[0] + c * local[1])
                direction_x = table.center_m[0] - x
                direction_z = table.center_m[2] - z
                yaw = math.atan2(-direction_x, -direction_z)
            else:
                yaw = float(rng.uniform(-math.pi, math.pi))
                ex, ez = footprint_extents(size, yaw)
                seating_clearance = 0.75 if object_id in seating_table_ids else 0.0
                x_limit = room.width_m / 2 - spec.wall_margin_m - ex - seating_clearance
                z_near = -(room.front_offset_m + spec.front_clearance_m + ez + seating_clearance)
                z_far = -(room.depth_m - spec.wall_margin_m - ez - seating_clearance)
                if x_limit <= 0 or z_far >= z_near:
                    break
                x = float(rng.uniform(-x_limit, x_limit))
                z = float(rng.uniform(z_far, z_near))

            ex, ez = footprint_extents(size, yaw)
            inside = (
                abs(x) + ex <= room.width_m / 2 - spec.wall_margin_m + 1e-9
                and -room.depth_m + spec.wall_margin_m - 1e-9 <= z - ez
                and z + ez <= -room.front_offset_m - spec.front_clearance_m + 1e-9
            )
            y = -room.sensor_height_m + size[2] / 2
            color = OBJECT_COLORS[kind]
            candidate = SceneObject(
                object_id=object_id,
                kind=kind,
                center_m=(x, y, z),
                size_m=size,
                yaw_deg=math.degrees(yaw),
                color=color,
            )
            if not inside or collides(candidate):
                continue
            found = candidate
            placed.append((found, ex, ez))
            break
        if found is None:
            raise ValueError(
                f"Could only place {len(placed)} of {spec.count} objects with the requested relationships; "
                "reduce the object count/margins or enlarge the room."
            )
    return [item[0] for item in placed]


def object_mesh(item: SceneObject):
    o3d = _o3d()
    width, depth, height = item.size_m
    center = np.asarray(item.center_m, dtype=np.float64)
    yaw = math.radians(item.yaw_deg)
    color = item.color
    floor_y = center[1] - height / 2

    if item.kind in {"box", "cabinet"}:
        return _box(item.size_m, center, color, yaw)

    if item.kind == "cylinder":
        radius = width / 2
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=24, split=4)
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((math.pi / 2, 0.0, 0.0)), center=(0, 0, 0))
        mesh.translate(center)
        mesh.paint_uniform_color(color)
        mesh.compute_triangle_normals()
        return mesh

    combined = o3d.geometry.TriangleMesh()
    if item.kind == "table":
        top_t = 0.07
        combined += _box((width, depth, top_t), np.array([0.0, height / 2 - top_t / 2, 0.0]), color)
        leg = min(0.075, width * 0.08, depth * 0.08)
        for sx in (-1, 1):
            for sz in (-1, 1):
                combined += _box(
                    (leg, leg, height - top_t),
                    np.array([sx * (width / 2 - leg), -top_t / 2, sz * (depth / 2 - leg)]),
                    color,
                )
    elif item.kind == "chair":
        seat_h = min(0.48, height * 0.52)
        seat_t, leg = 0.055, min(0.055, width * 0.10)
        combined += _box((width, depth, seat_t), np.array([0.0, seat_h - height / 2, 0.0]), color)
        for sx in (-1, 1):
            for sz in (-1, 1):
                combined += _box(
                    (leg, leg, seat_h),
                    np.array([sx * (width / 2 - leg), (seat_h - height) / 2, sz * (depth / 2 - leg)]),
                    color,
                )
        back_t = 0.055
        back_h = height - seat_h
        combined += _box(
            (width, back_t, back_h),
            np.array([0.0, seat_h + back_h / 2 - height / 2, depth / 2 - back_t / 2]),
            color,
        )
    else:  # guarded by configuration validation
        raise ValueError(f"Unsupported object kind: {item.kind}")

    # Composite objects were built around their own centre; put their bottom on the floor.
    _transform_local(combined, np.array([center[0], floor_y + height / 2, center[2]]), yaw)
    combined.compute_triangle_normals()
    return combined


def build_geometry(config: GeneratorConfig, rng: np.random.Generator):
    objects = place_objects(config, rng)
    parts = room_parts(config)
    for item in objects:
        parts.append(GeometryPart(object_mesh(item), item.object_id, item.kind, item.color))
    return parts, objects

