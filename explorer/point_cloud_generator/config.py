"""Configuration loading and validation for synthetic room scans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.json")


@dataclass(slots=True)
class RoomConfig:
    width_m: float
    depth_m: float
    height_m: float
    sensor_height_m: float
    front_offset_m: float
    include_ceiling: bool


@dataclass(slots=True)
class ObjectConfig:
    count: int
    allowed: list[str]
    weights: dict[str, float]
    wall_margin_m: float
    object_margin_m: float
    front_clearance_m: float
    placement_attempts: int


@dataclass(slots=True)
class ScanConfig:
    omniscient_scan: bool
    sensor_position_m: tuple[float, float, float]
    sensor_yaw_deg: float
    single_view_horizontal_fov_deg: float
    single_view_vertical_fov_deg: float
    points_per_m2: int
    noise_sigma_m: float
    dropout: float
    max_range_m: float
    outlier_fraction: float
    voxel_m: float
    max_points: int
    visibility_epsilon_m: float
    flat_plane_degradation: bool
    flat_plane_min_area_m2: float
    flat_plane_dropout: float
    flat_plane_center_noise_m: float
    flat_plane_edge_noise_m: float
    flat_plane_edge_width_m: float
    flat_plane_object_radius_m: float
    flat_plane_clarity_power: float


@dataclass(slots=True)
class GeneratorConfig:
    seed: int
    room: RoomConfig
    objects: ObjectConfig
    scan: ScanConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("The configuration file must contain a JSON object.")
    return value


def resolve_config(
    user: GeneratorConfig | Mapping[str, Any] | None = None,
) -> GeneratorConfig:
    """Resolve the package's default.json and optional user overrides."""
    if isinstance(user, GeneratorConfig):
        validate_config(user)
        return user

    raw = load_json(DEFAULT_CONFIG_PATH)
    if user:
        raw = _merge(raw, user)

    known = {"seed", "room", "objects", "scan"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {', '.join(unknown)}")

    try:
        config = GeneratorConfig(
            seed=int(raw["seed"]),
            room=RoomConfig(**raw["room"]),
            objects=ObjectConfig(**raw["objects"]),
            scan=ScanConfig(**raw["scan"]),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: GeneratorConfig) -> None:
    room, objects, scan = config.room, config.objects, config.scan
    if min(room.width_m, room.depth_m, room.height_m) <= 0:
        raise ValueError("Room dimensions must be positive.")
    if not 0 < room.sensor_height_m < room.height_m:
        raise ValueError("sensor_height_m must be inside the room height.")
    if not 0 <= room.front_offset_m < room.depth_m:
        raise ValueError("front_offset_m must be inside the room depth.")
    if objects.count < 0:
        raise ValueError("Object count cannot be negative.")
    supported = {"box", "cabinet", "table", "chair", "cylinder"}
    unsupported = sorted(set(objects.allowed) - supported)
    if not objects.allowed or unsupported:
        raise ValueError(f"Unsupported or empty object list: {unsupported or objects.allowed}")
    if any(value < 0 for value in objects.weights.values()):
        raise ValueError("Object weights cannot be negative.")
    if objects.placement_attempts < 1:
        raise ValueError("placement_attempts must be at least one.")
    if scan.points_per_m2 < 1 or scan.max_points < 1:
        raise ValueError("Point density and max_points must be positive.")
    if scan.noise_sigma_m < 0 or scan.voxel_m < 0 or scan.max_range_m <= 0:
        raise ValueError("Scan distances must be non-negative and max_range_m positive.")
    if not 0 <= scan.dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    if not 0 <= scan.outlier_fraction < 1:
        raise ValueError("outlier_fraction must be in [0, 1).")
    if len(scan.sensor_position_m) != 3 or not all(
        isinstance(value, (int, float)) for value in scan.sensor_position_m
    ):
        raise ValueError("sensor_position_m must contain three numbers.")
    if not 0 < scan.single_view_horizontal_fov_deg <= 360:
        raise ValueError("single_view_horizontal_fov_deg must be in (0, 360].")
    if not 0 < scan.single_view_vertical_fov_deg <= 180:
        raise ValueError("single_view_vertical_fov_deg must be in (0, 180].")
    if not 0 <= scan.flat_plane_dropout < 1:
        raise ValueError("flat_plane_dropout must be in [0, 1).")
    if min(
        scan.flat_plane_center_noise_m,
        scan.flat_plane_min_area_m2,
        scan.flat_plane_edge_noise_m,
        scan.flat_plane_edge_width_m,
        scan.flat_plane_object_radius_m,
    ) < 0:
        raise ValueError("Flat-plane noise distances cannot be negative.")
    if scan.flat_plane_edge_width_m == 0 or scan.flat_plane_object_radius_m == 0:
        raise ValueError("Flat-plane edge width and object radius must be positive.")
    if scan.flat_plane_clarity_power <= 0:
        raise ValueError("flat_plane_clarity_power must be positive.")

