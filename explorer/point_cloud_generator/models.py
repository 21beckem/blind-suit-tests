"""Public result structures for generated scans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SceneObject:
    object_id: int
    kind: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]  # width, depth, height
    yaw_deg: float
    color: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SensorInfo:
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forward: tuple[float, float, float] = (0.0, 0.0, -1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    units: str = "metres"
    frame: str = "x-right_y-up_z-back"
    world_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_yaw_deg: float = 0.0
    omniscient_scan: bool = True


@dataclass(slots=True)
class SyntheticScene:
    """A persistent randomized room which can be scanned repeatedly."""

    parts: tuple[Any, ...]
    objects: tuple[SceneObject, ...]
    config: dict[str, Any]


@dataclass(slots=True)
class ScanResult:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    object_ids: np.ndarray
    object_classes: dict[int, str]
    objects: tuple[SceneObject, ...]
    sensor: SensorInfo
    config: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "point_count": int(len(self.points)),
            "objects_frame": "world",
            "object_classes": {str(k): v for k, v in self.object_classes.items()},
            "objects": [asdict(item) for item in self.objects],
            "sensor": asdict(self.sensor),
            "config": self.config,
        }

