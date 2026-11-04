"""Procedural high-quality room point-cloud generation.

``generate_point_cloud`` is the small detector-facing API. ``generate_scan``
returns the same points with colors, normals, labels, and ground truth.
"""

from .config import GeneratorConfig, ObjectConfig, RoomConfig, ScanConfig, load_json, resolve_config
from .generator import generate_point_cloud, generate_scan, generate_scene, scan_scene
from .models import ScanResult, SceneObject, SensorInfo, SyntheticScene

__all__ = [
    "GeneratorConfig",
    "ObjectConfig",
    "RoomConfig",
    "ScanConfig",
    "ScanResult",
    "SceneObject",
    "SensorInfo",
    "SyntheticScene",
    "generate_point_cloud",
    "generate_scan",
    "generate_scene",
    "scan_scene",
    "load_json",
    "resolve_config",
]

