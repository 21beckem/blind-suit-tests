"""Command-line entry point for synthetic room scans."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np

from .config import load_json
from .generator import generate_scan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and display a procedural room point cloud.")
    parser.add_argument("--config", metavar="JSON", help="base JSON configuration")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--objects", type=int, help="number of room objects")
    parser.add_argument("--room-width", type=float)
    parser.add_argument("--room-depth", type=float)
    parser.add_argument("--room-height", type=float)
    parser.add_argument("--sensor-height", type=float)
    parser.add_argument("--points-per-m2", type=int)
    parser.add_argument("--noise-sigma", type=float, metavar="METRES")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--max-range", type=float, metavar="METRES")
    scan_mode = parser.add_mutually_exclusive_group()
    scan_mode.add_argument(
        "--omniscient-scan", dest="omniscient_scan", action="store_true", default=None,
        help="override configured mode and include every room surface",
    )
    scan_mode.add_argument(
        "--single-view-scan", dest="omniscient_scan", action="store_false",
        help="override configured mode with one forward scanner view",
    )
    parser.add_argument("--sensor-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--sensor-yaw", type=float, metavar="DEGREES")
    parser.add_argument("--single-view-horizontal-fov", type=float, metavar="DEGREES")
    parser.add_argument("--single-view-vertical-fov", type=float, metavar="DEGREES")
    parser.add_argument("--flat-plane-dropout", type=float)
    parser.add_argument("--flat-plane-center-noise", type=float, metavar="METRES")
    parser.add_argument("--flat-plane-edge-noise", type=float, metavar="METRES")
    parser.add_argument("--flat-plane-edge-width", type=float, metavar="METRES")
    parser.add_argument("--flat-plane-object-radius", type=float, metavar="METRES")
    parser.add_argument("--no-flat-plane-degradation", action="store_true")
    parser.add_argument("--save-ply", metavar="FILE")
    parser.add_argument("--save-npz", metavar="FILE")
    parser.add_argument("--save-metadata", metavar="JSON")
    parser.add_argument("--no-view", action="store_true")
    return parser


def _overrides(args) -> dict:
    data: dict = {}
    if args.seed is not None:
        data["seed"] = args.seed
    room_values = {
        "width_m": args.room_width,
        "depth_m": args.room_depth,
        "height_m": args.room_height,
        "sensor_height_m": args.sensor_height,
    }
    scan_values = {
        "points_per_m2": args.points_per_m2,
        "noise_sigma_m": args.noise_sigma,
        "dropout": args.dropout,
        "max_range_m": args.max_range,
        "omniscient_scan": args.omniscient_scan,
        "sensor_position_m": args.sensor_position,
        "sensor_yaw_deg": args.sensor_yaw,
        "single_view_horizontal_fov_deg": args.single_view_horizontal_fov,
        "single_view_vertical_fov_deg": args.single_view_vertical_fov,
        "flat_plane_dropout": args.flat_plane_dropout,
        "flat_plane_center_noise_m": args.flat_plane_center_noise,
        "flat_plane_edge_noise_m": args.flat_plane_edge_noise,
        "flat_plane_edge_width_m": args.flat_plane_edge_width,
        "flat_plane_object_radius_m": args.flat_plane_object_radius,
    }
    if any(value is not None for value in room_values.values()):
        data["room"] = {key: value for key, value in room_values.items() if value is not None}
    if any(value is not None for value in scan_values.values()):
        data["scan"] = {key: value for key, value in scan_values.items() if value is not None}
    if args.objects is not None:
        data["objects"] = {"count": args.objects}
    if args.no_flat_plane_degradation:
        data.setdefault("scan", {})["flat_plane_degradation"] = False
    return data


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    user = load_json(args.config) if args.config else {}
    user = _merge(user, _overrides(args))
    try:
        scan = generate_scan(user)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Generated {len(scan.points):,} visible points with seed {scan.config['seed']}.")
    room = scan.config["room"]
    print(f"Room: {room['width_m']:.1f} x {room['depth_m']:.1f} x {room['height_m']:.1f} m")
    print("Objects: " + (", ".join(f"#{item.object_id} {item.kind}" for item in scan.objects) or "none"))
    mode = "omniscient" if scan.sensor.omniscient_scan else "single-view"
    print(f"Scan: {mode}; pose={scan.sensor.world_position_m}, yaw={scan.sensor.world_yaw_deg:.1f} deg")

    if args.save_ply:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(scan.points)
        cloud.colors = o3d.utility.Vector3dVector(scan.colors)
        cloud.normals = o3d.utility.Vector3dVector(scan.normals)
        if not o3d.io.write_point_cloud(str(args.save_ply), cloud):
            print(f"error: failed to save {args.save_ply}", file=sys.stderr)
            return 2
        print(f"Saved {args.save_ply}")
    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            points=scan.points,
            colors=scan.colors,
            normals=scan.normals,
            object_ids=scan.object_ids,
            metadata_json=json.dumps(scan.metadata()),
        )
        print(f"Saved {args.save_npz}")
    if args.save_metadata:
        Path(args.save_metadata).write_text(json.dumps(scan.metadata(), indent=2), encoding="utf-8")
        print(f"Saved {args.save_metadata}")
    if not args.no_view:
        from .viewer import show_scan

        show_scan(scan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

