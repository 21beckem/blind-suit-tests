#!/usr/bin/env python3
"""Run every discovered stereo recording deterministically and save diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from module.calibration import load_calibration
from module.config import load_config
from module.depth_mapper import DepthMapper
from module.ground_plane import GroundPlaneEstimator
from module.image_tools import ImageTools
from module.map_view import render_map
from module.obstacle_map import ObstacleMap
from module.pose_tracker import PoseTracker
from module.recorded_camera_feed_sequential import CameraFeed


ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def discover_recordings(root=ROOT):
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and
                  "stereo_out" not in path.relative_to(root).parts)


def _metadata(path, config):
    calibration = load_calibration(config["calibration"])
    expected_eye = tuple(int(x) for x in calibration["image_size"].ravel())
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None, "cannot open video"
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec_number = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_number >> (8 * shift)) & 255) for shift in range(4))
        if not 1 <= fps <= 240:
            return None, f"invalid source frame rate: {fps}"
        if (width, height) != (2 * expected_eye[0], expected_eye[1]):
            return None, (f"frame is {width}x{height}; calibration requires "
                          f"{2 * expected_eye[0]}x{expected_eye[1]} side-by-side")
        ok, frame = capture.read()
        if not ok or frame is None or frame.shape[:2] != (height, width):
            return None, "first frame cannot be decoded at the advertised size"
        if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            return None, "decoded frame is not uint8 grayscale or BGR"
        q = calibration.get("Q")
        focal_times_baseline = (float(q[2, 3] / q[3, 2]) * config["depth"]["scale"]
                                if q is not None and q[3, 2] != 0 else None)
        return {"fps": fps, "advertised_frames": count, "width": width,
                "height": height, "eye_width": expected_eye[0],
                "eye_height": expected_eye[1], "codec": codec,
                "working_focal_times_baseline_px_m": focal_times_baseline,
                "decoded_channels": 1 if frame.ndim == 2 else frame.shape[2]}, None
    finally:
        capture.release()


def _summary(values):
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")


def _fixture_checkpoint(path, config, metadata, frames):
    """Guard the annotated apartment moment without constraining new clips."""
    fixture = config["map"].get("acceptance_fixture")
    if not fixture or path.name != fixture["recording_name"]:
        return {"matched": False}
    digest = hashlib.sha256()
    with path.open("rb") as recording:
        for chunk in iter(lambda: recording.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != fixture["sha256"]:
        return {"matched": False, "reason": "recording hash differs from fixture"}
    checkpoint = (min(frames, key=lambda row: abs(
        row["timestamp_s"] - fixture["checkpoint_s"])) if frames else None)
    timed = (checkpoint is not None and
             abs(checkpoint["timestamp_s"] - fixture["checkpoint_s"]) <=
             0.5 / metadata["fps"] + 1e-6)
    checks = {"fixture_checkpoint_present": bool(timed)}
    if timed:
        checks.update({
            "fixture_free_floor_coverage": checkpoint["map_free"] >=
                                           fixture["min_free_cells"],
            "fixture_obstacle_paint": checkpoint["map_obstacles"] <=
                                      fixture["max_obstacle_cells"],
            "fixture_dropoff_paint": checkpoint["map_danger_unknown"] <=
                                     fixture["max_danger_unknown_cells"],
        })
    early = (min(frames, key=lambda row: abs(
        row["timestamp_s"] - fixture["early_checkpoint_s"])) if frames else None)
    early_timed = (early is not None and
                   abs(early["timestamp_s"] - fixture["early_checkpoint_s"]) <=
                   0.5 / metadata["fps"] + 1e-6)
    checks["fixture_early_checkpoint_present"] = bool(early_timed)
    if early_timed:
        checks.update({
            "fixture_early_obstacle_paint": early["map_obstacles"] <=
                                            fixture["early_max_obstacle_cells"],
            "fixture_early_dropoff_paint": early["map_danger_unknown"] <=
                                           fixture["early_max_danger_unknown_cells"],
        })
    first_tracked = next((i for i, row in enumerate(frames)
                          if row["tracking_ok"]), None)
    checks["fixture_tracking_continuous_after_start"] = (
        first_tracked is not None and
        all(row["tracking_ok"] for row in frames[first_tracked:]))
    checks["fixture_no_map_reset"] = not any(row["reset_map"] for row in frames)
    return {"matched": True, "timestamp_s": fixture["checkpoint_s"],
            "frame": checkpoint["frame"] if timed else None,
            "counts": ({key: checkpoint[key] for key in
                        ("map_free", "map_obstacles", "map_danger_unknown")}
                       if timed else None),
            "early_timestamp_s": fixture["early_checkpoint_s"],
            "early_frame": early["frame"] if early_timed else None,
            "early_counts": ({key: early[key] for key in
                             ("map_free", "map_obstacles", "map_danger_unknown")}
                            if early_timed else None), "checks": checks}


def run_recording(path, config, output_root=None, debug_interval_s=None):
    path = Path(path).resolve()
    output_root = Path(output_root or config["map"]["output_dir"])
    relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path(path.name)
    label = "__".join(relative.with_suffix("").parts)
    output = output_root / label
    if debug_interval_s is None:
        debug_interval_s = config["map"]["debug_interval_s"]
    metrics = {"recording": str(path), "status": "unsuitable", "reason": None,
               "config": {"depth": config["depth"], "map": config["map"]}}
    try:
        metadata, problem = _metadata(path, config)
    except Exception as error:
        metadata, problem = None, f"calibration or metadata error: {type(error).__name__}: {error}"
    metrics["metadata"] = metadata
    if problem:
        metrics["reason"] = problem
        _save_json(output / "metrics.json", metrics)
        return metrics

    depth = DepthMapper(config)
    ground = GroundPlaneEstimator(config)
    tracker = PoseTracker(config)
    grid = ObstacleMap(config)
    frames = []
    depth_times = []
    mapping_times = []
    floor_heights = []
    source_fallbacks = 0
    prior_source_ms = None
    first_source_ms = None
    next_debug_s = 0.0
    last_result = None
    last_t = 0.0
    safety_ok = True
    try:
        with CameraFeed({**config["camera"], "recording": str(path)}) as feed:
            while True:
                try:
                    left, right = feed.read()
                except RuntimeError as error:
                    if str(error) == "End of stereo recording":
                        break
                    raise
                index = len(frames)
                source_ms = feed._last_timestamp_ms
                expected_ms = index * 1000 / metadata["fps"]
                if (source_ms is None or not np.isfinite(source_ms) or
                        (prior_source_ms is not None and source_ms <= prior_source_ms)):
                    source_ms = (expected_ms if prior_source_ms is None else
                                 max(expected_ms, prior_source_ms + 1000 / metadata["fps"]))
                    source_fallbacks += 1
                prior_source_ms = source_ms
                if first_source_ms is None:
                    first_source_ms = source_ms
                timestamp_s = max(0.0, (source_ms - first_source_ms) / 1000)
                last_t = timestamp_s

                start = time.perf_counter()
                result = depth.compute(left, right)
                depth_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                floor = ground.fit(result, timestamp_s)
                pose = tracker.update(result, floor, timestamp_s)
                if tracker.reset_map:
                    grid = ObstacleMap(config)
                integration = grid.integrate(result, floor, pose, timestamp_s)
                mapping_ms = (time.perf_counter() - start) * 1000
                if integration["integrated"] and not pose.tracking_ok:
                    safety_ok = False
                depth_times.append(depth_ms)
                mapping_times.append(mapping_ms)
                if floor.fresh:
                    floor_heights.append(floor.height_m)
                states = grid.state_grid()
                row = {"frame": index, "timestamp_s": timestamp_s,
                       "solid_fraction": float(np.mean(result.confidence == 2)),
                       "weak_fraction": float(np.mean(result.confidence == 1)),
                       "bright_fraction": float(np.mean(result.gray >= 250)),
                       "floor_fresh": floor.fresh, "floor_usable": floor.usable,
                       "floor_height_m": floor.height_m,
                       "floor_inliers": floor.inliers, "floor_candidates": floor.candidates,
                       "tracking_ok": pose.tracking_ok, "tracking_method": tracker.method,
                       "tracking_rejection": tracker.rejection,
                       "tracking_inliers": tracker.inliers, "reset_map": tracker.reset_map,
                       "pose_x_m": pose.x_m, "pose_z_m": pose.z_m,
                       "pose_yaw_deg": float(np.rad2deg(pose.yaw_rad)),
                       "integrated": integration["integrated"],
                       "observed_obstacle_cells": integration["obstacle_cells"],
                       "observed_floor_cells": integration["floor_cells"],
                       "carved_obstacle_cells": integration["carved_cells"],
                       "traversed_cells": integration["traversed_cells"],
                       "far_hints": integration["far_hints"],
                       "map_obstacles": int(np.count_nonzero(states == grid.OBSTACLE)),
                       "map_free": int(np.count_nonzero(states == grid.FREE)),
                       "map_danger_unknown": int(np.count_nonzero(states == grid.DANGER)),
                       "near_forward": grid.forward_coverage(
                           config["map"]["near_corridor_diagnostic_length_m"],
                           config["map"]["near_corridor_diagnostic_width_m"]),
                       "depth_ms": depth_ms, "mapping_ms": mapping_ms}
                frames.append(row)
                if timestamp_s >= next_debug_s:
                    ImageTools.save(output / f"map_{index:06d}.png", render_map(grid))
                    ImageTools.save(output / f"depth_{index:06d}.png",
                                    ImageTools.depth_image(result))
                    next_debug_s = timestamp_s + debug_interval_s
                last_result = result
        if last_result is not None:
            ImageTools.save(output / "map_final.png", render_map(grid))
            ImageTools.save(output / "depth_final.png", ImageTools.depth_image(last_result))
    except Exception as error:
        metrics["status"] = "error"
        metrics["reason"] = f"{type(error).__name__}: {error}"

    states = grid.state_grid()
    solid_mean = float(np.mean([r["solid_fraction"] for r in frames])) if frames else None
    frame_count = len(frames)
    suitability = {
        "solid_depth_coverage": bool(frame_count and solid_mean >=
                                      config["map"]["accept_min_solid_fraction"]),
        "floor_coverage": bool(frame_count and sum(r["floor_usable"] for r in frames) / frame_count >=
                               config["map"]["accept_min_floor_usable_fraction"]),
        "tracking_continuity": bool(frame_count and sum(r["tracking_ok"] for r in frames) / frame_count >=
                                    config["map"]["accept_min_tracking_fraction"]),
    }
    fixture_checkpoint = _fixture_checkpoint(path, config, metadata, frames)
    checks = {
        "decoded_all_advertised_frames": (metadata["advertised_frames"] <= 0 or
                                          len(frames) == metadata["advertised_frames"]),
        "no_integration_while_tracking_lost": safety_ok,
        "no_integration_without_fresh_floor": all(
            not row["integrated"] or row["floor_fresh"] for row in frames),
        "mapping_p95_under_30_ms": bool(mapping_times and
                                        np.percentile(mapping_times, 95) < 30.0),
        "near_free_requires_floor_evidence": all(
            row["near_forward"]["free"] <=
            row["near_forward"]["floor_evidence_cells"] for row in frames),
    }
    if fixture_checkpoint["matched"]:
        checks.update(fixture_checkpoint["checks"])
    metrics.update({
        "frames_decoded": len(frames), "duration_s": last_t,
        "source_timestamp_fallbacks": source_fallbacks,
        "floor_fresh_frames": sum(row["floor_fresh"] for row in frames),
        "floor_usable_frames": sum(row["floor_usable"] for row in frames),
        "tracking_ok_frames": sum(row["tracking_ok"] for row in frames),
        "tracking_resets": tracker.reset_count,
        "integrated_frames": sum(row["integrated"] for row in frames),
        "solid_fraction_mean": solid_mean,
        "weak_fraction_mean": float(np.mean([r["weak_fraction"] for r in frames])) if frames else None,
        "bright_fraction_mean": float(np.mean([r["bright_fraction"] for r in frames])) if frames else None,
        "floor_height_median": float(np.median(floor_heights)) if floor_heights else None,
        "depth_ms": _summary(depth_times), "mapping_ms": _summary(mapping_times),
        "final_cells": {"unknown": int(np.count_nonzero(states == grid.UNKNOWN)),
                        "free": int(np.count_nonzero(states == grid.FREE)),
                        "obstacle": int(np.count_nonzero(states == grid.OBSTACLE)),
                        "danger_unknown": int(np.count_nonzero(states == grid.DANGER))},
        "checks": checks,
        "fixture_checkpoint": fixture_checkpoint,
        "suitability": suitability,
        "limitations": ["The apartment clip has no labeled obstacle positions.",
                        "No free space is inferred through weak, missing, or distant depth.",
                        "A missing expected floor patch remains unknown; only measured below-floor points receive a danger warning.",
                        "Stereo eye order cannot be proven from recording metadata alone."],
        "frames": frames,
    })
    if metrics["status"] != "error":
        if not all(suitability.values()):
            metrics["status"] = "unsuitable"
            metrics["reason"] = "insufficient " + ", ".join(
                key.replace("_", " ") for key, passed in suitability.items() if not passed)
        else:
            metrics["status"] = ("processed" if all(metrics["checks"].values())
                                 else "regression")
    _save_json(output / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, action="append",
                        help="process these recordings instead of auto-discovering")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--debug-interval-s", type=float,
                        help="seconds between saved debug images (default: map config)")
    args = parser.parse_args()
    config = load_config()
    recordings = args.recording or discover_recordings()
    output_root = args.output_dir or Path(config["map"]["output_dir"])
    reports = []
    for recording in recordings:
        report = run_recording(recording, config, output_root, args.debug_interval_s)
        reports.append({key: report.get(key) for key in
                        ("recording", "status", "reason", "frames_decoded", "duration_s",
                         "floor_fresh_frames", "tracking_ok_frames", "tracking_resets",
                         "integrated_frames", "final_cells", "mapping_ms", "checks",
                         "suitability")})
        print(f"{Path(recording).name}: {report['status']}; "
              f"{report.get('frames_decoded', 0)} frames; {report.get('reason') or 'see metrics.json'}")
    _save_json(output_root / "summary.json", {"recordings": reports,
                                               "discovered_count": len(recordings)})
    if not reports or not any(report["status"] == "processed" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
