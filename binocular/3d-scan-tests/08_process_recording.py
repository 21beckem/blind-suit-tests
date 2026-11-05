"""Replay a raw stereo recording into one point cloud, trajectory, and diagnostics.

Example: python 08_process_recording.py recordings/recording_20261105_131121 --fps 15
Offline FPS selects captured timestamps; it never sleeps or fabricates frames.
"""
import argparse
import csv
import hashlib
import importlib
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from common import DEFAULTS, Calibration, Rectifier, StereoMatcher, _merge
from scanner_core import ScanSession
left_right_consistency = importlib.import_module("05_single_frame_test").left_right_consistency


def load_recording(directory):
    meta = json.loads((directory / "metadata.json").read_text())
    if meta.get("format_version") != 1:
        raise ValueError("Unsupported recording format version")
    shape = meta.get("eye_shape_hw")
    if not shape or len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Recording has no valid eye shape")
    frame_bytes = 2 * int(shape[0]) * int(shape[1])
    path = directory / "frames.bin"
    size = path.stat().st_size
    rows = []
    with (directory / "frames.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            entry = {key: int(row[key]) for key in
                     ("frame_id", "monotonic_ns", "wall_time_ns", "grab_spread_ns", "byte_offset")}
            if entry["byte_offset"] < 0 or entry["byte_offset"] + frame_bytes > size:
                raise ValueError("Frame %d extends beyond frames.bin" % entry["frame_id"])
            if rows and (entry["frame_id"] <= rows[-1]["frame_id"] or
                         entry["monotonic_ns"] <= rows[-1]["monotonic_ns"] or
                         entry["byte_offset"] < rows[-1]["byte_offset"] + frame_bytes):
                raise ValueError("Frame index is not strictly ordered")
            rows.append(entry)
    if not rows:
        raise ValueError("Recording has no indexed frames")
    if meta.get("stats", {}).get("written") != len(rows):
        raise ValueError("Metadata and frame index disagree")
    calibration = directory / "calibration.npz"
    expected_hash = meta.get("calibration_sha256")
    if expected_hash and hashlib.sha256(calibration.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Recorded calibration checksum mismatch")
    cfg = _merge(deepcopy(DEFAULTS), json.loads((directory / "config.json").read_text()))
    return meta, rows, cfg, calibration, frame_bytes


def select_frames(rows, fps, start_s, duration_s, max_frames):
    origin = rows[0]["monotonic_ns"]
    start_ns = origin + int(start_s * 1e9)
    end_ns = None if duration_s is None else start_ns + int(duration_s * 1e9)
    interval_ns = int(1e9 / fps)
    chosen = []
    next_ns = start_ns
    for row in rows:
        stamp = row["monotonic_ns"]
        if stamp < start_ns:
            continue
        if end_ns is not None and stamp >= end_ns:
            break
        if stamp >= next_ns:
            chosen.append(row)
            next_ns = stamp + interval_ns
            if max_frames is not None and len(chosen) >= max_frames:
                break
    return chosen


def process(directory, output, fps, start_s, duration_s, max_frames, fusion, lr_check):
    meta, rows, cfg, calibration, frame_bytes = load_recording(directory)
    chosen = select_frames(rows, fps, start_s, duration_s, max_frames)
    if not chosen:
        raise ValueError("No frames in the selected time range")
    calib = Calibration(calibration)
    rect = Rectifier(calib, cfg["stereo"]["scale"])
    if tuple(meta["eye_shape_hw"]) != (calib.size[1], calib.size[0]):
        raise ValueError("Recording eye dimensions differ from its calibration")
    matcher = StereoMatcher(cfg["stereo"], rect.fx, rect.baseline)
    width, height = rect.out_size
    session = ScanSession(rect.fx, rect.fy, rect.cx, rect.cy, width, height, cfg["scan"],
                          min_depth=cfg["stereo"]["min_depth"],
                          max_depth=cfg["stereo"]["max_depth"], fusion=fusion)
    output.mkdir(parents=True, exist_ok=False)
    (output / "settings.json").write_text(json.dumps({"recording": str(directory.resolve()),
        "requested_processing_fps": fps, "start_s": start_s, "duration_s": duration_s,
        "max_frames": max_frames, "fusion": session.fusion, "lr_check": lr_check,
        "config": cfg}, indent=2) + "\n")
    timings = []
    states = {}
    accepted = 0
    first_stamp = chosen[0]["monotonic_ns"]
    previous_stamp = None
    total_start = time.perf_counter()
    with (directory / "frames.bin").open("rb") as data, (output / "frames.csv").open("w", newline="") as report:
        writer = csv.writer(report)
        writer.writerow(["frame_id", "time_s", "gap_ms", "state", "integrated", "valid_pct",
                         "lr_consistent_pct", "x_m", "y_m", "z_m", "rectify_ms", "depth_ms",
                         "track_fuse_ms", "total_ms"])
        for n, row in enumerate(chosen, 1):
            data.seek(row["byte_offset"])
            pair = np.frombuffer(data.read(frame_bytes), dtype=np.uint8)
            if pair.size != frame_bytes:
                raise ValueError("Short read at frame %d" % row["frame_id"])
            left, right = pair.reshape(2, *meta["eye_shape_hw"])
            t0 = time.perf_counter()
            rl, rr = rect.rectify(left, right)
            t1 = time.perf_counter()
            disparity, depth = matcher.compute(rl, rr)
            lr_pct = None
            if lr_check:
                consistent, _ = left_right_consistency(rl, rr, disparity, cfg["stereo"], matcher.nd)
                valid_count = np.count_nonzero(depth)
                lr_pct = 100 * np.count_nonzero((depth > 0) & consistent) / max(1, valid_count)
                depth[~consistent] = 0
            t2 = time.perf_counter()
            info = session.process(rl, depth)
            t3 = time.perf_counter()
            state = info["state"]
            states[state] = states.get(state, 0) + 1
            accepted += bool(info["integrated"])
            gap_ms = None if previous_stamp is None else (row["monotonic_ns"] - previous_stamp) / 1e6
            previous_stamp = row["monotonic_ns"]
            dt = [(t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t3-t0)*1000]
            timings.append(dt)
            writer.writerow([row["frame_id"], (row["monotonic_ns"]-first_stamp)/1e9,
                             gap_ms, state, int(info["integrated"]), info["valid"]*100,
                             lr_pct, *session.pose[:3, 3], *dt])
            if n % 25 == 0 or n == len(chosen):
                print("%d/%d | %s | keyframes %d | depth %.1f%% | %.1f ms/frame" % (
                    n, len(chosen), state, accepted, info["valid"]*100,
                    np.mean(np.asarray(timings)[-25:, 3])), flush=True)
    points, colors = session.extract_cloud()
    if len(points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        poses = np.asarray(session.keyframe_poses)
        if cfg["scan"].get("export_y_up", True):
            flip = np.diag([1.0, -1.0, -1.0, 1.0])
            cloud.transform(flip)
            poses = flip @ poses
        o3d.io.write_point_cloud(str(output / "scan_replay_points.ply"), cloud)
        np.save(output / "scan_replay_trajectory.npy", poses)
    array = np.asarray(timings)
    summary = {"recording_frames": len(rows), "selected_frames": len(chosen),
               "recording_duration_s": (rows[-1]["monotonic_ns"]-rows[0]["monotonic_ns"])/1e9,
               "selected_duration_s": (chosen[-1]["monotonic_ns"]-first_stamp)/1e9,
               "capture_drops": meta.get("stats", {}).get("dropped_queue"),
               "states": states, "integrated_keyframes": accepted, "points": len(points),
               "fusion": session.fusion, "elapsed_processing_s": time.perf_counter()-total_start,
               "mean_ms": dict(zip(("rectify", "depth", "track_fuse", "total"), array.mean(axis=0).tolist())),
               "p95_ms": dict(zip(("rectify", "depth", "track_fuse", "total"), np.percentile(array, 95, axis=0).tolist()))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(points) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=float, default=15.0, help="maximum selected camera FPS")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fusion", choices=("tsdf", "points"), default=None)
    parser.add_argument("--no-lr-check", action="store_true", help="replay original live depth behavior")
    args = parser.parse_args()
    if args.fps <= 0 or args.start_seconds < 0 or (args.duration_seconds is not None and args.duration_seconds <= 0) or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("FPS, duration, and max-frames must be positive; start-seconds cannot be negative")
    output = args.output or Path(__file__).resolve().parent / "processing_runs" / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    return process(args.recording, output, args.fps, args.start_seconds,
                   args.duration_seconds, args.max_frames, args.fusion, not args.no_lr_check)


if __name__ == "__main__":
    raise SystemExit(main())
