"""Show recorded stereo pairs one at a time, without tracking or fusion.

Examples:
  python 09_view_recording_frame.py recordings/recording_20261105_131121 --time-seconds 20
  python 09_view_recording_frame.py recordings/recording_20261105_131121 --frame-id 1234
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import open3d as o3d

from common import Calibration, Rectifier, StereoMatcher

processor = importlib.import_module("08_process_recording")
single_frame = importlib.import_module("05_single_frame_test")


def choose_frame(rows, frame_id, seconds):
    if frame_id is not None:
        for row in rows:
            if row["frame_id"] == frame_id:
                return row
        raise ValueError("Frame ID %d is not present in this recording" % frame_id)
    target = rows[0]["monotonic_ns"] + int(seconds * 1e9)
    return min(rows, key=lambda row: abs(row["monotonic_ns"] - target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame-id", type=int, help="captured frame ID from frames.csv")
    selection.add_argument("--time-seconds", type=float, default=0.0,
                           help="select nearest frame this many seconds after recording start")
    parser.add_argument("--no-lr-check", action="store_true", help="show original live depth without independent left/right filtering")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--save-ply", type=Path, help="optionally save this one-frame cloud")
    parser.add_argument("--no-view", action="store_true", help="print diagnostics and optionally save without opening a window")
    args = parser.parse_args()
    if args.time_seconds < 0 or (args.frame_id is not None and args.frame_id < 1):
        parser.error("time and frame ID cannot be negative")

    meta, rows, cfg, calibration, frame_bytes = processor.load_recording(args.recording)
    row = choose_frame(rows, args.frame_id, args.time_seconds)
    position = next(i for i, item in enumerate(rows) if item["frame_id"] == row["frame_id"])
    calib = Calibration(calibration)
    if tuple(meta["eye_shape_hw"]) != (calib.size[1], calib.size[0]):
        raise ValueError("Recording dimensions differ from calibration")
    rect = Rectifier(calib, cfg["stereo"]["scale"])
    matcher = StereoMatcher(cfg["stereo"], rect.fx, rect.baseline)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(rect.out_size[0], rect.out_size[1],
                                                  rect.fx, rect.fy, rect.cx, rect.cy)

    def load_frame(item):
        with (args.recording / "frames.bin").open("rb") as handle:
            handle.seek(item["byte_offset"])
            pair = np.frombuffer(handle.read(frame_bytes), dtype=np.uint8)
        if pair.size != frame_bytes:
            raise ValueError("Frame %d is incomplete" % item["frame_id"])
        left, right = pair.reshape(2, *meta["eye_shape_hw"])
        rl, rr = rect.rectify(left, right)
        disparity, depth = matcher.compute(rl, rr)
        before = int(np.count_nonzero(depth))
        if not args.no_lr_check:
            consistent, _ = single_frame.left_right_consistency(
                rl, rr, disparity, cfg["stereo"], matcher.nd)
            depth[~consistent] = 0
        after = int(np.count_nonzero(depth))
        result = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            intrinsic, depth_scale=1.0, depth_trunc=float(cfg["scan"]["depth_trunc"]), stride=1)
        valid = (depth > 0) & (depth < float(cfg["scan"]["depth_trunc"]))
        intensity = rl[valid].astype(np.float64) / 255.0
        if len(result.points) != len(intensity):
            raise RuntimeError("Point/color count mismatch")
        result.colors = o3d.utility.Vector3dVector(np.column_stack((intensity, intensity, intensity)))
        if cfg["scan"].get("export_y_up", True):
            result.transform(np.diag([1.0, -1.0, -1.0, 1.0]))
        offset_s = (item["monotonic_ns"] - rows[0]["monotonic_ns"]) / 1e9
        print("Frame %d/%d at %.3f s | depth-valid %.1f%% | LR %.1f%% | displayed %d points" % (
            item["frame_id"], rows[-1]["frame_id"], offset_s, 100 * before / depth.size,
            100 * after / max(1, before), len(result.points)), flush=True)
        return result

    cloud = load_frame(row)
    if args.save_ply:
        args.save_ply.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(args.save_ply), cloud):
            raise RuntimeError("Could not save point cloud")
        print("Saved", args.save_ply)
    if args.no_view:
        return 0

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window("Recorded frame  [Left/Right 1 | J/L 2 | F fit | Q quit]", 1280, 800):
        raise RuntimeError("Could not open point cloud window")
    vis.add_geometry(cloud)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    vis.add_geometry(frame)
    vis.get_render_option().point_size = max(1.0, args.point_size)
    vis.get_render_option().background_color = np.array([0.04, 0.04, 0.06])

    def fit(view):
        view.reset_view_point(True)
        control = view.get_view_control()
        control.set_front([0, 0, 1] if cfg["scan"].get("export_y_up", True) else [0, 0, -1])
        control.set_up([0, 1, 0] if cfg["scan"].get("export_y_up", True) else [0, -1, 0])
        control.set_lookat(cloud.get_center())
        control.set_zoom(0.72)
        return False

    current = {"position": position}
    held = []
    pending = []

    def key_action(delta):
        def callback(_view, action, _mods):
            if action == 1:  # GLFW_PRESS: one step even for a short tap
                if delta in held:
                    held.remove(delta)
                held.append(delta)
                pending.append(delta)
            elif action == 0 and delta in held:  # GLFW_RELEASE
                held.remove(delta)
            # GLFW_REPEAT is ignored: the render loop advances held keys
            # only after the preceding frame has finished loading.
            return False
        return callback

    vis.register_key_callback(ord("F"), fit)
    vis.register_key_action_callback(263, key_action(-1))  # GLFW_KEY_LEFT
    vis.register_key_action_callback(262, key_action(1))   # GLFW_KEY_RIGHT
    vis.register_key_action_callback(ord("J"), key_action(-2))
    vis.register_key_action_callback(ord("L"), key_action(2))
    vis.register_key_action_callback(ord("U"), key_action(-4))
    vis.register_key_action_callback(ord("O"), key_action(4))
    fit(vis)
    while vis.poll_events():
        delta = pending.pop(0) if pending else (held[-1] if held else None)
        if delta is not None:
            target = current["position"] + delta
            if 0 <= target < len(rows):
                next_cloud = load_frame(rows[target])
                cloud.points = next_cloud.points
                cloud.colors = next_cloud.colors
                current["position"] = target
                vis.update_geometry(cloud)
        vis.update_renderer()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
