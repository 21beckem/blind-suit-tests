"""Step 4 - live 3D room scan.

    python 04_live_scan.py

Walk/pan SLOWLY and smoothly. Keep textured things (furniture, posters, door frames) in view: blank walls
give the stereo matcher nothing to lock onto. Overlap what you already scanned so tracking stays locked.

Keys (camera window or 3D window):  SPACE pause | R reset map | S save now | Q quit (auto-saves)
3D window only: F fit the whole scan | A toggle automatic fitting as the scan grows
Output goes to ./scans/ : *_mesh.ply, *_points.ply, *_trajectory.npy
"""
import argparse
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import open3d as o3d

from common import (SCAN_DIR, Calibration, Rectifier, StereoCamera, StereoMatcher, load_config)
from scanner_core import ScanSession

STATE_COLORS = {"TRACKING": (0, 255, 0), "INIT": (0, 255, 255), "NO DEPTH": (0, 120, 255), "LOST": (0, 0, 255)}


class ViewFramer:
    """Tracks map growth so the Open3D camera can frame real geometry, not the startup dummy point."""

    def __init__(self):
        self.auto = True
        self.lo = None
        self.hi = None

    def reset(self):
        self.lo = None
        self.hi = None

    @staticmethod
    def bounds(points):
        points = np.asarray(points)
        if len(points) == 0:
            return None
        finite = points[np.all(np.isfinite(points), axis=1)]
        if len(finite) == 0:
            return None
        return finite.min(axis=0), finite.max(axis=0)

    def should_fit(self, points):
        bounds = self.bounds(points)
        if bounds is None:
            return False
        lo, hi = bounds
        if self.lo is None:
            return True
        old_span = np.maximum(self.hi - self.lo, 0.10)
        union_span = np.maximum(np.maximum(self.hi, hi) - np.minimum(self.lo, lo), 0.10)
        return bool(np.max(union_span / old_span) >= 1.25)

    def fitted(self, points):
        bounds = self.bounds(points)
        if bounds is not None:
            self.lo, self.hi = bounds


def fit_scan_view(vis, points, framer):
    """Fit the complete live cloud while preserving the natural OpenCV camera orientation."""
    bounds = framer.bounds(points)
    if bounds is None:
        return False
    lo, hi = bounds
    vis.reset_view_point(True)  # recompute scale/clipping from the real cloud
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])   # camera behind the origin, looking along live scan +Z
    ctr.set_up([0, -1, 0])      # live OpenCV Y points down
    ctr.set_lookat(((lo + hi) * 0.5).tolist())
    ctr.set_zoom(0.72)
    framer.fitted(points)
    return True


class Controls:
    def __init__(self):
        self.quit = False
        self.paused = False
        self.reset_req = False
        self.save_req = False
        self.save_on_exit = True


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.cloud = None          # (pts, cols) pending for the viewer
        self.clear = False
        self.pose = np.eye(4)
        self.traj = np.zeros((0, 3))
        self.hud = None
        self.message = ""
        self.error = None

    def log(self, text):
        print(text)
        with self.lock:
            self.message = text


def colorize_disp(disp, nd):
    v = np.clip(disp * (255.0 / nd), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
    out[disp <= 0.5] = 0
    return out


def make_hud(rl, disp, nd, state, stats, message):
    h = 340
    w = int(rl.shape[1] * h / rl.shape[0])
    left = cv2.resize(cv2.cvtColor(rl, cv2.COLOR_GRAY2BGR), (w, h))
    dsp = cv2.resize(colorize_disp(disp, nd), (w, h), interpolation=cv2.INTER_NEAREST)
    col = STATE_COLORS.get(state, (255, 255, 255))
    cv2.putText(left, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.putText(left, stats, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if message:
        cv2.putText(dsp, message[-60:], (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.hstack([left, dsp])


def worker(cam, rect, matcher, session, ctrl, shared, cfg_scan):
    last_id, fps, t_prev = -1, 0.0, time.time()
    last_extract, interval = 0.0, float(cfg_scan["live_interval"])
    n_traj = 0
    try:
        while not ctrl.quit:
            if ctrl.reset_req:
                ctrl.reset_req = False
                session.reset()
                n_traj = 0
                with shared.lock:
                    shared.clear, shared.cloud, shared.traj = True, None, np.zeros((0, 3))
                    shared.pose = np.eye(4)
                shared.log("Map reset")
            if ctrl.save_req:
                ctrl.save_req = False
                shared.log("Saving ...")
                base = session.save(str(SCAN_DIR), log=shared.log)
                shared.log("Saved " + base if base else "Nothing to save yet")

            item = cam.read(last_id, 0.5)
            if item is None:
                continue
            left, right, _t, last_id = item
            rl, rr = rect.rectify(left, right)
            disp, depth = matcher.compute(rl, rr)

            if ctrl.paused:
                info = {"state": "PAUSED", "valid": float((depth > 0).mean())}
            else:
                info = session.process(rl, depth)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if fps else 1.0 / max(now - t_prev, 1e-3)
            t_prev = now

            if session.dirty and now - last_extract >= interval:
                t0 = time.time()
                pts, cols = session.extract_cloud()
                interval = max(float(cfg_scan["live_interval"]), (time.time() - t0) * 4.0)  # auto-throttle
                last_extract = time.time()
                with shared.lock:
                    shared.cloud = (pts, cols)

            stats = "%.1f fps (cam %.0f) | keyframes %d | depth %.0f%% | fusion %s" % (
                fps, cam.fps, session.n_integrated, info["valid"] * 100, session.fusion)
            hud = make_hud(rl, disp, matcher.nd, info.get("state", session.state), stats, shared.message)
            with shared.lock:
                shared.hud = hud
                shared.pose = session.pose.copy()
                if len(session.keyframe_poses) != n_traj:
                    n_traj = len(session.keyframe_poses)
                    shared.traj = np.array(session.keyframe_poses)[:, :3, 3]
    except Exception:  # noqa: BLE001
        shared.error = traceback.format_exc()
        ctrl.quit = True
        ctrl.save_on_exit = False
        return
    if ctrl.save_on_exit and session.n_integrated > 0:
        shared.log("Saving final scan ...")
        session.save(str(SCAN_DIR), log=shared.log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true", help="do not auto-save on exit")
    ap.add_argument("--fusion", choices=["tsdf", "points"], default=None, help="force a fusion backend")
    args = ap.parse_args()

    cfg = load_config()
    calib = Calibration()
    rect = Rectifier(calib, cfg["stereo"]["scale"])
    matcher = StereoMatcher(cfg["stereo"], rect.fx, rect.baseline)
    w, h = rect.out_size
    session = ScanSession(rect.fx, rect.fy, rect.cx, rect.cy, w, h, cfg["scan"],
                          min_depth=cfg["stereo"]["min_depth"], max_depth=cfg["stereo"]["max_depth"],
                          fusion=args.fusion)
    print("Processing %dx%d | baseline %.1f mm | closest measurable %.2f m | fusion: %s" % (
        w, h, rect.baseline * 1000, matcher.closest_depth(), session.fusion))

    cam = StereoCamera(cfg["camera"]).start()
    ctrl, shared = Controls(), Shared()
    ctrl.save_on_exit = not args.no_save
    th = threading.Thread(target=worker, args=(cam, rect, matcher, session, ctrl, shared, cfg["scan"]), daemon=True)
    th.start()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Live room scan  [F fit | A auto-fit | R reset | S save | SPACE pause | Q quit]", 1280, 800)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.colors = o3d.utility.Vector3dVector(np.ones((1, 3)))
    vis.add_geometry(pcd)
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    vis.add_geometry(cam_frame)
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 0.001]], dtype=np.float64))
    traj.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    traj.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.3, 0.1]]))
    vis.add_geometry(traj)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.04, 0.04, 0.06])
    # The startup geometry is only a tiny dummy point. Do not set the final view from its bounds;
    # fit again as soon as the first real cloud arrives.
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    framer = ViewFramer()
    view_action = {"fit": False}

    def key_cb(action):
        def f(_v):
            action()
            return False
        return f

    def toggle_pause():
        ctrl.paused = not ctrl.paused

    def do_quit():
        ctrl.quit = True

    def do_reset():
        ctrl.reset_req = True

    def do_save():
        ctrl.save_req = True

    def do_fit():
        view_action["fit"] = True

    def toggle_auto_fit():
        framer.auto = not framer.auto
        shared.log("Automatic view fitting " + ("ON" if framer.auto else "OFF"))

    for ch, act in (("Q", do_quit), ("R", do_reset), ("S", do_save), (" ", toggle_pause),
                    ("F", do_fit), ("A", toggle_auto_fit)):
        vis.register_key_callback(ord(ch), key_cb(act))

    drawn_pose = np.eye(4)
    shown_traj = 0
    try:
        while not ctrl.quit:
            if not vis.poll_events():
                ctrl.quit = True
                break
            with shared.lock:
                cloud, shared.cloud = shared.cloud, None
                clear, shared.clear = shared.clear, False
                pose = shared.pose.copy()
                tr = shared.traj
                hud = shared.hud
            if clear:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
                pcd.colors = o3d.utility.Vector3dVector(np.ones((1, 3)))
                vis.update_geometry(pcd)
                shown_traj = 0
                framer.reset()
            if cloud is not None and len(cloud[0]) > 0:
                pcd.points = o3d.utility.Vector3dVector(cloud[0].astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(cloud[1].astype(np.float64))
                vis.update_geometry(pcd)
                if framer.auto and framer.should_fit(cloud[0]):
                    view_action["fit"] = True
            if view_action["fit"]:
                view_action["fit"] = False
                current_points = np.asarray(pcd.points)
                if fit_scan_view(vis, current_points, framer):
                    shared.log("View fitted to the complete scan")
            if not np.allclose(pose, drawn_pose):
                cam_frame.transform(pose @ np.linalg.inv(drawn_pose))
                drawn_pose = pose
                vis.update_geometry(cam_frame)
            if len(tr) >= 2 and len(tr) != shown_traj:
                shown_traj = len(tr)
                traj.points = o3d.utility.Vector3dVector(tr.astype(np.float64))
                traj.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(len(tr) - 1)]))
                traj.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.3, 0.1], (len(tr) - 1, 1)))
                vis.update_geometry(traj)
            vis.update_renderer()
            if hud is not None:
                cv2.imshow("camera (left | disparity)  SPACE pause | R reset | S save | Q quit", hud)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q"), 27):
                ctrl.quit = True
            elif k in (ord("r"), ord("R")):
                ctrl.reset_req = True
            elif k in (ord("s"), ord("S")):
                ctrl.save_req = True
            elif k == ord(" "):
                ctrl.paused = not ctrl.paused
            time.sleep(0.004)
    finally:
        ctrl.quit = True
        th.join(timeout=120)
        cam.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
    if shared.error:
        print("\nScanner stopped because of an error:\n" + shared.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
