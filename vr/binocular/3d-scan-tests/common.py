"""Shared helpers: config, camera capture, calibration loading, rectification, stereo depth."""
import os

# Must be set before importing cv2: makes MSMF camera startup much faster on Windows.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
CALIB_PATH = ROOT / "calibration.npz"
CALIB_IMG_DIR = ROOT / "calib_images"
SCAN_DIR = ROOT / "scans"

DEFAULTS = {
    "camera": {
        # "sbs"  = one USB device that outputs left+right side by side in one frame (most dual-lens modules)
        # "dual" = two separate USB devices (index = left, index_right = right)
        "mode": "sbs",
        "index": 0,
        "index_right": 1,
        "backend": "dshow",      # dshow | msmf | any
        "width": 2560,           # sbs: COMBINED width. dual: per-eye width.
        "height": 800,
        "fps": 60,
        "fourcc": "MJPG",
        "swap_eyes": False,
        "auto_exposure": None,   # DirectShow: 0.25 = manual, 0.75 = auto (driver dependent)
        "exposure": None,        # DirectShow: log2 seconds, e.g. -6 = 1/64 s (driver dependent)
    },
    "calibration": {
        "board_cols": 9,         # INNER corners across (odd x even avoids orientation ambiguity)
        "board_rows": 6,         # INNER corners down
        "square_mm": 25.0,       # measure your printed square with a ruler!
        "num_views": 30,
    },
    "stereo": {
        "scale": 0.5,            # processing scale applied after rectification
        "num_disparities": 96,   # multiple of 16
        "block_size": 5,
        "uniqueness": 10,
        "speckle_window": 100,
        "speckle_range": 2,
        "disp12_max_diff": 1,
        "pre_filter_cap": 31,
        "min_depth": 0.25,       # metres
        "max_depth": 4.0,        # metres
        "edge_filter": 0.10,     # drop "flying pixels" where neighbours differ by > this fraction of depth
        "clahe": False,
    },
    "scan": {
        "voxel": 0.02,           # TSDF voxel size (m)
        "sdf_trunc": 0.08,
        "depth_trunc": 3.5,      # ignore depth beyond this when fusing / tracking (m)
        "key_trans": 0.04,       # new keyframe after moving this far (m)
        "key_rot_deg": 4.0,
        "max_step": 0.15,        # reject pose jumps larger than this per frame (m)
        "max_rot_deg": 25.0,
        "min_valid_ratio": 0.04, # need at least this fraction of pixels with depth
        "odo_depth_diff": 0.03,  # tight on purpose: stereo depth is noisy; 0.07 made tracking drift in tests
        "odo_term": "hybrid",    # "hybrid" (colour+depth) or "color" (photometric only; try if tracking drifts)
        "max_keyframes": 8,
        "live_interval": 1.0,    # seconds between live point-cloud refreshes (auto-slows if extraction is slow)
        "export_y_up": True,     # export with Y up / Z back (graphics convention) instead of OpenCV (Y down)
        "cleanup_min_triangles": 300,
    },
}


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    if CONFIG_PATH.exists():
        try:
            _merge(cfg, json.loads(CONFIG_PATH.read_text()))
        except Exception as e:  # noqa: BLE001
            print("WARNING: could not read config.json (%s); using defaults." % e)
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def eye_size(cam_cfg):
    """Expected (width, height) of ONE eye for the configured camera mode."""
    if cam_cfg["mode"] == "sbs":
        return cam_cfg["width"] // 2, cam_cfg["height"]
    return cam_cfg["width"], cam_cfg["height"]


# --------------------------------------------------------------------------- camera
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


def open_capture(index, backend="dshow", width=None, height=None, fps=None,
                 fourcc="MJPG", exposure=None, auto_exposure=None):
    cap = cv2.VideoCapture(int(index), BACKENDS.get(backend, cv2.CAP_ANY))
    if not cap.isOpened():
        cap.release()
        return None
    # FOURCC first: on DirectShow the format must be chosen before the resolution.
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if auto_exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure))
    if exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    return cap


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class StereoCamera:
    """Background-threaded capture that always hands back the NEWEST left/right gray pair."""

    def __init__(self, cam_cfg):
        self.cfg = cam_cfg
        self.caps = []
        self.cond = threading.Condition()
        self.latest = None          # (left, right, timestamp, frame_id)
        self.fid = 0
        self.fps = 0.0
        self.grab_spread_ms = 0.0   # dual mode only
        self._last_t = None
        self.running = False
        self.thread = None

    def start(self):
        c = self.cfg
        kw = dict(backend=c["backend"], fourcc=c["fourcc"], fps=c["fps"],
                  exposure=c.get("exposure"), auto_exposure=c.get("auto_exposure"))
        if c["mode"] == "sbs":
            cap = open_capture(c["index"], width=c["width"], height=c["height"], **kw)
            if cap is None:
                raise RuntimeError("Could not open camera index %s. Run 01_probe.py." % c["index"])
            self.caps = [cap]
            target = self._loop_sbs
        else:
            capl = open_capture(c["index"], width=c["width"], height=c["height"], **kw)
            capr = open_capture(c["index_right"], width=c["width"], height=c["height"], **kw)
            if capl is None or capr is None:
                for x in (capl, capr):
                    if x is not None:
                        x.release()
                raise RuntimeError("Could not open both cameras (%s, %s). Run 01_probe.py."
                                   % (c["index"], c["index_right"]))
            self.caps = [capl, capr]
            target = self._loop_dual
        self.running = True
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()
        return self

    def _publish(self, left, right, t):
        with self.cond:
            self.fid += 1
            if self._last_t is not None and t > self._last_t:
                inst = 1.0 / (t - self._last_t)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            self._last_t = t
            self.latest = (left, right, t, self.fid)
            self.cond.notify_all()

    def _loop_sbs(self):
        cap = self.caps[0]
        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            t = time.time()
            g = to_gray(frame)
            half = g.shape[1] // 2
            left, right = g[:, :half], g[:, half:2 * half]
            if self.cfg.get("swap_eyes"):
                left, right = right, left
            self._publish(np.ascontiguousarray(left), np.ascontiguousarray(right), t)

    def _loop_dual(self):
        capl, capr = self.caps
        while self.running:
            t0 = time.time()
            okl = capl.grab()
            okr = capr.grab()
            t1 = time.time()
            if not (okl and okr):
                time.sleep(0.005)
                continue
            okl, fl = capl.retrieve()
            okr, fr = capr.retrieve()
            if not (okl and okr):
                continue
            self.grab_spread_ms = (t1 - t0) * 1000.0
            left, right = to_gray(fl), to_gray(fr)
            if self.cfg.get("swap_eyes"):
                left, right = right, left
            self._publish(left, right, t1)

    def read(self, last_id=-1, timeout=1.0):
        """Block until a frame newer than last_id exists. Returns (left, right, t, id) or None."""
        end = time.time() + timeout
        with self.cond:
            while self.latest is None or self.latest[3] == last_id:
                remaining = end - time.time()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)
            return self.latest

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        for c in self.caps:
            c.release()
        self.caps = []


# --------------------------------------------------------------------------- calibration / rectification
class Calibration:
    def __init__(self, path=CALIB_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("%s not found. Run 02_calibrate.py first." % path.name)
        d = np.load(str(path))
        for k in ("K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2", "Q"):
            setattr(self, k, d[k])
        self.size = (int(d["image_size"][0]), int(d["image_size"][1]))  # (w, h) of ONE eye
        self.rms = float(d["rms"]) if "rms" in d else float("nan")
        self.baseline = float(-self.P2[0, 3] / self.P2[0, 0])


class Rectifier:
    """Undistort + rectify both eyes, then optionally downscale. Exposes scaled intrinsics."""

    def __init__(self, calib, scale=1.0):
        self.calib = calib
        w, h = calib.size
        self.size = (w, h)
        self.mapL = cv2.initUndistortRectifyMap(calib.K1, calib.D1, calib.R1, calib.P1, (w, h), cv2.CV_16SC2)
        self.mapR = cv2.initUndistortRectifyMap(calib.K2, calib.D2, calib.R2, calib.P2, (w, h), cv2.CV_16SC2)
        self.scale = float(scale)
        self.out_size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
        sx, sy = self.out_size[0] / w, self.out_size[1] / h
        P1 = calib.P1
        self.fx = float(P1[0, 0] * sx)
        self.fy = float(P1[1, 1] * sy)
        self.cx = float((P1[0, 2] + 0.5) * sx - 0.5)
        self.cy = float((P1[1, 2] + 0.5) * sy - 0.5)
        self.baseline = calib.baseline  # metres

    def rectify(self, left, right):
        w, h = self.size
        if left.shape[1] != w or left.shape[0] != h:
            raise RuntimeError(
                "Camera eye size is %dx%d but calibration was made at %dx%d. Re-run 02_calibrate.py "
                "(or switch back to the mode you calibrated)." % (left.shape[1], left.shape[0], w, h))
        l = cv2.remap(left, self.mapL[0], self.mapL[1], cv2.INTER_LINEAR)
        r = cv2.remap(right, self.mapR[0], self.mapR[1], cv2.INTER_LINEAR)
        if self.scale != 1.0:
            l = cv2.resize(l, self.out_size, interpolation=cv2.INTER_AREA)
            r = cv2.resize(r, self.out_size, interpolation=cv2.INTER_AREA)
        return l, r


# --------------------------------------------------------------------------- stereo depth
def remove_flying_pixels(depth, thr):
    """Zero out pixels whose 3x3 neighbourhood spans a depth range larger than thr * depth."""
    valid = depth > 0
    big = np.where(valid, depth, np.float32(1e6)).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    dmin = cv2.erode(big, k)
    dmax = cv2.dilate(depth, k)
    bad = valid & ((dmax - dmin) > thr * depth)
    out = depth.copy()
    out[bad] = 0
    return out


class StereoMatcher:
    """SGBM wrapper producing disparity (px) and metric depth (m) for rectified pairs."""

    def __init__(self, stereo_cfg, fx, baseline):
        self.cfg = dict(stereo_cfg)
        self.fx = float(fx)
        self.baseline = float(baseline)
        self.build()

    def build(self):
        c = self.cfg
        self.nd = max(16, int(round(c["num_disparities"] / 16.0)) * 16)
        bs = max(3, int(c["block_size"]) | 1)
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.nd, blockSize=bs,
            P1=8 * bs * bs, P2=32 * bs * bs,
            disp12MaxDiff=int(c["disp12_max_diff"]),
            uniquenessRatio=int(c["uniqueness"]),
            speckleWindowSize=int(c["speckle_window"]),
            speckleRange=int(c["speckle_range"]),
            preFilterCap=int(c["pre_filter_cap"]),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.clahe = cv2.createCLAHE(2.0, (8, 8)) if c.get("clahe") else None

    def closest_depth(self):
        """Nearest measurable distance with the current disparity range (m)."""
        return self.fx * self.baseline / self.nd

    def compute(self, left, right):
        if self.clahe is not None:
            left, right = self.clahe.apply(left), self.clahe.apply(right)
        disp = self.sgbm.compute(left, right).astype(np.float32) / 16.0
        valid = disp > 0.5
        depth = np.zeros_like(disp)
        depth[valid] = (self.fx * self.baseline) / disp[valid]
        c = self.cfg
        depth[(depth < c["min_depth"]) | (depth > c["max_depth"])] = 0
        if c["edge_filter"] > 0:
            depth = remove_flying_pixels(depth, c["edge_filter"])
        return disp, depth


def epipolar_error(left, right, max_features=800):
    """Median vertical mismatch (px) of ORB matches between a rectified pair.
    Good rectification: < ~0.5 px. Returns (median_dy, n_matches) or (None, 0)."""
    orb = cv2.ORB_create(max_features)
    kl, dl = orb.detectAndCompute(left, None)
    kr, dr = orb.detectAndCompute(right, None)
    if dl is None or dr is None or len(kl) < 20 or len(kr) < 20:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    dys = []
    for m in matches:
        pl, pr = kl[m.queryIdx].pt, kr[m.trainIdx].pt
        if pl[0] - pr[0] > 0 and abs(pl[1] - pr[1]) < 8:
            dys.append(abs(pl[1] - pr[1]))
    if len(dys) < 15:
        return None, len(dys)
    return float(np.median(dys)), len(dys)
