#!/usr/bin/env python3
"""
stereo_depth.py -- fast, CPU-only, confidence-aware depth + point cloud from a stereo pair.
Takes RAW camera frames (it undistorts + rectifies them itself from your calibration) or
already-rectified frames.

What it does differently from a plain StereoSGBM + reprojectImageTo3D call
    * recovers the left `num_disp` columns that OpenCV's SGBM silently leaves empty (pad trick)
    * built-in left/right consistency check + speckle filter (disp12MaxDiff, speckleWindowSize)
    * drops "flying pixels" along depth edges -- the main cause of streaky point clouds
    * drops pixels whose expected depth error is too large (range gate from sigma_d and baseline)
    * every output pixel is either trustworthy or explicitly invalid (0); holes are never guessed
    * optional half-resolution mode (~3x faster)

Use as a library (build once, call per frame):
    pipe = StereoPipeline("calib.npz")                 # raw=True: frames are rectified for you
    out = pipe.process(left_raw, right_raw)            # dict: disp, conf, gray, rect_left, rect_right, ms
    pts, inten = pipe.depth.to_points(out["disp"], out["gray"], out["conf"])   # Nx3 metres (x right, y down, z fwd)
    # disp: float32 px (0 = none); conf: 0 none / 1 weak (low texture) / 2 solid
    # USB modules that send ONE wide frame:  left, right = split_side_by_side(frame)

Command line, one raw pair:
    python stereo_depth.py left_raw.png right_raw.png --raw --calib calib.npz --report -o out/
Command line, already-rectified pair (or no calibration yet):
    python stereo_depth.py left.png right.png --calib calib.npz -o out/
    python stereo_depth.py left.png right.png --focal 740 --baseline 0.06 -o out/
Live from a camera / video file (q quits):
    python stereo_depth.py --live 0 --sbs --width 1280 --height 480 --raw --calib calib.npz --show
    python stereo_depth.py --live 0 --live2 1 --raw --calib calib.npz --show      # two separate USB cameras

Calibration: .npz / .json / .yml with K1 D1 K2 D2 R T and (optionally) R1 R2 P1 P2 Q image_size, i.e. what
cv2.stereoCalibrate + cv2.stereoRectify give you. Already-rectified use only needs Q.
Install: pip install numpy opencv-python        (opencv-contrib is NOT needed)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


# ----------------------------------------------------------------------------- calibration
def build_Q(f, baseline, cx, cy, cx_right=None):
    """Q matrix in OpenCV's convention. baseline in metres (positive), f/cx/cy in pixels."""
    cxr = cx if cx_right is None else cx_right
    return np.array([[1, 0, 0, -cx],
                     [0, 1, 0, -cy],
                     [0, 0, 0, f],
                     [0, 0, 1.0 / baseline, (cx - cxr) / baseline]], np.float64)


def load_Q(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path)
        for k in ("Q", "q"):
            if k in z:
                return np.array(z[k], np.float64)
        raise KeyError(f"no 'Q' in {path}; keys: {list(z.keys())}")
    if ext == ".json":
        return np.array(json.load(open(path))["Q"], np.float64)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    q = fs.getNode("Q").mat()
    fs.release()
    if q is None:
        raise KeyError(f"no node 'Q' in {path}")
    return q.astype(np.float64)


# ----------------------------------------------------------------------------- rectification
_ALIASES = {
    "K1": ("K1", "M1", "mtx1", "cameraMatrix1", "K_left"), "D1": ("D1", "dist1", "distCoeffs1", "D_left"),
    "K2": ("K2", "M2", "mtx2", "cameraMatrix2", "K_right"), "D2": ("D2", "dist2", "distCoeffs2", "D_right"),
    "R": ("R",), "T": ("T",), "R1": ("R1",), "R2": ("R2",), "P1": ("P1",), "P2": ("P2",), "Q": ("Q", "q"),
    "image_size": ("image_size", "imageSize", "size"),
}


def load_calibration(path):
    """Read a stereo calibration (.npz, .json, .yml/.xml) into a dict of float arrays.
    Understands the usual cv2.stereoCalibrate/stereoRectify names: K1 D1 K2 D2 R T R1 R2 P1 P2 Q image_size."""
    ext = os.path.splitext(path)[1].lower()
    raw = {}
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        raw = {k: z[k] for k in z.files}
    elif ext == ".json":
        raw = {k: np.array(v) for k, v in json.load(open(path)).items() if isinstance(v, list)}
    else:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        for names in _ALIASES.values():
            for n in names:
                m = fs.getNode(n).mat()
                if m is not None:
                    raw[n] = m
        fs.release()
    out = {}
    for key, names in _ALIASES.items():
        for n in names:
            if n in raw:
                out[key] = np.array(raw[n], np.float64) if key != "image_size" else np.array(raw[n]).astype(int).ravel()
                break
    if "Q" not in out and all(k in out for k in ("P1", "P2")):
        P1, P2 = out["P1"], out["P2"]
        out["Q"] = build_Q(P1[0, 0], -P2[0, 3] / P2[0, 0], P1[0, 2], P1[1, 2], P2[0, 2])
    if not out:
        raise KeyError(f"no calibration arrays found in {path}; keys seen: {list(raw)}")
    return out


class Rectifier:
    """Undistort + rectify RAW left/right frames. Maps are built once; per frame this is two
    cv2.remap calls (about 1 ms each at 640x480). Uses the calibration's own R1/R2/P1/P2 when
    present, otherwise runs cv2.stereoRectify (alpha=0 -> no black borders)."""

    def __init__(self, calib, alpha=0.0, interpolation=cv2.INTER_LINEAR):
        c = load_calibration(calib) if isinstance(calib, (str, os.PathLike)) else calib
        need = ("K1", "D1", "K2", "D2")
        if any(k not in c for k in need):
            raise KeyError("calibration needs K1, D1, K2, D2 (and R, T or R1, R2, P1, P2) to rectify raw frames")
        self.size = tuple(int(v) for v in c["image_size"])            # (width, height)
        if all(k in c for k in ("R1", "R2", "P1", "P2")):
            R1, R2, P1, P2 = c["R1"], c["R2"], c["P1"], c["P2"]
            Q = c.get("Q", build_Q(P1[0, 0], -P2[0, 3] / P2[0, 0], P1[0, 2], P1[1, 2], P2[0, 2]))
        else:
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(c["K1"], c["D1"], c["K2"], c["D2"], self.size,
                                                        c["R"], c["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        self.Q = np.array(Q, np.float64)
        self.interp = interpolation
        self._mapL = cv2.initUndistortRectifyMap(c["K1"], c["D1"], R1, P1[:, :3], self.size, cv2.CV_16SC2)
        self._mapR = cv2.initUndistortRectifyMap(c["K2"], c["D2"], R2, P2[:, :3], self.size, cv2.CV_16SC2)

    def __call__(self, left, right):
        for im in (left, right):
            if (im.shape[1], im.shape[0]) != self.size:
                raise ValueError(f"frame is {im.shape[1]}x{im.shape[0]} but the calibration is for "
                                 f"{self.size[0]}x{self.size[1]}; capture at the calibrated size")
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if right.ndim == 3:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return (cv2.remap(left, *self._mapL, self.interp), cv2.remap(right, *self._mapR, self.interp))


def split_side_by_side(frame, swap=False):
    """Many USB stereo modules deliver ONE wide frame (left | right). Split it into two images."""
    w = frame.shape[1] // 2
    l, r = frame[:, :w], frame[:, w:2 * w]
    return (r, l) if swap else (l, r)


class StereoPipeline:
    """RAW (or already-rectified) frames in -> depth out. Build once, call process() per frame."""

    def __init__(self, calib, raw=True, swap=False, **depth_kwargs):
        c = load_calibration(calib) if isinstance(calib, (str, os.PathLike)) else calib
        self.rect = Rectifier(c) if raw else None
        Q = self.rect.Q if self.rect is not None else c["Q"]
        self.depth = StereoDepth(Q, **depth_kwargs)
        self.swap = swap                         # True: the 'left' input is really the right camera

    def _rectified(self, a, b):
        return self.rect(a, b) if self.rect is not None else (a, b)

    def detect_order(self, a, b, min_matches=15):
        """Decide whether (a, b) is (left, right) or swapped by testing both orders, and set self.swap.
        Returns a dict with status 'ok', 'swapped' (self.swap flipped for you) or 'unclear'
        (neither order is stereo-consistent: wrong calibration, or too little texture in view)."""
        s_ab = pair_score(*self._rectified(a, b))
        s_ba = pair_score(*self._rectified(b, a))
        if max(s_ab, s_ba) < min_matches:
            status = "unclear"
        elif s_ba > 2 * s_ab + 5:
            status, self.swap = "swapped", True
        else:
            status, self.swap = "ok", False
        return {"status": status, "matches_as_given": s_ab, "matches_if_swapped": s_ba}

    def process(self, left, right):
        t0 = time.perf_counter()
        if self.swap:
            left, right = right, left
        if self.rect is not None:
            left, right = self.rect(left, right)
        t1 = time.perf_counter()
        disp, conf, gray = self.depth.disparity(left, right)
        t2 = time.perf_counter()
        return {"disp": disp, "conf": conf, "gray": gray, "rect_left": left, "rect_right": right,
                "ms": {"rectify": 1000 * (t1 - t0), "disparity": 1000 * (t2 - t1), "total": 1000 * (t2 - t0)}}


# ----------------------------------------------------------------------------- core class
class StereoDepth:
    def __init__(self, Q, num_disp=80, block=5, scale=1.0, uniqueness=10, speckle=120,
                 sigma_d=0.35, max_rel_err=0.10, flying_px=1.5, flying_rel=0.05,
                 tex_min=3.0, pad_left=True, prefilter=None, threads=0):
        """
        Q            4x4 reprojection matrix for the FULL-resolution images
        num_disp     disparity search range at full resolution (px). Nearest measurable point is
                     Z_min = f*B/(num_disp-1); make it just big enough for your closest objects.
        scale        0.5 = run at half resolution (about 3x faster, a little less detail)
        sigma_d      expected matching noise in disparity pixels at the working resolution
        max_rel_err  drop pixels whose depth error would exceed this fraction of their depth
                     (this sets the maximum range: d_min = sigma_d / max_rel_err)
        flying_px / flying_rel  3x3 disparity jump (px, and fraction of d) above which a pixel is
                     treated as a depth-edge artefact and removed
        tex_min      pixels whose neighbourhood has less horizontal image gradient than this
                     (saturated windows, black cabinets, blank walls) are marked confidence 1
                     ("weak"): the matcher produced a value, but only by smoothing across the gap
        prefilter   optional callable(gray)->gray applied to both images (e.g. local contrast
                     normalisation). It did NOT help on the test pair, so off by default.
        """
        if threads:
            cv2.setNumThreads(threads)
        self.s = float(scale)
        Q = np.array(Q, np.float64).copy()
        Q[:, 3] *= self.s                       # the pixel-unit column scales with resolution
        self.Q = Q
        self.fB = abs(Q[2, 3] / Q[3, 2])        # f*B at the working resolution (px * m)
        self.nd = int(np.ceil(num_disp * self.s / 16.0) * 16)
        self.d_min = max(1.0, sigma_d / max_rel_err)
        self.flying_px, self.flying_rel = flying_px, flying_rel
        self.tex_min, self.block = tex_min, block
        self.pad_left, self.prefilter = pad_left, prefilter
        self._xs = None
        bs = block
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.nd, blockSize=bs,
            P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1, preFilterCap=31,
            uniquenessRatio=uniqueness, speckleWindowSize=speckle, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self._k3 = np.ones((3, 3), np.uint8)

    # -- disparity ------------------------------------------------------------------------
    def disparity(self, left, right):
        """left/right: rectified 8-bit images (gray or BGR). Returns (disp, conf, gray_left) at
        the working resolution.
            disp  float32 pixels, 0 = no measurement
            conf  uint8: 0 none, 1 weak (low-texture, matcher interpolated), 2 solid
        """
        L = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
        R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
        if self.s != 1.0:
            L = cv2.resize(L, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
            R = cv2.resize(R, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)
        gray = L
        if self.prefilter is not None:
            L, R = self.prefilter(L), self.prefilter(R)
        nd = self.nd
        if self.pad_left:       # OpenCV leaves the first nd columns empty; pad so they are real
            Lp = cv2.copyMakeBorder(L, 0, 0, nd, 0, cv2.BORDER_REPLICATE)
            Rp = cv2.copyMakeBorder(R, 0, 0, nd, 0, cv2.BORDER_REPLICATE)
            d16 = self.sgbm.compute(Lp, Rp)[:, nd:]
        else:
            d16 = self.sgbm.compute(L, R)
        d = d16.astype(np.float32) * (1.0 / 16.0)
        d[d < 1.0] = 0
        return (*self._clean(d, gray), gray)

    def _clean(self, d, gray):
        H, W = d.shape
        if self._xs is None or self._xs.shape[1] != W:
            self._xs = np.arange(W, dtype=np.float32)[None, :]
        d[d < self.d_min] = 0                                        # too far to trust
        d[(self._xs - d) < self.block // 2] = 0                      # match would fall outside the right image
        v = d > 0
        hi = cv2.dilate(np.where(v, d, 0).astype(np.float32), self._k3)
        lo = cv2.erode(np.where(v, d, 1e4).astype(np.float32), self._k3)
        flying = v & ((hi - lo) > np.maximum(self.flying_px, self.flying_rel * d))
        d[flying] = 0
        gx = cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 1, 0, ksize=3)
        tex = cv2.boxFilter(np.abs(gx), -1, (2 * self.block + 1, 2 * self.block + 1))
        conf = np.where(d > 0, np.where(tex >= self.tex_min, 2, 1), 0).astype(np.uint8)
        return d, conf

    # -- geometry -------------------------------------------------------------------------
    def to_points(self, d, gray=None, conf=None, min_conf=2):
        """Valid pixels -> (N,3) float32 metres in the left-camera frame (x right, y down,
        z forward), plus per-point intensity if `gray` is given. With `conf`, only pixels with
        confidence >= min_conf are used (2 = solid only, 1 = include weak)."""
        sel = (d > 0) if conf is None else (conf >= min_conf)
        vs, us = np.nonzero(sel)
        dd = d[vs, us].astype(np.float64)
        H = self.Q @ np.stack([us, vs, dd, np.ones_like(dd)]).astype(np.float64)
        ok = H[3] > 1e-9
        P = (H[:3, ok] / H[3, ok]).T.astype(np.float32)
        ok2 = P[:, 2] > 0
        P = P[ok2]
        if gray is None:
            return P, None
        return P, gray[vs, us][ok][ok2]

    def sigma_z(self, z, sigma_d=0.35):
        """1-sigma depth error (m) at depth z (m): z^2 / (f*B) * sigma_d."""
        return z * z / self.fB * sigma_d


# ----------------------------------------------------------------------------- helpers
def voxel_downsample(pts, vox, vals=None):
    """Average points inside each vox-sized cell (fixes density bias near the camera)."""
    k = np.floor(pts / vox).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    out = np.stack([np.bincount(inv, pts[:, i]) / cnt for i in range(3)], 1).astype(np.float32)
    if vals is None:
        return out, None
    return out, (np.bincount(inv, vals.astype(np.float64)) / cnt).astype(np.uint8)


def write_ply(path, pts, gray=None):
    n = len(pts)
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    a = np.empty(n, dt)
    a["x"], a["y"], a["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    g = np.full(n, 200, np.uint8) if gray is None else gray
    a["r"] = a["g"] = a["b"] = g
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        a.tofile(f)


def render_ortho(pts, horiz, vert, hr, vr, size=(720, 540), color_by=2, flip_v=False):
    """Orthographic scatter view. horiz/vert = point columns; hr/vr = (min,max) metres."""
    w, h = size
    img = np.full((h, w, 3), 24, np.uint8)
    u = ((pts[:, horiz] - hr[0]) / (hr[1] - hr[0]) * (w - 1)).astype(int)
    v = ((pts[:, vert] - vr[0]) / (vr[1] - vr[0]) * (h - 1)).astype(int)
    if not flip_v:
        v = (h - 1) - v
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = pts[m, color_by]
    col = cv2.applyColorMap(np.clip((z - z.min()) / max(np.ptp(z), 1e-6) * 255, 0, 255).astype(np.uint8)[:, None],
                            cv2.COLORMAP_TURBO)[:, 0]
    order = np.argsort(-z)                      # far first, near on top
    img[v[m][order], u[m][order]] = col[order]
    return img


def render_perspective(pts, gray, yaw_deg=0.0, pitch_deg=0.0, size=(720, 540), f=None, dot=2):
    """Pinhole render of the cloud after rotating it about its own median point. Colours = image
    intensity. yaw_deg=0 reproduces the camera's own view; try +-25 to judge the 3D structure."""
    w, h = size
    c = np.median(pts, axis=0)
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    p = (pts - c) @ (Rx @ Ry).T + c
    f = f or 0.9 * w
    ok = p[:, 2] > 0.05
    p, g = p[ok], gray[ok]
    u = (f * p[:, 0] / p[:, 2] + w / 2).astype(int)
    v = (f * p[:, 1] / p[:, 2] + h / 2).astype(int)
    order = np.argsort(-p[:, 2])
    img = np.full((h, w, 3), 24, np.uint8)
    for du in range(dot):
        for dv in range(dot):
            uu, vv = u[order] + du, v[order] + dv
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[m], uu[m]] = g[order][m][:, None]
    return img


def rectification_report(left, right, max_feat=4000):
    """Quick health check on a rectified pair: vertical misalignment and disparity range."""
    L = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    orb = cv2.ORB_create(max_feat, fastThreshold=5)
    kl, dl = orb.detectAndCompute(L, None)
    kr, dr = orb.detectAndCompute(R, None)
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr), key=lambda x: x.distance)[:800]
    pl = np.array([kl[x.queryIdx].pt for x in m]); pr = np.array([kr[x.trainIdx].pt for x in m])
    dy, dx = pl[:, 1] - pr[:, 1], pl[:, 0] - pr[:, 0]
    ok = (np.abs(dy) < 8) & (dx > 0)
    if ok.sum() < 10:
        return {"matches": int(ok.sum()),
                "verdict": "NO USABLE MATCHES: left/right are probably swapped, the calibration does not belong "
                           "to these cameras, or the scene has too little texture"}, (np.zeros((0, 2)), np.zeros((0, 2)))
    return {"matches": int(ok.sum()),
            "dy_median_px": float(np.median(dy[ok])), "dy_std_px": float(dy[ok].std()),
            "dy_p95_abs_px": float(np.percentile(np.abs(dy[ok]), 95)),
            "disparity_px_p5_p50_p95_max": [float(v) for v in (*np.percentile(dx[ok], [5, 50, 95]), dx[ok].max())],
            "verdict": "rectification looks good" if abs(np.median(dy[ok])) < 0.7 and dy[ok].std() < 1.2
                       else "vertical misalignment: recalibrate / check rectification maps"}, (pl[ok], pr[ok])


def pair_score(left, right, max_feat=2000):
    """How stereo-consistent is this (left, right) pair? Counts feature matches that sit on the same row
    (|dy|<3 px) with a positive disparity. A correctly ordered, correctly rectified pair scores high;
    a swapped or mis-calibrated pair scores about zero."""
    L = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    R = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    orb = cv2.ORB_create(max_feat, fastThreshold=5)
    kl, dl = orb.detectAndCompute(L, None); kr, dr = orb.detectAndCompute(R, None)
    if dl is None or dr is None or len(kl) < 10 or len(kr) < 10:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    m = sorted(m, key=lambda x: x.distance)[:500]
    pl = np.array([kl[x.queryIdx].pt for x in m]); pr = np.array([kr[x.trainIdx].pt for x in m])
    dy, dx = pl[:, 1] - pr[:, 1], pl[:, 0] - pr[:, 0]
    return int(((np.abs(dy) < 3) & (dx > 1) & (dx < 250)).sum())


def sparse_check(sd, d, pl, pr):
    """Compare dense disparity with independent sparse ORB matches (full-res pixel units)."""
    s = sd.s
    xs = np.clip(np.round(pl[:, 0] * s).astype(int), 0, d.shape[1] - 1)
    ys = np.clip(np.round(pl[:, 1] * s).astype(int), 0, d.shape[0] - 1)
    dv = d[ys, xs] / s
    ok = dv > 0
    err = np.abs(dv[ok] - (pl[ok, 0] - pr[ok, 0]))
    return {"keypoints_with_depth": int(ok.sum()), "of": int(len(pl)),
            "median_abs_err_px": float(np.median(err)),
            "within_1px": float((err <= 1).mean()), "within_2px": float((err <= 2).mean())}


# ----------------------------------------------------------------------------- live loop
def run_live(a, pipe):
    """Capture -> (rectify) -> depth in a loop. Source: camera index or video file path."""
    src = int(a.live) if str(a.live).isdigit() else a.live
    caps = [cv2.VideoCapture(src)]
    if a.live2 is not None:
        caps.append(cv2.VideoCapture(int(a.live2) if str(a.live2).isdigit() else a.live2))
    for c in caps:
        if a.width:
            c.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        if a.height:
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        if not c.isOpened():
            raise SystemExit(f"cannot open camera/video: {a.live if c is caps[0] else a.live2}")
    n, t_start, win = 0, time.perf_counter(), {"capture": [], "rectify": [], "disparity": [], "total": []}
    last_print, last, order_checked = t_start, None, a.no_auto_order
    while a.frames == 0 or n < a.frames:
        tc = time.perf_counter()
        if len(caps) == 2:                       # grab both first so the two exposures are as close as possible
            if not (caps[0].grab() and caps[1].grab()):
                break
            _, l = caps[0].retrieve(); _, r = caps[1].retrieve()
        else:
            ok, frame = caps[0].read()
            if not ok:
                break
            l, r = split_side_by_side(frame) if a.sbs else (frame, frame)
        win["capture"].append(1000 * (time.perf_counter() - tc))
        if not order_checked and not a.no_auto_order:
            chk = pipe.detect_order(l, r)
            if chk["status"] != "unclear" or n >= 15:        # keep trying for a few frames if the view is blank
                order_checked = True
                msg = {"ok": "left/right order verified",
                       "swapped": "LEFT/RIGHT WERE SWAPPED: corrected automatically (use --swap to make it permanent)",
                       "unclear": "could not verify left/right order (neither order is stereo-consistent): check the "
                                  "calibration belongs to these cameras and that the scene has texture"}[chk["status"]]
                print(f"[order check] {msg}  {chk}", flush=True)
        out = pipe.process(l, r)
        for k in ("rectify", "disparity", "total"):
            win[k].append(out["ms"][k])
        last, n = out, n + 1
        if a.show:
            d, conf = out["disp"], out["conf"]
            vis = cv2.applyColorMap(np.clip(d * (255.0 / max(a.num_disp * pipe.depth.s, 1)), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            vis[conf == 1] = (vis[conf == 1] * 0.35).astype(np.uint8)
            vis[d == 0] = (20, 20, 20)
            gl = cv2.cvtColor(out["gray"], cv2.COLOR_GRAY2BGR)
            cv2.imshow("left (rectified) | depth: bright=solid dim=weak black=none", np.hstack([gl, vis]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        now = time.perf_counter()
        if now - last_print >= 1.0:
            m = {k: np.mean(v) for k, v in win.items() if v}
            print(f"{n / (now - t_start):5.1f} fps | capture {m['capture']:5.1f}  rectify {m['rectify']:4.1f}  "
                  f"disparity {m['disparity']:5.1f}  = processing {m['total']:5.1f} ms", flush=True)
            win = {k: [] for k in win}; last_print = now
    for c in caps:
        c.release()
    if a.show:
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t_start
    print(f"frames: {n}  average {n / max(elapsed, 1e-9):.1f} fps")
    return last, n


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", nargs="?"); ap.add_argument("right", nargs="?")
    ap.add_argument("-o", "--out", default="stereo_out")
    ap.add_argument("--calib", default="calibration.npz", help="calibration file (see header)")
    ap.add_argument("--raw", action="store_true", help="inputs are RAW (unrectified); rectify with --calib")
    ap.add_argument("--focal", type=float); ap.add_argument("--baseline", type=float, help="metres")
    ap.add_argument("--cx", type=float); ap.add_argument("--cy", type=float)
    ap.add_argument("--num-disp", type=int, default=80)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--scale", type=float, default=1.0, help="0.5 = half-resolution, ~3x faster")
    ap.add_argument("--max-rel-err", type=float, default=0.10)
    ap.add_argument("--tex-min", type=float, default=3.0, help="texture needed for 'solid' confidence")
    ap.add_argument("--voxel", type=float, default=0.02, help="voxel size (m) for the saved cloud; 0 = off")
    ap.add_argument("--bench", type=int, default=20, help="timing repetitions")
    ap.add_argument("--report", action="store_true", help="print rectification + sparse-agreement report")
    g = ap.add_argument_group("live mode")
    g.add_argument("--live", help="camera index or video file; runs a capture loop instead of one pair")
    g.add_argument("--live2", help="second camera index/file when left and right are separate devices")
    g.add_argument("--sbs", action="store_true", help="one frame holding left|right side by side")
    g.add_argument("--swap", action="store_true", help="swap left and right")
    ap.add_argument("--no-auto-order", action="store_true",
                    help="do not test for swapped left/right (default: test, and correct it with a warning)")
    g.add_argument("--width", type=int, help="requested capture width (whole frame for --sbs)")
    g.add_argument("--height", type=int)
    g.add_argument("--show", action="store_true", help="show a window")
    g.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = until q)")
    a = ap.parse_args()
    kw = dict(num_disp=a.num_disp, block=a.block, scale=a.scale, max_rel_err=a.max_rel_err, tex_min=a.tex_min,
              swap=a.swap)

    # ---- live loop
    if a.live is not None:
        if not a.calib:
            raise SystemExit("--live needs --calib")
        pipe = StereoPipeline(a.calib, raw=a.raw, **kw)
        run_live(a, pipe)
        return

    if a.left is None or a.right is None:
        raise SystemExit("give LEFT and RIGHT images, or use --live")
    Lim = cv2.imread(a.left, cv2.IMREAD_UNCHANGED); Rim = cv2.imread(a.right, cv2.IMREAD_UNCHANGED)
    h, w = Lim.shape[:2]
    assumed = False
    if a.calib:
        pipe = StereoPipeline(a.calib, raw=a.raw, **kw)
    else:
        if a.raw:
            raise SystemExit("--raw needs --calib (K1 D1 K2 D2 ...)")
        assumed = a.focal is None or a.baseline is None
        f = a.focal or 500.0; B = a.baseline or 0.06
        Q = build_Q(f, B, a.cx if a.cx is not None else w / 2, a.cy if a.cy is not None else h / 2)
        if assumed:
            print(f"[WARNING] no calibration given: ASSUMING f={f:.0f}px, baseline={B*100:.0f}cm. "
                  f"Shapes are right only up to scale; metric values are NOT reliable.", file=sys.stderr)
        pipe = StereoPipeline({"Q": Q}, raw=False, **kw)
    sd = pipe.depth
    order = None
    if not a.no_auto_order:
        order = pipe.detect_order(Lim, Rim)
        if order["status"] == "swapped":
            print(f"[order check] LEFT/RIGHT WERE SWAPPED: corrected automatically. {order}", file=sys.stderr)
        elif order["status"] == "unclear":
            print(f"[order check] could not verify left/right order. {order}", file=sys.stderr)
    res = pipe.process(Lim, Rim)                          # warm-up + result
    d, conf, gray, RL, RR = res["disp"], res["conf"], res["gray"], res["rect_left"], res["rect_right"]
    ms = {"rectify": [], "disparity": [], "total": []}
    for _ in range(max(a.bench, 1)):
        for k, v in pipe.process(Lim, Rim)["ms"].items():
            ms[k].append(v)
    pts, inten = sd.to_points(d, gray, conf, min_conf=2)           # solid points only
    pts_all, inten_all = sd.to_points(d, gray, conf, min_conf=1)   # solid + weak
    t = time.perf_counter(); sd.to_points(d, gray, conf, 2); t_pts = time.perf_counter() - t

    os.makedirs(a.out, exist_ok=True)
    if a.raw:
        cv2.imwrite(os.path.join(a.out, "rectified_left.png"), RL)
        cv2.imwrite(os.path.join(a.out, "rectified_right.png"), RR)
    vis = cv2.applyColorMap(np.clip(d * (255.0 / max(d.max(), 1)), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[conf == 1] = (vis[conf == 1] * 0.35).astype(np.uint8)       # weak = dimmed
    vis[d == 0] = (20, 20, 20)
    cv2.imwrite(os.path.join(a.out, "disparity.png"), vis)          # bright = solid, dim = weak, black = none
    cv2.imwrite(os.path.join(a.out, "confidence.png"), (conf * 127).astype(np.uint8))
    p, g = (voxel_downsample(pts, a.voxel, inten) if a.voxel > 0 else (pts, inten))
    write_ply(os.path.join(a.out, "points_solid.ply"), p, g)
    p2, g2 = (voxel_downsample(pts_all, a.voxel, inten_all) if a.voxel > 0 else (pts_all, inten_all))
    write_ply(os.path.join(a.out, "points_solid_plus_weak.ply"), p2, g2)
    zmax = float(np.percentile(pts[:, 2], 99)) if len(pts) else 5
    xr = (float(np.percentile(pts[:, 0], 1)), float(np.percentile(pts[:, 0], 99)))
    yr = (float(np.percentile(pts[:, 1], 1)), float(np.percentile(pts[:, 1], 99)))
    for yaw in (0, 25, -25):
        cv2.imwrite(os.path.join(a.out, f"view_yaw{yaw:+d}.png"), render_perspective(pts, inten, yaw))
    cv2.imwrite(os.path.join(a.out, "top_view.png"), render_ortho(pts, 0, 2, xr, (0, zmax)))
    cv2.imwrite(os.path.join(a.out, "side_view.png"), render_ortho(pts, 2, 1, (0, zmax), yr, flip_v=True))

    z = pts[:, 2]
    summary = {
        "calibration": "ASSUMED (not metric)" if assumed else "as given",
        "input": "raw (rectified here)" if a.raw else "already rectified",
        "left_right_order_check": order,
        "working_resolution": [d.shape[1], d.shape[0]], "num_disp": sd.nd, "scale": a.scale,
        "rectify_ms_median": round(float(np.median(ms["rectify"])), 1),
        "disparity_ms_median": round(float(np.median(ms["disparity"])), 1),
        "disparity_ms_p95": round(float(np.percentile(ms["disparity"], 95)), 1),
        "total_ms_median": round(float(np.median(ms["total"])), 1),
        "to_points_ms": round(1000 * t_pts, 1),
        "solid_fraction": round(float((conf == 2).mean()), 3),
        "weak_fraction": round(float((conf == 1).mean()), 3),
        "points_solid": int(len(pts)), "points_solid_plus_weak": int(len(pts_all)),
        "z_range_m_p1_p50_p99": [round(float(v), 2) for v in np.percentile(z, [1, 50, 99])],
        "nearest_measurable_m": round(float(sd.fB / (sd.nd - 1)), 2),
        "max_range_m_at_rel_err": round(float(sd.fB / sd.d_min), 2),
        "sigma_z_cm_at_1m_2m_3m": [round(100 * sd.sigma_z(x), 1) for x in (1, 2, 3)],
    }
    if a.report:
        if a.raw:
            a0, b0 = (Rim, Lim) if pipe.swap else (Lim, Rim)
            summary["raw_pair_before_rectification"] = {k: v for k, v in rectification_report(a0, b0)[0].items()
                                                         if k.startswith("dy") or k == "matches"}
        rep, (pl, pr) = rectification_report(RL, RR)
        summary["rectification"] = rep
        summary["agreement_with_sparse_orb_matches"] = sparse_check(sd, d, pl, pr)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()