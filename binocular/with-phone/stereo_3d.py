#!/usr/bin/env python3
"""
stereo_3d.py - interactive 3D point cloud from two photos taken from slightly
different positions (e.g. a phone moved ~5 inches sideways, or a stereo module).

    python stereo_3d.py left.jpg right.jpg -c stereo_config.json

What it does
  1. Matches features between the photos and estimates the relative camera
     rotation/direction (essential matrix) - so the photos do NOT need to be
     perfectly aligned or pre-rectified.
  2. Uses the baseline from the config to give the scene real-world scale (mm).
  3. Rectifies the pair, computes a dense disparity map (SGBM), and
     reprojects it to 3D.
  4. Opens a window you can orbit / pan / zoom (Open3D; matplotlib fallback).

Viewer controls (Open3D)
  left-drag: orbit    ctrl+left-drag or middle-drag: pan    wheel: zoom
  + / - : bigger / smaller points      q or Esc: quit

Install:  pip install opencv-python numpy open3d
Optional: pip install opencv-contrib-python   (smoother disparity via WLS filter)
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

DEFAULTS = {
    "focal_length_px": 3000.0,            # focal length in pixels ...
    "focal_reference_long_side_px": 4000,  # ... measured on an image with this long side
    "baseline_mm": 127.0,                 # distance between the two camera positions
    "min_depth_mm": 400.0,                # nearest thing you expect (sets disparity search range)
    "max_depth_mm": 6000.0,               # points farther than this are dropped
    "max_image_side_px": 1200,            # photos are downscaled to this for speed
    "block_size": 5,
    "remove_outliers": True,
    "point_size": 2.0,
}


# --------------------------------------------------------------------------- setup
def load_config(path):
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        cfg.update({k: v for k, v in user.items() if not k.startswith("_")})
    elif path and path != "stereo_config.json":
        sys.exit(f"Config file not found: {path}")
    return cfg


def load_resized(path, max_side):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # honours EXIF rotation
    if img is None:
        sys.exit(f"Could not read image: {path}")
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def make_K(w, h, cfg):
    f = cfg["focal_length_px"] * max(w, h) / cfg["focal_reference_long_side_px"]
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)


# --------------------------------------------------------------------------- geometry
def match_features(gray1, gray2):
    sift = cv2.SIFT_create(nfeatures=8000)
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        sys.exit("Too few features found. Use a scene with more texture/detail.")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        sys.exit(f"Only {len(good)} feature matches - the photos overlap too little "
                 "or the scene has too little texture.")
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def estimate_pose(p1, p2, K):
    """Rotation R and unit translation t with x2 = R*x1 + t."""
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        sys.exit("Could not estimate camera motion between the photos.")
    E = E[:3]
    n, R, t, _ = cv2.recoverPose(E, p1, p2, K, mask=mask)
    return R, t.ravel(), int(n)


def compute_disparity(gl, gr, num_disp, block):
    block = block if block % 2 else block + 1
    # OpenCV's SGBM leaves the first `num_disp` columns empty; pad both images on
    # the left so the left part of the scene is not thrown away, and crop afterwards.
    pad = num_disp
    gl = cv2.copyMakeBorder(gl, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    gr = cv2.copyMakeBorder(gr, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    raw = matcher.compute(gl, gr)
    if hasattr(cv2, "ximgproc"):
        right = cv2.ximgproc.createRightMatcher(matcher)
        raw_r = right.compute(gr, gl)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        raw = wls.filter(raw, gl, disparity_map_right=raw_r)
    disp = raw.astype(np.float32)[:, pad:] / 16.0
    disp[disp <= 0] = np.nan
    return disp


def reconstruct(img_l, img_r, K, cfg, debug_dir=None):
    """Return (points_mm Nx3 in OpenCV camera coords, colors Nx3 RGB 0..1, info)."""
    h, w = img_l.shape[:2]
    gl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    p1, p2 = match_features(gl, gr)
    R, t, n_in = estimate_pose(p1, p2, K)
    swapped = False
    if t[0] > 0:  # second photo was taken from the LEFT of the first: swap them
        print("Photos look like they were given right-then-left; swapping them.")
        img_l, img_r, gl, gr, p1, p2 = img_r, img_l, gr, gl, p2, p1
        R, t, n_in = estimate_pose(p1, p2, K)
        swapped = True
    print(f"Feature matches: {len(p1)}, pose inliers: {n_in}, "
          f"sideways share of motion: {abs(t[0]):.2f}")
    if abs(t[0]) < 0.7:
        print("WARNING: the phone moved mostly forward/back/up/down rather than sideways; "
              "results will be poor. Move it horizontally between shots.", file=sys.stderr)

    B = float(cfg["baseline_mm"])
    T = (t * B).reshape(3, 1)
    D = np.zeros((5, 1))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(K, D, R1, P1, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K, D, R2, P2, (w, h), cv2.CV_32FC1)
    rect_l = cv2.remap(img_l, m1x, m1y, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, m2x, m2y, cv2.INTER_LINEAR)
    inside = cv2.remap(np.full((h, w), 255, np.uint8), m1x, m1y, cv2.INTER_NEAREST) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    f_rect = float(P1[0, 0])
    d_max = f_rect * B / float(cfg["min_depth_mm"])
    num_disp = int(min(640, max(16, math.ceil((d_max + 16) / 16) * 16)))
    print(f"Disparity search range: {num_disp} px (nearest depth {cfg['min_depth_mm']:.0f} mm)")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gl2 = clahe.apply(cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY))
    gr2 = clahe.apply(cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY))
    disp = compute_disparity(gl2, gr2, num_disp, int(cfg["block_size"]))

    valid = np.isfinite(disp) & inside
    d0 = np.where(valid, disp, 0).astype(np.float32)
    if cfg["remove_outliers"]:
        med = cv2.medianBlur(d0, 5)
        valid &= np.abs(d0 - med) <= 2.0

    xyz = cv2.reprojectImageTo3D(np.where(valid, d0, 0).astype(np.float32), Q)
    Z = xyz[..., 2]
    valid &= (d0 > 0) & np.isfinite(Z) & (Z > 0.5 * cfg["min_depth_mm"]) & (Z <= cfg["max_depth_mm"])

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        pair = np.hstack([rect_l, rect_r])
        for y in range(0, h, 40):
            cv2.line(pair, (0, y), (2 * w, y), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(debug_dir, "rectified_pair.png"), pair)
        dv = np.where(valid, d0, 0)
        dn = np.clip(dv / max(np.percentile(dv[valid], 98), 1) if valid.any() else dv, 0, 1)
        cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cm[~valid] = 0
        cv2.imwrite(os.path.join(debug_dir, "disparity.png"), cm)

    if not valid.any():
        sys.exit("No valid 3D points. Check the rectified preview (--debug-dir), the "
                 "baseline/focal settings, and that the scene has texture.")

    pts = xyz[valid].astype(np.float64)
    cols = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    ys, xs = np.nonzero(valid)  # same (row-major) order as xyz[valid]
    return pts, cols, {"swapped": swapped, "f_rect": f_rect,
                       "pix": np.stack([ys, xs], 1), "disp": d0[valid].astype(np.float32),
                       "shape": valid.shape}


# --------------------------------------------------------------------------- output
def to_view_coords(pts_mm):
    """OpenCV (x right, y down, z forward, mm) -> viewer (x right, y up, z toward viewer, m)."""
    return pts_mm * np.array([1.0, -1.0, -1.0]) / 1000.0


def save_ply(path, pts, cols):
    data = np.hstack([pts, (cols * 255).astype(np.uint8)])
    header = ("ply\nformat ascii 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header" % len(pts))
    np.savetxt(path, data, fmt="%.4f %.4f %.4f %d %d %d", header=header, comments="")
    print(f"Saved {path} ({len(pts)} points, metres)")


def show_open3d(o3d, pts, cols, point_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # = the camera

    vis = o3d.visualization.Visualizer()
    vis.create_window("Stereo 3D - drag to orbit, ctrl+drag to pan, wheel to zoom", 1280, 800)
    vis.add_geometry(pcd)
    vis.add_geometry(origin)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([0.08, 0.08, 0.10])

    ctr = vis.get_view_control()
    front = np.array([0.35, 0.2, 1.0])
    ctr.set_front((front / np.linalg.norm(front)).tolist())
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_lookat(np.median(pts, axis=0).tolist())
    ctr.set_zoom(0.45)
    vis.run()
    vis.destroy_window()


def show_matplotlib(pts, cols):
    import matplotlib.pyplot as plt
    n = len(pts)
    idx = np.random.default_rng(0).choice(n, size=min(n, 60000), replace=False)
    p, c = pts[idx], cols[idx]
    fig = plt.figure("Stereo 3D (matplotlib fallback)", figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(p[:, 0], -p[:, 2], p[:, 1], c=c, s=1)  # x right, depth away, y up
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    ax.set_zlabel("y (m)")
    ax.set_box_aspect((np.ptp(p[:, 0]), np.ptp(p[:, 2]), np.ptp(p[:, 1])))
    ax.view_init(elev=15, azim=-70)
    plt.show()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Interactive 3D view from a stereo/photo pair.")
    ap.add_argument("left", help="first photo (left position)")
    ap.add_argument("right", help="second photo (moved to the right)")
    ap.add_argument("-c", "--config", default="stereo_config.json")
    ap.add_argument("--save-ply", metavar="FILE", help="also save the point cloud as .ply")
    ap.add_argument("--no-view", action="store_true", help="don't open the 3D window")
    ap.add_argument("--debug-dir", help="save rectified pair + disparity previews here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    img_l = load_resized(args.left, cfg["max_image_side_px"])
    img_r = load_resized(args.right, cfg["max_image_side_px"])
    if img_r.shape[:2] != img_l.shape[:2]:
        img_r = cv2.resize(img_r, (img_l.shape[1], img_l.shape[0]))
    h, w = img_l.shape[:2]
    K = make_K(w, h, cfg)
    print(f"Working size {w}x{h}, focal {K[0, 0]:.0f}px, baseline {cfg['baseline_mm']} mm")

    pts_mm, cols, _ = reconstruct(img_l, img_r, K, cfg, args.debug_dir)
    z = pts_mm[:, 2] / 1000.0
    print(f"{len(pts_mm)} points; depth (m) 5%/50%/95%: "
          f"{np.percentile(z, 5):.2f} / {np.percentile(z, 50):.2f} / {np.percentile(z, 95):.2f}")

    pts = to_view_coords(pts_mm)
    if args.save_ply:
        save_ply(args.save_ply, pts, cols)
    if args.no_view:
        return
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed (pip install open3d); using matplotlib fallback.")
        show_matplotlib(pts, cols)
        return
    show_open3d(o3d, pts, cols, cfg["point_size"])


if __name__ == "__main__":
    main()