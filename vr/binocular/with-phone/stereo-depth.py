#!/usr/bin/env python3
"""
stereo_depth.py - compute a depth-information image from a stereo image pair.

Usage examples
--------------
Relative depth only (no calibration; output is a disparity map, near = warm):
    python stereo_depth.py left.png right.png -o depth.png

Metric depth (needs focal length in PIXELS and baseline in any unit, e.g. mm):
    python stereo_depth.py left.png right.png -o depth.png \
        --focal-px 700 --baseline 60 --max-depth 3000

With a calibration file (.npz with K1, D1, K2, D2, R, T) - the images are
rectified first and focal length / baseline are taken from the calibration:
    python stereo_depth.py left.png right.png -o depth.png --calib calib.npz

Requirements:
    pip install opencv-python numpy
    (optional, better results)  pip install opencv-contrib-python   # WLS filter

IMPORTANT: stereo matching assumes the two images are RECTIFIED, i.e. a point
in the left image appears on the same row in the right image. Raw images from
a cheap binocular module usually are not, so for real depth either pass --calib
or rectify the images beforehand. Left/right must also be the correct way round
(swap them if the result looks inverted or empty).
"""
import argparse
import sys

import cv2
import numpy as np


def load_gray_and_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rectify_pair(left, right, calib_path):
    """Rectify using a stereo calibration; returns (left, right, focal_px, baseline)."""
    c = np.load(calib_path)
    K1, D1, K2, D2, R, T = (c[k] for k in ("K1", "D1", "K2", "D2", "R", "T"))
    h, w = left.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)
    left_r = cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR)
    right_r = cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR)
    focal_px = float(P1[0, 0])
    baseline = float(abs(P2[0, 3] / P2[0, 0]))  # same unit as T
    return left_r, right_r, focal_px, baseline


def compute_disparity(gray_l, gray_r, num_disp, block, min_disp, use_wls):
    """Return disparity in pixels (float32); invalid pixels are NaN."""
    num_disp = max(16, (num_disp // 16) * 16)  # must be divisible by 16
    block = block if block % 2 == 1 else block + 1

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw_l = matcher.compute(gray_l, gray_r)  # int16, fixed point (x16)

    filtered = raw_l
    if use_wls and hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        raw_r = right_matcher.compute(gray_r, gray_l)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        filtered = wls.filter(raw_l, gray_l, disparity_map_right=raw_r)
    elif use_wls:
        print("Note: opencv-contrib not installed, skipping WLS smoothing "
              "(pip install opencv-contrib-python).", file=sys.stderr)

    disp = filtered.astype(np.float32) / 16.0
    disp[disp <= min_disp] = np.nan  # SGBM marks "no match" as min_disp - 1
    return disp


def colorize(values, vmin, vmax, near_is_high=True):
    """Map float array (NaN = invalid) to a BGR image, invalid pixels black."""
    valid = np.isfinite(values)
    norm = np.zeros(values.shape, np.float32)
    norm[valid] = np.clip((values[valid] - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    if not near_is_high:
        norm = 1.0 - norm
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def main():
    ap = argparse.ArgumentParser(description="Depth image from a stereo pair.")
    ap.add_argument("--left", help="left image", default="room-left.jpg")
    ap.add_argument("--right", help="right image", default="room-right.jpg")
    ap.add_argument("-o", "--output", default="depth.png", help="output colour depth image")
    ap.add_argument("--calib", help=".npz with K1, D1, K2, D2, R, T (rectifies images)")
    ap.add_argument("--focal-px", type=float, help="focal length in pixels (for metric depth)")
    ap.add_argument("--baseline", type=float, help="distance between cameras (e.g. mm)")
    ap.add_argument("--num-disp", type=int, default=128,
                    help="disparity search range in px, multiple of 16 (default 128; raise for close objects)")
    ap.add_argument("--min-disp", type=int, default=0)
    ap.add_argument("--block", type=int, default=5, help="odd matching window size (default 5)")
    ap.add_argument("--max-depth", type=float, help="clip display range to this depth (metric mode)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="downscale factor before matching, e.g. 0.5 for speed")
    ap.add_argument("--no-wls", action="store_true", help="disable WLS smoothing")
    ap.add_argument("--save-raw", action="store_true",
                    help="also save <output>.npy with float32 depth (or disparity)")
    args = ap.parse_args()

    img_l, gray_l = load_gray_and_color(args.left)
    img_r, gray_r = load_gray_and_color(args.right)

    if gray_l.shape != gray_r.shape:
        print("Images differ in size; resizing right image to match left.", file=sys.stderr)
        gray_r = cv2.resize(gray_r, (gray_l.shape[1], gray_l.shape[0]))
        img_r = cv2.resize(img_r, (gray_l.shape[1], gray_l.shape[0]))

    focal_px, baseline = args.focal_px, args.baseline
    if args.calib:
        img_l, img_r, focal_px, baseline = rectify_pair(img_l, img_r, args.calib)
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if args.scale != 1.0:
        gray_l = cv2.resize(gray_l, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        gray_r = cv2.resize(gray_r, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        if focal_px:
            focal_px *= args.scale

    # Light contrast normalisation helps when the two sensors differ slightly.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_l, gray_r = clahe.apply(gray_l), clahe.apply(gray_r)

    disp = compute_disparity(gray_l, gray_r, args.num_disp, args.block,
                             args.min_disp, use_wls=not args.no_wls)

    valid = np.isfinite(disp)
    if not valid.any():
        sys.exit("No valid disparities found. Check that the images are rectified, "
                 "left/right aren't swapped, and try a larger --num-disp.")

    if focal_px and baseline:
        depth = np.full_like(disp, np.nan)
        depth[valid] = focal_px * baseline / disp[valid]
        lo = float(np.nanpercentile(depth, 2))
        hi = args.max_depth or float(np.nanpercentile(depth, 98))
        out = colorize(np.clip(depth, None, hi), lo, hi, near_is_high=False)
        raw = depth
        print(f"Metric depth (baseline units). Displayed range: {lo:.1f} .. {hi:.1f}")
    else:
        lo = float(np.nanpercentile(disp, 2))
        hi = float(np.nanpercentile(disp, 98))
        out = colorize(disp, lo, hi, near_is_high=True)
        raw = disp
        print("No focal length/baseline given: output is RELATIVE depth (disparity). "
              "Warm = near, cool = far. Pass --focal-px and --baseline for metric depth.")

    if args.scale != 1.0:  # bring result back to original resolution
        out = cv2.resize(out, (img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(args.output, out)
    print(f"Saved {args.output}")
    if args.save_raw:
        np.save(args.output.rsplit(".", 1)[0] + ".npy", raw.astype(np.float32))
        print("Saved raw array (.npy)")


if __name__ == "__main__":
    main()