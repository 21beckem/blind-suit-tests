"""Capture and audit one stereo frame without involving tracking or fusion.

Typical use:

    python 05_single_frame_test.py
    python 05_single_frame_test.py --known-distance 1.20
    python 05_single_frame_test.py --known-distance 1.20 --roi 220 150 100 80 --no-view
    python 05_single_frame_test.py --left left.png --right right.png --no-view

The tool saves raw and rectified images, disparity/depth arrays, left-right
consistency, point clouds, a summary image, and report.json under
validation_runs/.  In the interactive view, click a textured target to sample
its depth; pass --known-distance to include absolute and percentage error.

Positive --vertical-shift moves the rectified RIGHT image downward.  This is a
diagnostic for a nearly constant epipolar offset, not a replacement for a good
calibration.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from common import (CALIB_PATH, Calibration, Rectifier, StereoCamera,
                    StereoMatcher, load_config)

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "validation_runs"
WIN = "05_single_frame_test"


def capture_pair(camera_cfg, settle_frames):
    cam = StereoCamera(camera_cfg).start()
    item, last_id = None, -1
    try:
        for _ in range(max(1, settle_frames)):
            got = cam.read(last_id, 1.5)
            if got is None:
                raise RuntimeError("No frame received from the camera.")
            item, last_id = got, got[3]
    finally:
        cam.stop()
    left, right, captured_at, frame_id = item
    return left.copy(), right.copy(), {
        "kind": "live_camera",
        "captured_unix_s": float(captured_at),
        "frame_id": int(frame_id),
    }


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Could not read image: %s" % path)
    return img


def load_pair(left_path, right_path):
    left, right = load_gray(left_path), load_gray(right_path)
    if left.shape != right.shape:
        raise RuntimeError("Left/right shapes differ: %s vs %s" % (left.shape, right.shape))
    return left, right, {
        "kind": "saved_images",
        "left": str(Path(left_path).resolve()),
        "right": str(Path(right_path).resolve()),
    }


def shift_vertical(img, pixels):
    if abs(pixels) < 1e-9:
        return img
    h, w = img.shape
    M = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, float(pixels)]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def epipolar_metrics(left, right, max_disparity):
    """Robust signed vertical mismatch from descriptor matches.

    The signed convention is y_left - y_right.  A positive value means the
    rectified right image would need to move downward by approximately that
    many pixels to align the pair.
    """
    try:
        detector = cv2.SIFT_create(nfeatures=5000)
        norm, ratio = cv2.NORM_L2, 0.72
    except AttributeError:
        detector = cv2.ORB_create(4000)
        norm, ratio = cv2.NORM_HAMMING, 0.78
    kl, dl = detector.detectAndCompute(left, None)
    kr, dr = detector.detectAndCompute(right, None)
    if dl is None or dr is None or len(kl) < 20 or len(kr) < 20:
        return {"matches": 0, "status": "insufficient_features"}
    pairs = cv2.BFMatcher(norm).knnMatch(dl, dr, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if not good:
        return {"matches": 0, "status": "insufficient_features"}
    pl = np.float32([kl[m.queryIdx].pt for m in good])
    pr = np.float32([kr[m.trainIdx].pt for m in good])
    dx, dy = pl[:, 0] - pr[:, 0], pl[:, 1] - pr[:, 1]
    keep = (dx > 0.5) & (dx < max_disparity + 16) & (np.abs(dy) < 15.0)
    pl, dx, dy = pl[keep], dx[keep], dy[keep]
    if len(dy) < 15:
        return {"matches": int(len(dy)), "status": "insufficient_stereo_matches"}

    # Remove descriptor outliers around a simple affine vertical-error model.
    X = np.column_stack([np.ones(len(pl)), pl[:, 0] - left.shape[1] / 2.0,
                         pl[:, 1] - left.shape[0] / 2.0])
    mask = np.ones(len(dy), dtype=bool)
    coef = np.zeros(3)
    residual = dy.copy()
    for _ in range(5):
        coef = np.linalg.lstsq(X[mask], dy[mask], rcond=None)[0]
        residual = dy - X @ coef
        centered = residual[mask] - np.median(residual[mask])
        sigma = 1.4826 * np.median(np.abs(centered))
        mask = np.abs(residual - np.median(residual[mask])) < max(1.0, 2.5 * sigma)
    vals, abs_vals = dy[mask], np.abs(dy[mask])
    med_abs = float(np.median(abs_vals))
    quality = "good" if med_abs <= 0.5 else "marginal" if med_abs <= 1.0 else "poor"
    return {
        "status": quality,
        "matches": int(mask.sum()),
        "ratio_matches_before_geometry_filter": int(len(good)),
        "median_signed_dy_px": float(np.median(vals)),
        "median_abs_dy_px": med_abs,
        "p95_abs_dy_px": float(np.percentile(abs_vals, 95)),
        "suggested_right_image_shift_px": float(np.median(vals)),
        "vertical_model": {
            "constant_px": float(coef[0]),
            "per_x_px_per_pixel": float(coef[1]),
            "per_y_px_per_pixel": float(coef[2]),
            "residual_median_abs_px": float(np.median(np.abs(residual[mask]))),
            "residual_p95_abs_px": float(np.percentile(np.abs(residual[mask]), 95)),
        },
        "horizontal_disparity_px": {
            "p05": float(np.percentile(dx[mask], 5)),
            "median": float(np.median(dx[mask])),
            "p95": float(np.percentile(dx[mask], 95)),
        },
    }


def make_sgbm(stereo_cfg, min_disparity, num_disparities):
    bs = max(3, int(stereo_cfg["block_size"]) | 1)
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity), numDisparities=int(num_disparities), blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,
        disp12MaxDiff=int(stereo_cfg["disp12_max_diff"]),
        uniquenessRatio=int(stereo_cfg["uniqueness"]),
        speckleWindowSize=int(stereo_cfg["speckle_window"]),
        speckleRange=int(stereo_cfg["speckle_range"]),
        preFilterCap=int(stereo_cfg["pre_filter_cap"]),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def left_right_consistency(left, right, disparity, stereo_cfg, nd):
    """Return a mask where independent left/right disparities agree."""
    if stereo_cfg.get("clahe"):
        clahe = cv2.createCLAHE(2.0, (8, 8))
        left, right = clahe.apply(left), clahe.apply(right)
    right_matcher = make_sgbm(stereo_cfg, -nd, nd)
    disp_r = right_matcher.compute(right, left).astype(np.float32) / 16.0
    h, w = disparity.shape
    yy, xx = np.indices((h, w))
    xr = np.rint(xx - disparity).astype(np.int32)
    inside = (disparity > 0.5) & (xr >= 0) & (xr < w)
    sampled = np.zeros_like(disparity)
    sampled[inside] = disp_r[yy[inside], xr[inside]]
    tolerance = max(1.0, float(stereo_cfg["disp12_max_diff"]))
    consistent = inside & (sampled < -0.5) & (np.abs(disparity + sampled) <= tolerance)
    return consistent, disp_r


def sample_region(depth, x, y, width, height, known_distance=None):
    h, w = depth.shape
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, x0 + max(1, int(width))), min(h, y0 + max(1, int(height)))
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    out = {
        "roi": [x0, y0, x1 - x0, y1 - y0],
        "pixels": int(patch.size),
        "valid_pixels": int(vals.size),
        "valid_pct": 100.0 * float(vals.size) / max(1, patch.size),
    }
    if vals.size:
        med = float(np.median(vals))
        out.update({
            "median_depth_m": med,
            "p10_depth_m": float(np.percentile(vals, 10)),
            "p90_depth_m": float(np.percentile(vals, 90)),
            "mad_m": float(np.median(np.abs(vals - med))),
        })
        if known_distance is not None:
            err = med - known_distance
            out.update({
                "known_distance_m": float(known_distance),
                "error_m": float(err),
                "absolute_error_m": float(abs(err)),
                "error_pct": 100.0 * float(err) / known_distance,
            })
    return out


def colorize_disparity(disparity, nd):
    v = np.clip(disparity * (255.0 / max(1, nd)), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
    out[disparity <= 0.5] = 0
    return out


def colorize_depth(depth, zmax):
    v = np.clip(depth / max(zmax, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(255 - v, cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return out


def add_epipolar_lines(img):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    step = max(20, img.shape[0] // 12)
    for y in range(step // 2, img.shape[0], step):
        cv2.line(out, (0, y), (img.shape[1], y), (0, 180, 0), 1)
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return out


def save_point_cloud(path, depth, gray, rect, mask=None, export_y_up=True):
    import open3d as o3d

    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    yy, xx = np.nonzero(valid)
    z = depth[yy, xx].astype(np.float64)
    xyz = np.column_stack([(xx - rect.cx) * z / rect.fx,
                           (yy - rect.cy) * z / rect.fy, z])
    if export_y_up:
        xyz *= np.array([1.0, -1.0, -1.0])
    colors = np.repeat((gray[yy, xx].astype(np.float64) / 255.0)[:, None], 3, axis=1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False)
    return int(len(xyz))


def depth_statistics(depth):
    valid = np.isfinite(depth) & (depth > 0)
    vals = depth[valid]
    out = {"valid_pixels": int(vals.size), "valid_pct": 100.0 * float(valid.mean())}
    if vals.size:
        out.update({
            "p05_m": float(np.percentile(vals, 5)),
            "p10_m": float(np.percentile(vals, 10)),
            "median_m": float(np.median(vals)),
            "p90_m": float(np.percentile(vals, 90)),
            "p95_m": float(np.percentile(vals, 95)),
        })
    return out


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n")


def draw_sample_markers(canvas, samples, w, h):
    for i, sample in enumerate(samples, 1):
        x, y, rw, rh = sample["roi"]
        for ox, oy in ((0, 0), (w, 0), (0, h), (w, h)):
            cv2.rectangle(canvas, (ox + x, oy + y), (ox + x + rw, oy + y + rh),
                          (255, 255, 255), 1)
        if "median_depth_m" in sample:
            text = "%d: %.3fm" % (i, sample["median_depth_m"])
            cv2.putText(canvas, text, (x + 4, min(2 * h - 8, h + y + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Validate one calibrated stereo frame and save a report bundle.")
    ap.add_argument("--left", help="saved left-eye image (requires --right)")
    ap.add_argument("--right", help="saved right-eye image (requires --left)")
    ap.add_argument("--calibration", default=str(CALIB_PATH), help="calibration .npz")
    ap.add_argument("--output", help="output run directory (default: validation_runs/<timestamp>)")
    ap.add_argument("--settle-frames", type=int, default=12, help="live frames to discard before capture")
    ap.add_argument("--known-distance", type=float, help="metres; applied to clicks and --roi")
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                    help="sample this rectified-image region")
    ap.add_argument("--point", nargs=2, type=int, action="append", metavar=("X", "Y"),
                    help="sample a 9x9 patch; may be repeated")
    ap.add_argument("--vertical-shift", type=float, default=0.0,
                    help="diagnostic: move rectified RIGHT image down by this many pixels")
    ap.add_argument("--no-view", action="store_true", help="save outputs without opening the click viewer")
    ap.add_argument("--display", type=float, default=1.0, help="interactive window scale")
    args = ap.parse_args()

    if bool(args.left) != bool(args.right):
        ap.error("--left and --right must be supplied together")
    if args.known_distance is not None and args.known_distance <= 0:
        ap.error("--known-distance must be positive")
    if args.display <= 0:
        ap.error("--display must be positive")

    cfg = load_config()
    calib = Calibration(args.calibration)
    rect = Rectifier(calib, cfg["stereo"]["scale"])
    matcher = StereoMatcher(cfg["stereo"], rect.fx, rect.baseline)
    if args.left:
        raw_l, raw_r, source = load_pair(args.left, args.right)
    else:
        raw_l, raw_r, source = capture_pair(cfg["camera"], args.settle_frames)
    rect_l, rect_r_original = rect.rectify(raw_l, raw_r)
    epi_before = epipolar_metrics(rect_l, rect_r_original, matcher.nd)
    rect_r = shift_vertical(rect_r_original, args.vertical_shift)
    epi_after = epipolar_metrics(rect_l, rect_r, matcher.nd)

    disparity, depth = matcher.compute(rect_l, rect_r)
    consistency, _disp_r = left_right_consistency(rect_l, rect_r, disparity, cfg["stereo"], matcher.nd)
    depth_valid = depth > 0
    consistent_valid = depth_valid & consistency
    lr_pct = 100.0 * float(consistent_valid.sum()) / max(1, int(depth_valid.sum()))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output) if args.output else RUNS_DIR / ("frame_" + stamp)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "raw_left.png"), raw_l)
    cv2.imwrite(str(out / "raw_right.png"), raw_r)
    cv2.imwrite(str(out / "rectified_left.png"), rect_l)
    cv2.imwrite(str(out / "rectified_right.png"), rect_r)
    np.save(str(out / "disparity_px.npy"), disparity.astype(np.float32))
    np.save(str(out / "depth_m.npy"), depth.astype(np.float32))
    np.save(str(out / "lr_consistent.npy"), consistency.astype(np.bool_))

    disp_color = colorize_disparity(disparity, matcher.nd)
    depth_color = colorize_depth(depth, cfg["stereo"]["max_depth"])
    conf_color = np.zeros((*consistency.shape, 3), np.uint8)
    conf_color[depth_valid] = (0, 0, 160)
    conf_color[consistent_valid] = (0, 200, 0)
    cv2.imwrite(str(out / "disparity.png"), disp_color)
    cv2.imwrite(str(out / "depth.png"), depth_color)
    cv2.imwrite(str(out / "lr_consistency.png"), conf_color)

    export_y_up = bool(cfg.get("scan", {}).get("export_y_up", True))
    n_cloud = save_point_cloud(out / "pointcloud.ply", depth, rect_l, rect, export_y_up=export_y_up)
    n_conf = save_point_cloud(out / "pointcloud_confident.ply", depth, rect_l, rect,
                              mask=consistency, export_y_up=export_y_up)

    h, w = rect_l.shape
    fov = math.degrees(2.0 * math.atan(w / (2.0 * rect.fx)))
    report = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "calibration": {
            "path": str(Path(args.calibration).resolve()),
            "image_size": list(calib.size),
            "stereo_rms_px": float(calib.rms),
            "rectified_focal_px": float(rect.fx),
            "rectified_hfov_deg": float(fov),
            "baseline_mm": 1000.0 * float(rect.baseline),
        },
        "processing": {
            "size": [w, h],
            "vertical_shift_applied_to_right_px": float(args.vertical_shift),
            "stereo_config": dict(cfg["stereo"]),
            "closest_measurable_depth_m": float(matcher.closest_depth()),
        },
        "epipolar_before_manual_shift": epi_before,
        "epipolar_used_for_depth": epi_after,
        "depth": depth_statistics(depth),
        "left_right_consistency": {
            "consistent_depth_pixels": int(consistent_valid.sum()),
            "consistent_pct_of_depth_valid": float(lr_pct),
        },
        "point_cloud": {
            "points": n_cloud,
            "left_right_consistent_points": n_conf,
            "coordinate_convention": "x-right, y-up, z-back" if export_y_up else "OpenCV x-right, y-down, z-forward",
        },
        "warnings": [],
        "samples": [],
    }

    if args.roi:
        report["samples"].append(sample_region(depth, *args.roi, args.known_distance))
    for point in args.point or []:
        report["samples"].append(sample_region(depth, point[0] - 4, point[1] - 4, 9, 9,
                                               args.known_distance))

    epi_abs = epi_after.get("median_abs_dy_px")
    if epi_abs is not None and epi_abs > 1.0:
        report["warnings"].append(
            "Rectified median vertical error is %.2f px; recalibrate before trusting metric depth." % epi_abs)
    elif epi_abs is not None and epi_abs > 0.5:
        report["warnings"].append(
            "Rectified median vertical error is marginal at %.2f px (target <= 0.5 px)." % epi_abs)
    if args.known_distance is not None:
        required_disp = rect.fx * rect.baseline / args.known_distance
        required_nd = max(16, int(math.ceil(required_disp / 16.0) * 16))
        report["known_distance_test"] = {
            "distance_m": float(args.known_distance),
            "expected_disparity_px": float(required_disp),
            "minimum_num_disparities": int(required_nd),
        }
        if args.known_distance < matcher.closest_depth():
            report["warnings"].append(
                "Known target %.3f m is closer than the current %.3f m measurable limit; "
                "use at least %d disparities before evaluating this target." % (
                    args.known_distance, matcher.closest_depth(), required_nd))
        if args.known_distance > cfg["stereo"]["max_depth"]:
            report["warnings"].append(
                "Known target %.3f m exceeds max_depth %.3f m and will be filtered out." % (
                    args.known_distance, cfg["stereo"]["max_depth"]))
    if lr_pct < 50.0:
        report["warnings"].append(
            "Only %.1f%% of depth-valid pixels pass the independent left/right check." % lr_pct)

    epi_used = epi_after.get("median_abs_dy_px")
    epi_text = "epi n/a" if epi_used is None else "epi %.2fpx" % epi_used
    valid_text = "valid %.1f%% | LR %.1f%%" % (report["depth"]["valid_pct"], lr_pct)
    lc = label(add_epipolar_lines(rect_l), "rectified left | " + valid_text)
    rc = label(add_epipolar_lines(rect_r), "rectified right | " + epi_text)
    dc = label(disp_color, "disparity | range %dpx" % matcher.nd)
    zc = label(depth_color, "depth | %.2f..%.2fm" % (cfg["stereo"]["min_depth"],
                                                       cfg["stereo"]["max_depth"]))
    canvas = np.vstack([np.hstack([lc, rc]), np.hstack([dc, zc])])
    draw_sample_markers(canvas, report["samples"], w, h)
    cv2.imwrite(str(out / "summary.png"), canvas)
    write_report(out / "report.json", report)

    print("Saved validation run: %s" % out.resolve())
    print("Calibration: RMS %.3f px | baseline %.2f mm | focal %.1f px" % (
        calib.rms, rect.baseline * 1000.0, rect.fx))
    if "median_abs_dy_px" in epi_before:
        print("Epipolar before shift: signed %.3f px | abs median %.3f px | p95 %.3f px | %s" % (
            epi_before["median_signed_dy_px"], epi_before["median_abs_dy_px"],
            epi_before["p95_abs_dy_px"], epi_before["status"]))
    if args.vertical_shift and "median_abs_dy_px" in epi_after:
        print("Epipolar after shift:  signed %.3f px | abs median %.3f px | p95 %.3f px | %s" % (
            epi_after["median_signed_dy_px"], epi_after["median_abs_dy_px"],
            epi_after["p95_abs_dy_px"], epi_after["status"]))
    print("Depth valid: %.1f%% | LR-consistent: %.1f%% of valid | closest %.3f m" % (
        report["depth"]["valid_pct"], lr_pct, matcher.closest_depth()))
    for warning in report["warnings"]:
        print("WARNING: " + warning)
    for i, sample in enumerate(report["samples"], 1):
        if "median_depth_m" not in sample:
            print("Sample %d: no valid depth (%s)" % (i, sample["roi"]))
        elif "error_m" in sample:
            print("Sample %d: %.3f m vs %.3f m | error %+.3f m (%+.1f%%)" % (
                i, sample["median_depth_m"], sample["known_distance_m"],
                sample["error_m"], sample["error_pct"]))
        else:
            print("Sample %d: %.3f m | valid %.1f%%" % (
                i, sample["median_depth_m"], sample["valid_pct"]))

    if args.no_view:
        return 0

    display = canvas
    click_samples = report["samples"]

    def refresh():
        nonlocal display
        display = canvas.copy()
        draw_sample_markers(display, click_samples, w, h)
        shown = display
        if args.display != 1.0:
            shown = cv2.resize(shown, None, fx=args.display, fy=args.display,
                               interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN, shown)

    def on_mouse(event, x, y, _flags, _data):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        px = int((x / args.display) % w)
        py = int((y / args.display) % h)
        sample = sample_region(depth, px - 4, py - 4, 9, 9, args.known_distance)
        click_samples.append(sample)
        if "median_depth_m" in sample:
            msg = "%.3f m" % sample["median_depth_m"]
            if "error_m" in sample:
                msg += " | error %+.3f m (%+.1f%%)" % (sample["error_m"], sample["error_pct"])
        else:
            msg = "no valid depth"
        print("Click (%d,%d): %s" % (px, py, msg))
        report["samples"] = click_samples
        write_report(out / "report.json", report)
        refresh()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    print("Viewer: click to sample | C clear samples | Q/Esc quit")
    refresh()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("c"), ord("C")):
            click_samples.clear()
            report["samples"] = click_samples
            write_report(out / "report.json", report)
            refresh()
    cv2.imwrite(str(out / "summary_annotated.png"), display)
    write_report(out / "report.json", report)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
