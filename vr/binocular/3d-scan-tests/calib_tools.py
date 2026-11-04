"""Stereo calibration math (no camera access, so it can be unit-tested)."""
import math

import cv2
import numpy as np


def board_object_points(cols, rows, square_m):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return objp


def find_corners(gray, cols, rows):
    """Returns (N,1,2) float32 corners or None. Uses the robust 'SB' detector."""
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    try:
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
    except cv2.error:
        return None
    if not ok or c is None:
        return None
    return c.reshape(-1, 1, 2).astype(np.float32)


def pose_features(corners, cols, rows, size):
    """Small descriptor of how the board sits in the image: position, size, tilt."""
    w, h = size
    pts = corners.reshape(rows, cols, 2)
    cx, cy = pts.reshape(-1, 2).mean(axis=0)
    top = np.linalg.norm(pts[0, -1] - pts[0, 0])
    bot = np.linalg.norm(pts[-1, -1] - pts[-1, 0])
    lft = np.linalg.norm(pts[-1, 0] - pts[0, 0])
    rgt = np.linalg.norm(pts[-1, -1] - pts[0, -1])
    area = math.sqrt(max((top + bot) / 2 * (lft + rgt) / 2, 1e-6))
    return np.array([cx / w, cy / h, area / w, math.log(max(top, 1e-6) / max(bot, 1e-6)),
                     math.log(max(lft, 1e-6) / max(rgt, 1e-6))])


def run_calibration(obj_pts, pts_l, pts_r, size, min_views=10, log=print):
    """Intrinsics per eye -> stereo extrinsics, dropping outlier views. size = (w, h).
    Returns dict with K1,D1,K2,D2,R,T,rms,rmsL,rmsR,kept (indices into the input lists)."""
    obj_pts, pts_l, pts_r = list(obj_pts), list(pts_l), list(pts_r)
    idx = list(range(len(obj_pts)))
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-8)
    res = None
    for it in range(3):
        rmsL, K1, D1, _, _ = cv2.calibrateCamera(obj_pts, pts_l, size, None, None, flags=cv2.CALIB_FIX_K3)
        rmsR, K2, D2, _, _ = cv2.calibrateCamera(obj_pts, pts_r, size, None, None, flags=cv2.CALIB_FIX_K3)
        out = cv2.stereoCalibrateExtended(obj_pts, pts_l, pts_r, K1, D1, K2, D2, size, None, None,
                                          flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
        rms = out[0]
        K1, D1, K2, D2, R, T = out[1:7]
        per_view = np.asarray(out[-1], dtype=np.float64).reshape(len(obj_pts), -1)
        err = per_view.max(axis=1)
        res = dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, rms=float(rms),
                   rmsL=float(rmsL), rmsR=float(rmsR), kept=list(idx), per_view=err)
        log("  pass %d: %d views, stereo RMS %.3f px (left %.3f, right %.3f)"
            % (it + 1, len(obj_pts), rms, rmsL, rmsR))
        thr = max(2.0 * float(np.median(err)), 0.6)
        keep = err <= thr
        if keep.all() or int(keep.sum()) < min_views:
            break
        log("  dropping %d outlier view(s) with error > %.2f px" % (int((~keep).sum()), thr))
        obj_pts = [o for o, k in zip(obj_pts, keep) if k]
        pts_l = [o for o, k in zip(pts_l, keep) if k]
        pts_r = [o for o, k in zip(pts_r, keep) if k]
        idx = [i for i, k in zip(idx, keep) if k]
    return res


def build_rectification(res, size):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        res["K1"], res["D1"], res["K2"], res["D2"], size, res["R"], res["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return dict(R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, baseline=float(-P2[0, 3] / P2[0, 0]))


def rectified_epipolar_error(res, rect, pts_l, pts_r):
    """Mean / max vertical disparity (px) of the calibration corners after rectification."""
    errs = []
    for pl, pr in zip(pts_l, pts_r):
        ul = cv2.undistortPoints(pl.reshape(-1, 1, 2), res["K1"], res["D1"], R=rect["R1"], P=rect["P1"]).reshape(-1, 2)
        ur = cv2.undistortPoints(pr.reshape(-1, 1, 2), res["K2"], res["D2"], R=rect["R2"], P=rect["P2"]).reshape(-1, 2)
        errs.append(np.abs(ul[:, 1] - ur[:, 1]))
    e = np.concatenate(errs)
    return float(e.mean()), float(e.max())


def save_calibration(path, res, rect, size, board, square_m):
    np.savez(str(path), K1=res["K1"], D1=res["D1"], K2=res["K2"], D2=res["D2"], R=res["R"], T=res["T"],
             R1=rect["R1"], R2=rect["R2"], P1=rect["P1"], P2=rect["P2"], Q=rect["Q"],
             image_size=np.array(size), rms=res["rms"], board=np.array(board), square_m=square_m)


def describe(res, rect, size):
    fx = float(rect["P1"][0, 0])
    hfov = math.degrees(2 * math.atan(size[0] / 2.0 / fx))
    return ("baseline %.1f mm | focal %.1f px | rectified HFOV %.1f deg | stereo RMS %.3f px"
            % (rect["baseline"] * 1000.0, fx, hfov, res["rms"]))
