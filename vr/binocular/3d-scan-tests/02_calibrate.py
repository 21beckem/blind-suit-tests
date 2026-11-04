"""Step 2 - stereo calibration with a printed checkerboard.

    python make_board.py                    # print the board first (100% scale), then measure a square
    python 02_calibrate.py --square-mm 25   # live guided capture (auto-captures when the board is steady)
    python 02_calibrate.py --recalibrate    # redo the maths from the saved images in calib_images/

Tips: hold the board FLAT and still for a moment. Cover the whole image incl. corners, vary distance
(0.4-2 m) and tilt it (up to ~40 degrees). Good result: stereo RMS < 0.5 px.
Keys: SPACE = capture now | U = undo last | ENTER = finish | Q = quit
"""
import argparse
import sys
import time

import cv2
import numpy as np

import calib_tools as ct
from common import CALIB_IMG_DIR, CALIB_PATH, StereoCamera, load_config, save_config

FEATURE_MIN_DIST = 0.10
COOLDOWN_S = 0.8
STABLE_PX = 2.0


def finish(obj_pts, pts_l, pts_r, size, cfg, board, square_m):
    print("\nCalibrating from %d views ..." % len(obj_pts))
    res = ct.run_calibration(obj_pts, pts_l, pts_r, size)
    rect = ct.build_rectification(res, size)
    if rect["baseline"] <= 0:
        print("\nERROR: baseline came out negative -> left/right eyes are swapped.\n"
              "Toggle 'swap_eyes' (press X in 01_probe.py) and run calibration again.")
        return 1
    kept = res["kept"]
    mean_e, max_e = ct.rectified_epipolar_error(res, rect, [pts_l[i] for i in kept], [pts_r[i] for i in kept])
    print("\n" + ct.describe(res, rect, size))
    print("Used %d of %d views | rectified vertical error: mean %.3f px, max %.2f px" % (
        len(kept), len(obj_pts), mean_e, max_e))
    verdict = "EXCELLENT" if res["rms"] < 0.4 else "GOOD" if res["rms"] < 0.7 else \
        "ACCEPTABLE" if res["rms"] < 1.0 else "POOR - recapture with a flatter board and more varied poses"
    print("Quality: %s" % verdict)
    print("Sanity check: does the baseline match your module's physical lens spacing? If it is off by a constant\n"
          "factor, your --square-mm value is wrong (measure a printed square).")
    ct.save_calibration(CALIB_PATH, res, rect, size, board, square_m)
    cfg["calibration"].update(board_cols=board[0], board_rows=board[1], square_mm=square_m * 1000.0)
    save_config(cfg)
    print("Saved %s. Next: python 03_depth_test.py" % CALIB_PATH.name)
    return 0


def load_saved_pairs(cols, rows):
    files = sorted(CALIB_IMG_DIR.glob("pair_*_L.png"))
    objs, pl, pr, size = [], [], [], None
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(str(f).replace("_L.png", "_R.png"), cv2.IMREAD_GRAYSCALE)
        if g is None or gr is None:
            continue
        size = (g.shape[1], g.shape[0])
        cl, cr = ct.find_corners(g, cols, rows), ct.find_corners(gr, cols, rows)
        print("  %s: %s" % (f.name, "ok" if cl is not None and cr is not None else "board not found, skipped"))
        if cl is not None and cr is not None:
            pl.append(cl)
            pr.append(cr)
    return pl, pr, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--views", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None, help="INNER corners across")
    ap.add_argument("--rows", type=int, default=None, help="INNER corners down")
    ap.add_argument("--square-mm", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    c = cfg["calibration"]
    cols = args.cols or c["board_cols"]
    rows = args.rows or c["board_rows"]
    square_mm = args.square_mm or c["square_mm"]
    want = args.views or c["num_views"]
    square_m = square_mm / 1000.0
    board = (cols, rows)
    obj = ct.board_object_points(cols, rows, square_m)
    print("Board: %dx%d inner corners, %.2f mm squares" % (cols, rows, square_mm))

    if args.recalibrate:
        pl, pr, size = load_saved_pairs(cols, rows)
        if len(pl) < 10:
            print("Need at least 10 usable saved pairs (found %d)." % len(pl))
            return 1
        return finish([obj] * len(pl), pl, pr, size, cfg, board, square_m)

    CALIB_IMG_DIR.mkdir(exist_ok=True)
    for old in CALIB_IMG_DIR.glob("pair_*.png"):
        old.unlink()

    cam = StereoCamera(cfg["camera"]).start()
    pts_l, pts_r, feats, centroids = [], [], [], []
    size, last_id, last_corners, last_capture = None, -1, None, 0.0
    msg = "Show the board to both eyes"
    try:
        while True:
            item = cam.read(last_id, 1.0)
            if item is None:
                print("No frames from camera.")
                return 1
            l, r, _t, last_id = item
            size = (l.shape[1], l.shape[0])
            cl = ct.find_corners(l, cols, rows)
            cr = ct.find_corners(r, cols, rows) if cl is not None else None
            both = cl is not None and cr is not None
            sane = both and float(np.abs(cl[:, 0, 1] - cr[:, 0, 1]).mean()) < 25.0 and \
                float((cl[:, 0, 0] - cr[:, 0, 0]).mean()) > 0
            if both and not sane:
                msg = "Eyes disagree (corner order/swap?) - check swap_eyes in 01_probe.py"
            elif both:
                stable = last_corners is not None and \
                    float(np.linalg.norm(cl[:, 0] - last_corners[:, 0], axis=1).mean()) < STABLE_PX
                f = ct.pose_features(cl, cols, rows, size)
                diverse = all(np.linalg.norm(f - g) > FEATURE_MIN_DIST for g in feats)
                msg = "Board found" + ("" if stable else " - hold still") + ("" if diverse else " - move/tilt to a NEW pose")
                if stable and diverse and time.time() - last_capture > COOLDOWN_S:
                    k = len(pts_l)
                    cv2.imwrite(str(CALIB_IMG_DIR / ("pair_%03d_L.png" % k)), l)
                    cv2.imwrite(str(CALIB_IMG_DIR / ("pair_%03d_R.png" % k)), r)
                    pts_l.append(cl)
                    pts_r.append(cr)
                    feats.append(f)
                    centroids.append(cl[:, 0].mean(axis=0))
                    last_capture = time.time()
                    print("captured %d/%d" % (len(pts_l), want))
            else:
                msg = "Board not visible in both eyes"
            last_corners = cl

            # ---- drawing
            vl, vr = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR), cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
            if cl is not None:
                cv2.drawChessboardCorners(vl, (cols, rows), cl, True)
            if cr is not None:
                cv2.drawChessboardCorners(vr, (cols, rows), cr, True)
            for cx, cy in centroids:
                cv2.circle(vl, (int(cx), int(cy)), 10, (0, 255, 255), 2)
            vis = np.hstack([vl, vr])
            scale = min(1.0, 1600.0 / vis.shape[1])
            if scale < 1.0:
                vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            col = (0, 255, 0) if both else (0, 120, 255)
            cv2.putText(vis, "%d / %d views | %s" % (len(pts_l), want, msg), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            cv2.imshow("02_calibrate  (SPACE capture | U undo | ENTER finish | Q quit)", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord(" ") and sane:
                k = len(pts_l)
                cv2.imwrite(str(CALIB_IMG_DIR / ("pair_%03d_L.png" % k)), l)
                cv2.imwrite(str(CALIB_IMG_DIR / ("pair_%03d_R.png" % k)), r)
                pts_l.append(cl)
                pts_r.append(cr)
                feats.append(ct.pose_features(cl, cols, rows, size))
                centroids.append(cl[:, 0].mean(axis=0))
                last_capture = time.time()
            if key in (ord("u"), ord("U")) and pts_l:
                pts_l.pop(), pts_r.pop(), feats.pop(), centroids.pop()
                k = len(pts_l)
                for suf in ("L", "R"):
                    p = CALIB_IMG_DIR / ("pair_%03d_%s.png" % (k, suf))
                    if p.exists():
                        p.unlink()
            if len(pts_l) >= want or (key in (13, 10) and len(pts_l) >= 10):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    if len(pts_l) < 10:
        print("Only %d views captured; need at least 10." % len(pts_l))
        return 1
    return finish([obj] * len(pts_l), pts_l, pts_r, size, cfg, board, square_m)


if __name__ == "__main__":
    sys.exit(main())
