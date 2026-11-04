"""Step 3 - verify calibration and tune stereo matching before scanning.

    python 03_depth_test.py [--scale 0.5] [--display 0.75]

Panels: rectified LEFT | rectified RIGHT (horizontal green lines: matching features must sit on the SAME line)
        disparity colour map | depth colour map (red = near).
CLICK anywhere to read the distance there; check it against a tape measure.
Keys: S = save slider settings to config.json | C = toggle CLAHE | SPACE = freeze/unfreeze | Q = quit
"""
import argparse
import sys
import time

import cv2
import numpy as np

from common import (Calibration, Rectifier, StereoCamera, StereoMatcher, epipolar_error,
                    load_config, save_config)

WIN = "03_depth_test"


def colorize_disp(disp, nd):
    v = np.clip(disp * (255.0 / nd), 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
    out[disp <= 0.5] = 0
    return out


def colorize_depth(depth, zmax):
    v = np.clip(depth / zmax * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(255 - v, cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--display", type=float, default=0.75, help="window size factor")
    args = ap.parse_args()

    cfg = load_config()
    sc = cfg["stereo"]
    if args.scale:
        sc["scale"] = args.scale
    calib = Calibration()
    rect = Rectifier(calib, sc["scale"])
    print("Baseline %.1f mm | processing size %dx%d | fx %.1f px" % (
        rect.baseline * 1000, rect.out_size[0], rect.out_size[1], rect.fx))
    matcher = StereoMatcher(sc, rect.fx, rect.baseline)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    tb = [("numDisp/16", max(2, matcher.nd // 16), 24), ("blockSize", sc["block_size"], 21),
          ("uniqueness", sc["uniqueness"], 30), ("speckleWin", sc["speckle_window"], 400),
          ("speckleRng", sc["speckle_range"], 8), ("minDepth cm", int(sc["min_depth"] * 100), 300),
          ("maxDepth cm", int(sc["max_depth"] * 100), 1000), ("edge %", int(sc["edge_filter"] * 100), 40)]
    for name, init, mx in tb:
        cv2.createTrackbar(name, WIN, int(init), mx, lambda _v: None)

    click = {"p": None}

    def on_mouse(event, x, y, _flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            w, h = rect.out_size
            click["p"] = (int((x / args.display) % w), int((y / args.display) % h))
    cv2.setMouseCallback(WIN, on_mouse)

    cam = StereoCamera(cfg["camera"]).start()
    last_id, frozen, pair, fps, t_prev, epi = -1, False, None, 0.0, time.time(), (None, 0)
    n = 0
    prev_key = None
    try:
        while True:
            if not frozen or pair is None:
                item = cam.read(last_id, 1.0)
                if item is None:
                    print("No frames from camera.")
                    break
                l, r, _t, last_id = item
                pair = rect.rectify(l, r)
            rl, rr = pair

            vals = {name: max(cv2.getTrackbarPos(name, WIN), 0) for name, _i, _m in tb}
            sc["num_disparities"] = max(2, vals["numDisp/16"]) * 16
            sc["block_size"] = max(3, vals["blockSize"] | 1)
            sc["uniqueness"], sc["speckle_window"] = vals["uniqueness"], vals["speckleWin"]
            sc["speckle_range"] = vals["speckleRng"]
            sc["min_depth"] = max(vals["minDepth cm"], 5) / 100.0
            sc["max_depth"] = max(vals["maxDepth cm"], 50) / 100.0
            sc["edge_filter"] = vals["edge %"] / 100.0
            key_now = tuple(sorted((k, v) for k, v in sc.items() if k != "scale"))
            if key_now != prev_key:
                matcher.cfg = dict(sc)
                matcher.build()
                prev_key = key_now

            disp, depth = matcher.compute(rl, rr)
            n += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3) if fps else 1.0 / max(now - t_prev, 1e-3)
            t_prev = now
            if n % 20 == 1:
                epi = epipolar_error(rl, rr)

            h, w = rl.shape
            lc, rc = cv2.cvtColor(rl, cv2.COLOR_GRAY2BGR), cv2.cvtColor(rr, cv2.COLOR_GRAY2BGR)
            step = max(20, h // 12)
            for y in range(step // 2, h, step):
                cv2.line(lc, (0, y), (w, y), (0, 200, 0), 1)
                cv2.line(rc, (0, y), (w, y), (0, 200, 0), 1)
            dc, zc = colorize_disp(disp, matcher.nd), colorize_depth(depth, sc["max_depth"])

            if click["p"] is not None:
                px, py = click["p"]
                patch = depth[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3]
                v = patch[patch > 0]
                label = "%.3f m" % float(np.median(v)) if v.size else "no depth here"
                for img in (lc, dc, zc):
                    cv2.drawMarker(img, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
                cv2.putText(zc, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            valid_pct = 100.0 * float((depth > 0).mean())
            epi_txt = "epipolar err %.2f px (%d matches)" % epi if epi[0] is not None else "epipolar err n/a"
            cv2.putText(lc, "%.1f fps | valid %.0f%% | closest %.2f m" % (fps, valid_pct, matcher.closest_depth()),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(rc, epi_txt + (" [FROZEN]" if frozen else ""), (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            canvas = np.vstack([np.hstack([lc, rc]), np.hstack([dc, zc])])
            if args.display != 1.0:
                canvas = cv2.resize(canvas, None, fx=args.display, fy=args.display, interpolation=cv2.INTER_AREA)
            cv2.imshow(WIN, canvas)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                frozen = not frozen
            if k in (ord("c"), ord("C")):
                sc["clahe"] = not sc.get("clahe", False)
                matcher.cfg = dict(sc)
                matcher.build()
                print("CLAHE:", sc["clahe"])
            if k in (ord("s"), ord("S")):
                cfg["stereo"] = sc
                save_config(cfg)
                print("Saved stereo settings to config.json")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
