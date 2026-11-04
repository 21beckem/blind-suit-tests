"""Step 1 - find the camera, learn how it presents itself, choose a mode, verify left/right eyes.

    python 01_probe.py                 # auto-detect and sweep modes
    python 01_probe.py --index 1       # you already know which camera index is the module
    python 01_probe.py --dual 0 1      # module shows up as TWO separate webcams
    python 01_probe.py --mode 2560x800x60   # test just one mode

Writes the result to config.json (after you confirm).
"""
import argparse
import sys
import time

import cv2
import numpy as np

from common import StereoCamera, load_config, open_capture, save_config, to_gray, eye_size, CALIB_PATH

CANDIDATE_MODES = [(2560, 800, 60), (2560, 800, 30), (1280, 400, 120), (1280, 400, 60),
                   (1280, 800, 60), (1280, 800, 30), (640, 400, 120), (640, 480, 30), (1280, 720, 30)]


def parse_mode(s):
    w, h, f = (int(x) for x in s.lower().split("x"))
    return w, h, f


def list_cameras(backend, max_index):
    found = []
    for i in range(max_index):
        cap = open_capture(i, backend)
        if cap is None:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            found.append((i, frame.shape[1], frame.shape[0]))
    return found


def eyes_look_stereo(gray):
    """Normalised correlation between the two halves. Real stereo pairs score high (>0.3)."""
    half = gray.shape[1] // 2
    a = cv2.resize(gray[:, :half], (160, 100)).astype(np.float32).ravel()
    b = cv2.resize(gray[:, half:2 * half], (160, 100)).astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-6 else 0.0


def test_mode(index, backend, w, h, fps, fourcc="MJPG", seconds=2.0):
    cap = open_capture(index, backend, w, h, fps, fourcc)
    if cap is None:
        return None
    for _ in range(6):                      # warm-up (auto exposure, driver buffers)
        cap.read()
    n, t0, sample = 0, time.time(), None
    while time.time() - t0 < seconds:
        ok, f = cap.read()
        if ok and f is not None:
            n += 1
            sample = f
    dt = time.time() - t0
    cap.release()
    if sample is None or n < 3:
        return None
    return dict(w=sample.shape[1], h=sample.shape[0], fps=n / dt, sample=sample)


def choose_best(results):
    best, best_score = None, -1.0
    for r in results:
        sbs = r["w"] / r["h"] >= 2.5
        eye_px = (r["w"] // 2 if sbs else r["w"]) * r["h"]
        f = min(r["fps"], 60.0)
        score = eye_px * f * (1.0 if r["fps"] >= 45 else 0.3 if r["fps"] >= 25 else 0.05)
        r["sbs"] = sbs
        if score > best_score:
            best, best_score = r, score
    return best


def preview(cam_cfg):
    """Live side-by-side preview. X = swap eyes, S = save & quit, Q = quit without saving."""
    cam = StereoCamera(cam_cfg).start()
    print("\nPreview: COVER YOUR LEFT LENS with a finger. The LEFT half of the picture must go dark.")
    print("If the RIGHT half goes dark instead, press X to swap eyes.  S = save, Q = quit.\n")
    last, saved = -1, False
    try:
        while True:
            item = cam.read(last, 1.0)
            if item is None:
                print("No frames arriving from the camera.")
                break
            l, r, _t, last = item
            vis = np.hstack([l, r])
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            scale = min(1.0, 1600.0 / vis.shape[1])
            if scale < 1.0:
                vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = vis.shape[:2]
            cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
            sync = ("dual: grab spread %.1f ms" % cam.grab_spread_ms if cam_cfg["mode"] == "dual"
                    else "single USB frame (hardware-synced)")
            cv2.putText(vis, "LEFT eye   mean=%3d" % l.mean(), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "RIGHT eye  mean=%3d" % r.mean(), (w // 2 + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "%.1f fps | %dx%d per eye | %s | swap=%s" % (
                cam.fps, l.shape[1], l.shape[0], sync, cam_cfg["swap_eyes"]),
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("01_probe  (X swap eyes | S save | Q quit)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("x"), ord("X")):
                cam_cfg["swap_eyes"] = not cam_cfg["swap_eyes"]
                cam.cfg["swap_eyes"] = cam_cfg["swap_eyes"]
            elif k in (ord("s"), ord("S")):
                saved = True
                break
            elif k in (ord("q"), ord("Q"), 27):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["dshow", "msmf", "any"], default=None)
    ap.add_argument("--max-index", type=int, default=8)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--dual", type=int, nargs=2, metavar=("LEFT", "RIGHT"))
    ap.add_argument("--mode", type=parse_mode, default=None, help="WxHxFPS, e.g. 2560x800x60")
    ap.add_argument("--yes", action="store_true", help="accept the recommended mode without asking")
    args = ap.parse_args()

    cfg = load_config()
    cam = cfg["camera"]
    if args.backend:
        cam["backend"] = args.backend
    backend = cam["backend"]

    if args.dual:
        cam["mode"], cam["index"], cam["index_right"] = "dual", args.dual[0], args.dual[1]
        w, h, f = args.mode or (1280, 800, 60)
        cam["width"], cam["height"], cam["fps"] = w, h, f
        print("Dual-device mode: left=%d right=%d at %dx%d@%d (per eye)" % (args.dual[0], args.dual[1], w, h, f))
    else:
        print("Scanning camera indices 0..%d with backend '%s' ..." % (args.max_index - 1, backend))
        found = list_cameras(backend, args.max_index)
        if not found:
            print("No cameras found. Check the USB cable/port (USB 2.0 is fine), close other apps using the camera,\n"
                  "and try --backend msmf.")
            return 1
        for i, w, h in found:
            tag = "  <-- looks like side-by-side stereo" if w / h >= 2.5 else ""
            print("  index %d: default %dx%d%s" % (i, w, h, tag))
        idx = args.index
        if idx is None:
            sbs = [i for i, w, h in found if w / h >= 2.5]
            if len(sbs) == 1:
                idx = sbs[0]
            elif len(found) == 1:
                idx = found[0][0]
            else:
                idx = int(input("Which index is the stereo module? ").strip())
        cam["mode"], cam["index"] = "sbs", idx

        modes = [args.mode] if args.mode else CANDIDATE_MODES
        print("\nTesting modes on index %d (about 2 s each) ..." % idx)
        results, seen = [], set()
        for (w, h, f) in modes:
            r = test_mode(idx, backend, w, h, f)
            if r is None:
                print("  %4dx%-4d @%3d  -> not supported" % (w, h, f))
                continue
            key = (r["w"], r["h"], round(r["fps"] / 5))
            gray = to_gray(r["sample"])
            corr = eyes_look_stereo(gray) if r["w"] / r["h"] >= 2.5 else float("nan")
            mono = r["sample"].ndim == 3 and float(np.abs(
                r["sample"][:, :, 0].astype(np.int16) - r["sample"][:, :, 2]).mean()) < 2.0
            print("  %4dx%-4d @%3d  -> got %4dx%-4d at %5.1f fps%s%s" % (
                w, h, f, r["w"], r["h"], r["fps"],
                "  eye-correlation %.2f" % corr if corr == corr else "", "  (grayscale)" if mono else ""))
            if key not in seen:
                seen.add(key)
                results.append(r)
        if not results:
            print("No usable modes. Try --backend msmf, another USB port, or --mode WxHxFPS.")
            return 1
        best = choose_best(results)
        eye_w = best["w"] // 2 if best["sbs"] else best["w"]
        print("\nRecommended: %dx%d @ %.0f fps  (%s, %dx%d per eye)" % (
            best["w"], best["h"], best["fps"], "side-by-side" if best["sbs"] else "single image", eye_w, best["h"]))
        if not best["sbs"]:
            print("WARNING: that mode is not side-by-side. If your module should output both eyes in one frame, "
                  "try --mode 2560x800x60 or --dual 0 1.")
        if not args.yes:
            a = input("Use this mode? [Y/n, or type WxHxFPS to pick another]: ").strip().lower()
            if a == "n":
                return 0
            if "x" in a:
                w, h, f = parse_mode(a)
                best = dict(w=w, h=h, fps=f, sbs=w / h >= 2.5)
        cam["width"], cam["height"] = best["w"], best["h"]
        cam["fps"] = int(round(min(best["fps"], 120) / 5.0) * 5) or 30
        # Use the modes' nominal fps if we are close to it (measured fps is a hair below nominal).
        for nominal in (120, 60, 30):
            if abs(best["fps"] - nominal) <= max(4.0, nominal * 0.1):
                cam["fps"] = nominal
                break

    saved = preview(cam)
    if not saved:
        print("Not saved. Re-run when ready.")
        return 0
    ew, eh = eye_size(cam)
    cfg["stereo"]["scale"] = 0.5 if ew >= 1000 else 1.0
    save_config(cfg)
    print("\nSaved config.json: %s index %s, %dx%d @ %d fps, swap_eyes=%s, per-eye %dx%d, processing scale %.1f" % (
        cam["mode"], cam["index"], cam["width"], cam["height"], cam["fps"], cam["swap_eyes"], ew, eh,
        cfg["stereo"]["scale"]))
    if CALIB_PATH.exists():
        print("NOTE: calibration.npz exists. If you changed mode/resolution/eye-swap, re-run 02_calibrate.py.")
    print("Next: python 02_calibrate.py   (print a board first: python make_board.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
