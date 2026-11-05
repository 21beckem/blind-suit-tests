"""Capture stereo pairs without rectification, depth estimation, or tracking.

Output: metadata.json, frames.bin (packed uint8 left then right), frames.csv.
Run `python 07_record_scan.py --seconds 60`; Ctrl+C stops and finalizes.
"""
import argparse
import csv
import hashlib
import json
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from common import CALIB_PATH, CONFIG_PATH, ROOT, load_config, open_capture, to_gray


def camera_details(cap):
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported": cap.get(cv2.CAP_PROP_FPS),
            "fourcc": "".join(chr((code >> shift) & 255) for shift in (0, 8, 16, 24))}


def record(output, cfg, seconds, queue_size, preview=True):
    output.mkdir(parents=True, exist_ok=False)
    camera = cfg["camera"]
    kw = dict(backend=camera["backend"], fourcc=camera["fourcc"], fps=camera["fps"],
              exposure=camera.get("exposure"), auto_exposure=camera.get("auto_exposure"))
    indices = [camera["index"]] if camera["mode"] == "sbs" else [camera["index"], camera["index_right"]]
    caps = []
    pending = queue.Queue(maxsize=queue_size)
    stats = {"captured": 0, "written": 0, "dropped_queue": 0, "capture_failures": 0,
             "shape_errors": 0, "write_error": None}
    shape = None
    started = time.monotonic_ns()

    def writer():
        try:
            with (output / "frames.bin").open("wb") as data, (output / "frames.csv").open("w", newline="") as index:
                rows = csv.writer(index)
                rows.writerow(["frame_id", "monotonic_ns", "wall_time_ns", "grab_spread_ns", "byte_offset"])
                while True:
                    item = pending.get()
                    if item is None:
                        break
                    fid, mono, wall, spread, pair = item
                    offset = data.tell()
                    data.write(pair.tobytes())
                    rows.writerow([fid, mono, wall, spread, offset])
                    stats["written"] += 1
        except Exception as exc:
            stats["write_error"] = repr(exc)

    thread = threading.Thread(target=writer, daemon=True)
    try:
        for index in indices:
            cap = open_capture(index, width=camera["width"] if camera["mode"] == "dual" else camera["width"],
                               height=camera["height"], **kw)
            if cap is None:
                raise RuntimeError("Could not open camera index %s" % index)
            caps.append(cap)
        actual = [camera_details(cap) for cap in caps]
        thread.start()
        print("Recording to", output, "| camera modes:", actual)
        while seconds is None or (time.monotonic_ns() - started) < seconds * 1e9:
            if stats["write_error"]:
                raise RuntimeError(stats["write_error"])
            spread = 0
            if len(caps) == 1:
                ok, frame = caps[0].read()
                mono = time.monotonic_ns()
                wall = time.time_ns()
                if not ok or frame is None:
                    stats["capture_failures"] += 1
                    continue
                gray = to_gray(frame)
                half = gray.shape[1] // 2
                if gray.shape[1] != half * 2:
                    stats["shape_errors"] += 1
                    continue
                left, right = gray[:, :half], gray[:, half:]
            else:
                t0 = time.monotonic_ns()
                ok_l, ok_r = caps[0].grab(), caps[1].grab()
                t1 = time.monotonic_ns()
                spread = t1 - t0
                ok_l, fl = caps[0].retrieve() if ok_l else (False, None)
                ok_r, fr = caps[1].retrieve() if ok_r else (False, None)
                mono, wall = t1, time.time_ns()
                if not ok_l or not ok_r or fl is None or fr is None:
                    stats["capture_failures"] += 1
                    continue
                left, right = to_gray(fl), to_gray(fr)
            if left.shape != right.shape:
                stats["shape_errors"] += 1
                continue
            if camera.get("swap_eyes"):
                left, right = right, left
            if shape is None:
                shape = list(left.shape)
            if list(left.shape) != shape:
                stats["shape_errors"] += 1
                continue
            stats["captured"] += 1
            pair = np.ascontiguousarray(np.stack((left, right)))
            try:
                pending.put_nowait((stats["captured"], mono, wall, spread, pair))
            except queue.Full:
                stats["dropped_queue"] += 1
            if preview:
                display = np.hstack((left, right))
                if display.shape[1] > 1280:
                    scale = 1280 / display.shape[1]
                    display = cv2.resize(display, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                cv2.putText(display, "REC  %d frames  %d dropped  |  Q/ESC stop" % (
                    stats["captured"], stats["dropped_queue"]), (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.imshow("Stereo recording: left | right", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    except KeyboardInterrupt:
        print("Stopping capture...")
    finally:
        if preview:
            cv2.destroyAllWindows()
        for cap in caps:
            cap.release()
        if thread.ident is not None:
            pending.put(None)
            thread.join()
        metadata = {"format_version": 1, "layout": "uint8 [2,height,width], left then right, row-major",
                    "eye_shape_hw": shape, "camera_requested": camera, "camera_actual": actual if caps else [],
                    "started_local": datetime.now().astimezone().isoformat(), "duration_s": (time.monotonic_ns() - started) / 1e9,
                    "stats": stats}
        if CONFIG_PATH.exists():
            shutil.copy2(CONFIG_PATH, output / "config.json")
        if CALIB_PATH.exists():
            shutil.copy2(CALIB_PATH, output / "calibration.npz")
            metadata["calibration_sha256"] = hashlib.sha256((output / "calibration.npz").read_bytes()).hexdigest()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
    return 1 if stats["write_error"] or stats["written"] == 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=None, help="stop after this many seconds; otherwise Ctrl+C")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--no-preview", action="store_true", help="disable live camera preview")
    args = parser.parse_args()
    if args.queue_size < 1 or (args.seconds is not None and args.seconds <= 0):
        parser.error("queue-size and seconds must be positive")
    output = args.output or ROOT / "recordings" / ("recording_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    return record(output, load_config(), args.seconds, args.queue_size, preview=not args.no_preview)


if __name__ == "__main__":
    raise SystemExit(main())
