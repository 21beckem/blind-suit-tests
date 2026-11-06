"""Extract one recorded pair and baseline depth statistics for browser testing.

Requires `opencv-python-headless` and NumPy in vendor/pydeps. Run from this
directory with `python prepare_fixture.py` after installing those packages.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "pydeps"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SOURCE = ROOT.parent.parent.parent / "binocular" / "3d-scan-tests"
RECORDING = SOURCE / "recordings" / "recording_dining_table"
FRAME_INDEX = 100


def main():
    meta = json.loads((RECORDING / "metadata.json").read_text())
    config = json.loads((SOURCE / "config.json").read_text())
    calibration = np.load(SOURCE / "calibration.npz")
    height, width = meta["eye_shape_hw"]
    frame_bytes = 2 * height * width
    with (RECORDING / "frames.bin").open("rb") as recording:
        recording.seek((FRAME_INDEX - 1) * frame_bytes)
        pair = np.frombuffer(recording.read(frame_bytes), dtype=np.uint8).reshape(2, height, width)
    left, right = pair[0], pair[1]
    sbs = np.hstack([left, right])
    cv2.imwrite(str(ROOT / "sample_sbs.png"), sbs)

    left_map = cv2.initUndistortRectifyMap(calibration["K1"], calibration["D1"],
        calibration["R1"], calibration["P1"], (width, height), cv2.CV_32FC1)
    right_map = cv2.initUndistortRectifyMap(calibration["K2"], calibration["D2"],
        calibration["R2"], calibration["P2"], (width, height), cv2.CV_32FC1)
    rl = cv2.remap(left, *left_map, cv2.INTER_LINEAR)
    rr = cv2.remap(right, *right_map, cv2.INTER_LINEAR)
    s = config["stereo"]
    bs = s["block_size"]
    matcher = cv2.StereoSGBM_create(0, s["num_disparities"], bs,
        8 * bs * bs, 32 * bs * bs, s["disp12_max_diff"], s["pre_filter_cap"],
        s["uniqueness"], s["speckle_window"], s["speckle_range"],
        cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = matcher.compute(rl, rr).astype(np.float32) / 16
    fx = float(calibration["P1"][0, 0])
    baseline = float(-calibration["P2"][0, 3] / calibration["P2"][0, 0])
    depth = np.zeros_like(disparity)
    good = disparity > 0.5
    depth[good] = fx * baseline / disparity[good]
    depth[(depth < s["min_depth"]) | (depth > s["max_depth"])] = 0
    if s["edge_filter"] > 0:
        big = np.where(depth > 0, depth, np.float32(1e6)).astype(np.float32)
        kernel = np.ones((3, 3), np.uint8)
        low = cv2.erode(big, kernel)
        high = cv2.dilate(depth, kernel)
        depth[(depth > 0) & ((high - low) > s["edge_filter"] * depth)] = 0
    valid = depth[depth > 0]
    reference = {
        "recording": RECORDING.name,
        "frame_id": FRAME_INDEX,
        "image_size": [width, height],
        "valid_percent": float(np.mean(depth > 0) * 100),
        "median_depth_m": float(np.median(valid)) if valid.size else None,
        "center_5x5_m": float(np.median(depth[height//2-2:height//2+3, width//2-2:width//2+3][depth[height//2-2:height//2+3, width//2-2:width//2+3] > 0])) if np.any(depth[height//2-2:height//2+3, width//2-2:width//2+3] > 0) else None,
    }
    (ROOT / "sample_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(reference)


if __name__ == "__main__":
    main()
