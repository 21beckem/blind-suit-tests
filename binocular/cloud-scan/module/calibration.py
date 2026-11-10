"""Load OpenCV stereo calibration and cache raw-frame rectification maps."""

import cv2
import numpy as np


def load_calibration(path):
    """Read an .npz produced by stereoCalibrate/stereoRectify (T in metres)."""
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


class Rectifier:
    def __init__(self, calibration):
        c = calibration
        required = ("K1", "D1", "K2", "D2", "image_size")
        missing = [key for key in required if key not in c]
        if missing:
            raise ValueError(f"Calibration is missing: {', '.join(missing)}")
        self.size = tuple(int(value) for value in c["image_size"].ravel())
        if all(key in c for key in ("R1", "R2", "P1", "P2")):
            r1, r2, p1, p2 = (c[key] for key in ("R1", "R2", "P1", "P2"))
            if "Q" in c:
                q = c["Q"]
            else:
                baseline = -p2[0, 3] / p2[0, 0]
                q = np.array([[1, 0, 0, -p1[0, 2]],
                              [0, 1, 0, -p1[1, 2]],
                              [0, 0, 0, p1[0, 0]],
                              [0, 0, 1 / baseline, (p1[0, 2] - p2[0, 2]) / baseline]])
        else:
            if "R" not in c or "T" not in c:
                raise ValueError("Calibration needs R/T or R1/R2/P1/P2 for rectification")
            r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
                c["K1"], c["D1"], c["K2"], c["D2"], self.size,
                c["R"], np.asarray(c["T"], dtype=np.float64).reshape(3, 1),
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.Q = np.asarray(q, dtype=np.float64)
        self._left_map = cv2.initUndistortRectifyMap(
            c["K1"], c["D1"], r1, p1[:, :3], self.size, cv2.CV_16SC2)
        self._right_map = cv2.initUndistortRectifyMap(
            c["K2"], c["D2"], r2, p2[:, :3], self.size, cv2.CV_16SC2)

    def rectify(self, left, right):
        for image in (left, right):
            if (image.shape[1], image.shape[0]) != self.size:
                raise ValueError(f"Frame is {image.shape[1]}x{image.shape[0]}; calibration "
                                 f"requires {self.size[0]}x{self.size[1]} per eye")
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        if right.ndim == 3:
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        return (cv2.remap(left, *self._left_map, cv2.INTER_LINEAR),
                cv2.remap(right, *self._right_map, cv2.INTER_LINEAR))
