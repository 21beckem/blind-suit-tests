"""Reusable CPU stereo matcher. Construct once, then compute(left, right)."""

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Rectifier, load_calibration


@dataclass
class DepthResult:
    """Arrays at working resolution; disparity is pixels, 0 means invalid.

    confidence: 0 invalid, 1 weak (low texture), 2 solid.
    Q: reprojection matrix adjusted to the working resolution.
    """

    disparity: np.ndarray
    confidence: np.ndarray
    gray: np.ndarray
    Q: np.ndarray
    max_disparity: int

    @property
    def depth_m(self):
        """Metric Z image, with invalid pixels set to 0; calculated on demand."""
        xyz = cv2.reprojectImageTo3D(self.disparity, self.Q)
        z = xyz[:, :, 2]
        return np.where((self.disparity > 0) & np.isfinite(z) & (z > 0), z, 0)


class DepthMapper:
    def __init__(self, config):
        settings = config["depth"]
        self.scale = float(settings["scale"])
        self.max_disparity = int(np.ceil(settings["num_disp"] * self.scale / 16) * 16)
        self.block = settings["block"]
        if not 0 < self.scale <= 1:
            raise ValueError("depth.scale must be greater than 0 and at most 1")
        if settings["num_disp"] <= 0:
            raise ValueError("depth.num_disp must be positive")
        if self.block < 1 or self.block % 2 != 1:
            raise ValueError("depth.block must be a positive odd integer")
        if settings["max_rel_err"] <= 0:
            raise ValueError("depth.max_rel_err must be positive")
        self._min_disparity = max(1.0, 0.35 / settings["max_rel_err"])
        self._texture_min = settings["tex_min"]
        calibration = load_calibration(config["calibration"])
        self._rectifier = Rectifier(calibration) if settings["rectify"] else None
        q = self._rectifier.Q if self._rectifier is not None else calibration["Q"]
        self.Q = np.array(q, dtype=np.float64, copy=True)
        if self.Q.shape != (4, 4) or not np.isfinite(self.Q).all():
            raise ValueError("Calibration Q must be a finite 4x4 matrix")
        self.Q[:, 3] *= self.scale
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.max_disparity, blockSize=self.block,
            P1=8 * self.block ** 2, P2=32 * self.block ** 2,
            disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=10,
            speckleWindowSize=120, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self._kernel = np.ones((3, 3), np.uint8)

    def compute(self, left, right):
        """Accept uint8 grayscale/BGR frames in left, right order. Never swaps eyes."""
        for image in (left, right):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise ValueError("Frames must be uint8 NumPy images")
            if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
                raise ValueError("Frames must be grayscale or three-channel BGR")
            if image.size == 0:
                raise ValueError("Frames cannot be empty")
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("Left and right frames must have the same size")
        if self._rectifier is not None:
            left, right = self._rectifier.rectify(left, right)
        else:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
        if self.scale != 1:
            left = cv2.resize(left, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        # Recover the columns SGBM would otherwise leave empty on the left.
        pad = self.max_disparity
        padded_left = cv2.copyMakeBorder(left, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        padded_right = cv2.copyMakeBorder(right, 0, 0, pad, 0, cv2.BORDER_REPLICATE)
        disparity = self._matcher.compute(padded_left, padded_right)[:, pad:].astype(np.float32) / 16
        disparity[disparity < self._min_disparity] = 0
        xs = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
        disparity[xs - disparity < self.block // 2] = 0
        valid = disparity > 0
        hi = cv2.dilate(np.where(valid, disparity, 0).astype(np.float32), self._kernel)
        lo = cv2.erode(np.where(valid, disparity, 1e4).astype(np.float32), self._kernel)
        # Reject depth-edge artifacts ("flying pixels") using the original thresholds.
        disparity[valid & (hi - lo > np.maximum(1.5, 0.05 * disparity))] = 0
        gradient = cv2.Sobel(cv2.GaussianBlur(left, (3, 3), 0), cv2.CV_32F, 1, 0, ksize=3)
        texture = cv2.boxFilter(np.abs(gradient), -1, (2 * self.block + 1, 2 * self.block + 1))
        confidence = np.where(disparity > 0, np.where(texture >= self._texture_min, 2, 1), 0).astype(np.uint8)
        return DepthResult(disparity, confidence, left, self.Q, self.max_disparity)
