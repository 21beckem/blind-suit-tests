"""Short-range stereo visual odometry with strict map-integration gating."""

import math

import cv2
import numpy as np

from .map_geometry import Pose2D


class PoseTracker:
    """Estimate world X/Z/yaw; obtain tilt and camera height from the floor fit.

    A confidence-2 keyframe supplies 3D points. LK or ORB matches locate them
    in the current rectified left image, then PnP RANSAC estimates motion.
    Failed tracking returns a pose with tracking_ok=False. After a long loss,
    reinitialization requests a fresh rolling map rather than joining poses
    across an unobserved motion interval.
    """

    def __init__(self, config):
        self.settings = config["map"]
        self._anchor = None
        self._anchor_floor = None
        self._anchor_pose = None
        self._anchor_points_2d = None
        self._anchor_points_3d = None
        self._previous = None
        self._previous_floor = None
        self._previous_pose = None
        self._previous_points_2d = None
        self._previous_points_3d = None
        self._orb = cv2.ORB_create(nfeatures=self.settings["tracking_orb_features"])
        # OpenCV initializes feature kernels lazily. Pay that one-time cost at
        # construction, outside the live per-frame mapping benchmark.
        warm_height = max(32, int(config["camera"]["height"] * config["depth"]["scale"]))
        warm_width = max(32, int(config["camera"]["width"] * config["depth"]["scale"] / 2))
        warm_image = np.random.default_rng(0).integers(
            0, 256, (warm_height, warm_width), dtype=np.uint8)
        cv2.goodFeaturesToTrack(
            warm_image, maxCorners=self.settings["tracking_max_corners_per_bucket"],
            qualityLevel=self.settings["tracking_corner_quality"],
            minDistance=self.settings["tracking_corner_min_distance_px"])
        self._orb.detectAndCompute(warm_image, None)
        self._orb_keypoints = None
        self._orb_descriptors = None
        self._orb_points_3d = None
        self.pose = Pose2D(tracking_ok=False)
        self.tracking_ok = False
        self.inliers = 0
        self.method = "uninitialized"
        self.reset_map = False
        self.reset_count = 0
        self.rejection = None
        self._last_good_s = None
        self._lost_since_s = None
        self._last_usable_floor = None
        self._last_usable_floor_s = None
        self._ever_initialized = False

    @staticmethod
    def _camera_matrix(result):
        q = result.Q
        focal = float(q[2, 3])
        return np.array([[focal, 0, -q[0, 3]],
                         [0, focal, -q[1, 3]], [0, 0, 1]], np.float64)

    def _lift(self, result, points_2d):
        rows = np.rint(points_2d[:, 1]).astype(np.int32)
        cols = np.rint(points_2d[:, 0]).astype(np.int32)
        height, width = result.disparity.shape
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)
        disp = result.disparity[rows, cols]
        valid = inside & (result.confidence[rows, cols] == 2) & (disp > 0)
        pixels = np.stack((points_2d[:, 0], points_2d[:, 1], disp,
                           np.ones(len(disp))), axis=0)
        xyz_h = np.asarray(result.Q, np.float64) @ pixels
        w = xyz_h[3]
        valid &= np.isfinite(xyz_h).all(axis=0) & (w > 1e-9)
        p_cam = np.full((len(disp), 3), np.nan, np.float64)
        p_cam[valid] = (xyz_h[:3, valid] / w[valid]).T
        valid &= (p_cam[:, 2] >= self.settings["tracking_min_range_m"])
        valid &= (p_cam[:, 2] <= self.settings["tracking_max_range_m"])
        return p_cam.astype(np.float32), valid

    def _bucketed_corners(self, result):
        s = self.settings
        rows, cols = s["tracking_bucket_rows"], s["tracking_bucket_cols"]
        h, w = result.gray.shape
        mask = ((result.confidence == 2) & (result.disparity > 0)).astype(np.uint8) * 255
        corners = []
        for row in range(rows):
            y0, y1 = row * h // rows, (row + 1) * h // rows
            for col in range(cols):
                x0, x1 = col * w // cols, (col + 1) * w // cols
                found = cv2.goodFeaturesToTrack(
                    result.gray[y0:y1, x0:x1],
                    maxCorners=s["tracking_max_corners_per_bucket"],
                    qualityLevel=s["tracking_corner_quality"],
                    minDistance=s["tracking_corner_min_distance_px"],
                    mask=mask[y0:y1, x0:x1])
                if found is not None:
                    found = found.reshape(-1, 2)
                    found += np.array([x0, y0], np.float32)
                    corners.append(found)
        if not corners:
            return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
        points_2d = np.concatenate(corners).astype(np.float32)
        points_3d, valid = self._lift(result, points_2d)
        return points_2d[valid], points_3d[valid]

    def _set_anchor(self, result, floor, pose):
        points_2d, points_3d = self._bucketed_corners(result)
        if len(points_2d) < self.settings["tracking_min_inliers"]:
            return False
        self._anchor = result
        self._anchor_floor = floor
        self._anchor_pose = pose
        self._anchor_points_2d = points_2d
        self._anchor_points_3d = points_3d
        mask = ((result.confidence == 2) & (result.disparity > 0)).astype(np.uint8) * 255
        keypoints, descriptors = self._orb.detectAndCompute(result.gray, mask)
        if keypoints:
            orb_2d = np.array([point.pt for point in keypoints], np.float32)
            orb_3d, valid = self._lift(result, orb_2d)
            self._orb_keypoints = [point for point, keep in zip(keypoints, valid) if keep]
            self._orb_descriptors = descriptors[valid] if descriptors is not None else None
            self._orb_points_3d = orb_3d[valid]
        else:
            self._orb_keypoints = []
            self._orb_descriptors = None
            self._orb_points_3d = None
        return True

    def _set_previous(self, result, floor, pose):
        points_2d, points_3d = self._bucketed_corners(result)
        if len(points_2d) < self.settings["tracking_min_inliers"]:
            return False
        self._previous = result
        self._previous_floor = floor
        self._previous_pose = pose
        self._previous_points_2d = points_2d
        self._previous_points_3d = points_3d
        return True

    def _lk_correspondences(self, result):
        if self._previous is None:
            return None
        source = self._previous_points_2d.reshape(-1, 1, 2)
        target, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous.gray, result.gray, source, None,
            winSize=(self.settings["tracking_lk_window_px"],) * 2,
            maxLevel=self.settings["tracking_lk_pyramid_levels"],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      self.settings["tracking_lk_iterations"],
                      self.settings["tracking_lk_epsilon"]))
        if target is None:
            return None
        target = target.reshape(-1, 2)
        valid = status.ravel().astype(bool) & np.isfinite(target).all(axis=1)
        h, w = result.gray.shape
        valid &= (target[:, 0] >= 0) & (target[:, 0] < w)
        valid &= (target[:, 1] >= 0) & (target[:, 1] < h)
        return self._previous_points_3d[valid], target[valid], self._previous_points_2d[valid]

    def _orb_correspondences(self, result):
        if self._orb_descriptors is None or len(self._orb_descriptors) < 25:
            return None
        keypoints, descriptors = self._orb.detectAndCompute(result.gray, None)
        if descriptors is None or len(descriptors) < 25:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            self._orb_descriptors, descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and
                pair[0].distance < self.settings["tracking_orb_ratio"] * pair[1].distance]
        if len(good) < self.settings["tracking_min_inliers"]:
            return None
        source = np.array([self._orb_keypoints[m.queryIdx].pt for m in good], np.float32)
        target = np.array([keypoints[m.trainIdx].pt for m in good], np.float32)
        points_3d = np.array([self._orb_points_3d[m.queryIdx] for m in good], np.float32)
        return points_3d, target, source

    def _solve(self, correspondences, result, floor, timestamp_s,
               source_floor=None, source_pose=None):
        if correspondences is None:
            self.rejection = "few_matches"
            return None
        points_3d, target_2d, source_2d = correspondences
        s = self.settings
        if len(points_3d) < s["tracking_min_inliers"]:
            self.rejection = "few_matches"
            return None
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, target_2d, self._camera_matrix(result), None,
                iterationsCount=s["tracking_ransac_iterations"],
                reprojectionError=s["tracking_reprojection_px"],
                confidence=s["tracking_ransac_confidence"], flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            self.rejection = "pnp_error"
            return None
        if not ok or inliers is None or len(inliers) < s["tracking_min_inliers"]:
            self.rejection = "few_inliers"
            return None
        inlier_points = source_2d[inliers.ravel()]
        h, w = result.gray.shape
        bucket_x = np.clip((inlier_points[:, 0] * s["tracking_bucket_cols"] / w).astype(int),
                           0, s["tracking_bucket_cols"] - 1)
        bucket_y = np.clip((inlier_points[:, 1] * s["tracking_bucket_rows"] / h).astype(int),
                           0, s["tracking_bucket_rows"] - 1)
        spread = len(np.unique(bucket_y * s["tracking_bucket_cols"] + bucket_x))
        if spread < s["tracking_min_spread_buckets"]:
            self.rejection = "clustered_inliers"
            return None
        rotation_cur_anchor, _ = cv2.Rodrigues(rvec)
        center_anchor_cam = -rotation_cur_anchor.T @ tvec.reshape(3)
        source_floor = source_floor or self._anchor_floor
        source_pose = source_pose or self._anchor_pose
        center_anchor_grav = source_floor.rotation_grav_cam @ center_anchor_cam
        rotation_anchor_grav_cur_grav = (
            source_floor.rotation_grav_cam @ rotation_cur_anchor.T @
            floor.rotation_grav_cam.T)
        forward_anchor_grav = rotation_anchor_grav_cur_grav @ np.array([0., 0., 1.])
        yaw_relative = math.atan2(forward_anchor_grav[0], forward_anchor_grav[2])
        anchor_yaw = source_pose.yaw_rad
        c, si = math.cos(anchor_yaw), math.sin(anchor_yaw)
        x_world = (source_pose.x_m + c * center_anchor_grav[0] +
                   si * center_anchor_grav[2])
        z_world = (source_pose.z_m - si * center_anchor_grav[0] +
                   c * center_anchor_grav[2])
        pose = Pose2D(float(x_world), float(z_world),
                      float(anchor_yaw + yaw_relative), True)
        if self._last_good_s is not None:
            dt = max(timestamp_s - self._last_good_s, 1e-3)
            speed = math.hypot(pose.x_m - self.pose.x_m,
                               pose.z_m - self.pose.z_m) / dt
            yaw_delta = (pose.yaw_rad - self.pose.yaw_rad + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = abs(math.degrees(yaw_delta)) / dt
            if speed > s["tracking_max_speed_m_s"] or yaw_rate > s["tracking_max_yaw_deg_s"]:
                self.rejection = "implausible_motion"
                return None
        return pose, len(inliers)

    def update(self, result, floor, timestamp_s):
        self.reset_map = False
        self.inliers = 0
        self.rejection = None
        floor_bridge = False
        if floor.usable:
            self._last_usable_floor = floor
            self._last_usable_floor_s = timestamp_s
        elif (self._anchor is not None and self._last_usable_floor is not None and
              timestamp_s - self._last_usable_floor_s <=
              self.settings["tracking_floor_bridge_s"]):
            # Visual motion can keep the pose alive briefly while floor fitting
            # fails. ObstacleMap still refuses to integrate without a fresh floor.
            floor = self._last_usable_floor
            floor_bridge = True
        else:
            return self._lost(timestamp_s, "floor_lost")
        if self._anchor is None:
            origin = Pose2D()
            if not self._set_anchor(result, floor, origin):
                return self._lost(timestamp_s, "few_features")
            self._set_previous(result, floor, origin)
            if self._ever_initialized and self._lost_since_s is not None:
                self.reset_map = True
                self.reset_count += 1
            self._ever_initialized = True
            self.pose = origin
            self.tracking_ok = True
            self._last_good_s = timestamp_s
            self._lost_since_s = None
            self.method = "initialized"
            return origin
        method = "lk"
        solved = self._solve(self._lk_correspondences(result), result, floor,
                             timestamp_s, self._previous_floor, self._previous_pose)
        if solved is None:
            method = "orb"
            solved = self._solve(self._orb_correspondences(result), result, floor, timestamp_s)
        if solved is None:
            return self._lost(timestamp_s, "tracking_lost")
        pose, self.inliers = solved
        self.pose = pose
        self.tracking_ok = True
        self._last_good_s = timestamp_s
        self._lost_since_s = None
        self.method = method + "_floor_bridge" if floor_bridge else method
        self._set_previous(result, floor, pose)
        anchor_distance = math.hypot(pose.x_m - self._anchor_pose.x_m,
                                     pose.z_m - self._anchor_pose.z_m)
        anchor_yaw_delta = abs(math.degrees(
            (pose.yaw_rad - self._anchor_pose.yaw_rad + math.pi) % (2 * math.pi) - math.pi))
        if not floor_bridge and floor.fresh and (
                anchor_distance >= self.settings["keyframe_translation_m"] or
                anchor_yaw_delta >= self.settings["keyframe_yaw_deg"]):
            self._set_anchor(result, floor, pose)
        return pose

    def _lost(self, timestamp_s, reason):
        if self._lost_since_s is None:
            self._lost_since_s = timestamp_s
        self.tracking_ok = False
        self.method = reason
        if timestamp_s - self._lost_since_s >= self.settings["tracking_reset_after_s"]:
            self._anchor = None
            self._previous = None
        self.pose = Pose2D(self.pose.x_m, self.pose.z_m, self.pose.yaw_rad, False)
        return self.pose
