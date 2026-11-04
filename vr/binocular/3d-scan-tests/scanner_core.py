"""Camera tracking (keyframe RGB-D odometry) + TSDF fusion. Uses Open3D, CPU only."""
import os
import time
from collections import deque

import numpy as np
import open3d as o3d


_TSDF_OK = None


def tsdf_works():
    """Some Open3D builds return an empty ScalableTSDFVolume. Detect that once at startup."""
    global _TSDF_OK
    if _TSDF_OK is None:
        try:
            w, h = 160, 120
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, 130.0, 130.0, w / 2 - 0.5, h / 2 - 0.5)
            yy, xx = np.mgrid[0:h, 0:w]
            d = (1.513 - 0.3 * np.exp(-(((xx - w / 2) / 25.0) ** 2 + ((yy - h / 2) / 25.0) ** 2))).astype(np.float32)
            col = o3d.geometry.Image(np.full((h, w, 3), 128, np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                col, o3d.geometry.Image(d), 1.0, 3.5, False)
            vol = o3d.pipelines.integration.ScalableTSDFVolume(
                0.02, 0.08, o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            vol.integrate(rgbd, intr, np.eye(4))
            _TSDF_OK = len(vol.extract_point_cloud().points) > 0
        except Exception:  # noqa: BLE001
            _TSDF_OK = False
    return _TSDF_OK


def rot_angle_deg(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))


class ScanSession:
    """Feed it rectified left gray images + metric depth; it tracks the camera and fuses a 3D map.

    Pose convention: T_world_cam, OpenCV camera axes (x right, y down, z forward).
    World origin = camera pose of the first frame.
    """

    def __init__(self, fx, fy, cx, cy, width, height, cfg, min_depth=0.2, max_depth=4.0, fusion=None):
        self.cfg = cfg
        # "tsdf" (best quality) or "points" (voxel-averaged cloud + Poisson mesh; always works)
        self.fusion = fusion or ("tsdf" if tsdf_works() else "points")
        self.w, self.h = int(width), int(height)
        self.intr = o3d.camera.PinholeCameraIntrinsic(self.w, self.h, fx, fy, cx, cy)
        opt = o3d.pipelines.odometry.OdometryOption()
        opt.iteration_number_per_pyramid_level = o3d.utility.IntVector([20, 10, 5])
        opt.depth_diff_max = float(cfg["odo_depth_diff"])
        opt.depth_min = float(min_depth)
        opt.depth_max = float(min(max_depth, cfg["depth_trunc"]))
        self.opt = opt
        if str(cfg.get("odo_term", "hybrid")) == "color":
            self.jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm()
        else:
            self.jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        if self.fusion == "tsdf":
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(self.cfg["voxel"]), sdf_trunc=float(self.cfg["sdf_trunc"]),
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            self.volume = None
            self.cloud = o3d.geometry.PointCloud()
            self._since_merge = 0
        self.keyframes = deque(maxlen=int(self.cfg["max_keyframes"]))   # (rgbd_odo, T_wk)
        self.anchors = []                                               # sparse long-term keyframes
        self.keyframe_poses = []
        self.pose = np.eye(4)
        self.last_rel = np.eye(4)
        self.state = "INIT"
        self.lost_frames = 0
        self.n_integrated = 0
        self.dirty = False

    # ------------------------------------------------------------------ helpers
    def _rgbd(self, gray, depth, intensity):
        color = o3d.geometry.Image(np.ascontiguousarray(np.dstack([gray, gray, gray])))
        d = o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, d, depth_scale=1.0, depth_trunc=float(self.cfg["depth_trunc"]),
            convert_rgb_to_intensity=intensity)

    def _odo(self, src, tgt, init):
        """Returns (ok, T_tgt_src). Maps points in src camera frame into tgt camera frame."""
        try:
            ok, T, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
                src, tgt, self.intr, np.asarray(init, dtype=np.float64), self.jac, self.opt)
        except RuntimeError:
            return False, None
        if not ok or not np.all(np.isfinite(T)):
            return False, None
        return True, T

    def _plausible(self, T_wc):
        k = 1.0 + min(self.lost_frames, 40) / 4.0   # allow bigger jumps the longer we've been lost
        dt = float(np.linalg.norm(T_wc[:3, 3] - self.pose[:3, 3]))
        da = rot_angle_deg((np.linalg.inv(self.pose) @ T_wc)[:3, :3])
        return dt <= self.cfg["max_step"] * k and da <= min(180.0, self.cfg["max_rot_deg"] * k)

    def _integrate(self, rgbd, T_wc):
        extr = np.linalg.inv(T_wc)
        if self.fusion == "tsdf":
            self.volume.integrate(rgbd, self.intr, extr)
            return
        kf = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intr, extr)
        kf = kf.voxel_down_sample(float(self.cfg["voxel"]))
        self.cloud += kf
        self._since_merge += 1
        if self._since_merge >= 6:           # merge overlapping keyframes (averages points per voxel)
            self.cloud = self.cloud.voxel_down_sample(float(self.cfg["voxel"]))
            self._since_merge = 0

    def _add_keyframe(self, gray, depth, odo, T_wc):
        self._integrate(self._rgbd(gray, depth, False), T_wc)
        self.keyframes.append((odo, T_wc.copy()))
        self.keyframe_poses.append(T_wc.copy())
        self.n_integrated += 1
        if self.n_integrated % 10 == 0:
            self.anchors.append((odo, T_wc.copy()))
            if len(self.anchors) > 30:
                self.anchors.pop(1)
        self.last_rel = np.eye(4)
        self.dirty = True

    def _relocalize(self, cur_odo):
        cands = list(reversed(self.keyframes))
        near = sorted(self.anchors, key=lambda a: np.linalg.norm(a[1][:3, 3] - self.pose[:3, 3]))[:6]
        cands += near
        for odo_k, T_wk in cands:
            init = np.linalg.inv(T_wk) @ self.pose
            ok, T = self._odo(cur_odo, odo_k, init)
            if ok:
                T_wc = T_wk @ T
                if self._plausible(T_wc):
                    return True, T_wc
        return False, None

    # ------------------------------------------------------------------ main entry
    def process(self, gray, depth):
        """gray: uint8 HxW, depth: float32 HxW metres (0 = invalid). Returns info dict."""
        valid = float((depth > 0).mean())
        info = {"valid": valid, "integrated": False}
        if valid < self.cfg["min_valid_ratio"]:
            if self.n_integrated:
                self.state = "NO DEPTH"
            info["state"] = self.state
            return info

        cur_odo = self._rgbd(gray, depth, True)
        if not self.keyframes:
            self._add_keyframe(gray, depth, cur_odo, np.eye(4))
            self.state = "TRACKING"
            info.update(state=self.state, integrated=True)
            return info

        key_odo, T_wk = self.keyframes[-1]
        ok, T_rel = self._odo(cur_odo, key_odo, self.last_rel)
        T_wc = None
        if ok:
            T_wc = T_wk @ T_rel
            ok = self._plausible(T_wc)
        if not ok and self.lost_frames % 3 == 0:
            ok, T_wc = self._relocalize(cur_odo)

        if ok:
            self.lost_frames = 0
            self.state = "TRACKING"
            self.pose = T_wc
            self.last_rel = np.linalg.inv(T_wk) @ T_wc   # relative to the newest keyframe
            moved = (np.linalg.norm(self.last_rel[:3, 3]) > self.cfg["key_trans"] or
                     rot_angle_deg(self.last_rel[:3, :3]) > self.cfg["key_rot_deg"])
            if moved:
                self._add_keyframe(gray, depth, cur_odo, T_wc)
                info["integrated"] = True
        else:
            self.lost_frames += 1
            self.state = "LOST"
        info["state"] = self.state
        return info

    # ------------------------------------------------------------------ outputs
    def extract_cloud(self):
        pcd = self.volume.extract_point_cloud() if self.fusion == "tsdf" else self.cloud
        self.dirty = False
        return (np.asarray(pcd.points, dtype=np.float32).copy(),
                np.asarray(pcd.colors, dtype=np.float32).copy())

    def _final_mesh_and_cloud(self, log):
        """Returns (mesh, pcd) in OpenCV world coordinates."""
        if self.fusion == "tsdf":
            mesh = self.volume.extract_triangle_mesh()
            pcd = self.volume.extract_point_cloud()
            if len(mesh.vertices) == 0:
                return None, None
            mesh.compute_vertex_normals()
            return mesh, pcd
        pcd = self.cloud.voxel_down_sample(float(self.cfg["voxel"]))
        if len(pcd.points) < 100:
            return None, None
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        r = float(self.cfg["voxel"]) * 3.0
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=30))
        centre = np.array(self.keyframe_poses)[:, :3, 3].mean(axis=0)
        pcd.orient_normals_towards_camera_location(centre)
        log("Building mesh (Poisson)...")
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
        mesh.compute_vertex_normals()
        return mesh, pcd

    def save(self, out_dir, log=print):
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        mesh, pcd = self._final_mesh_and_cloud(log)
        if mesh is None:
            log("Nothing to save yet (no geometry).")
            return None
        try:
            tri_cl, n_tri, _area = mesh.cluster_connected_triangles()
            tri_cl, n_tri = np.asarray(tri_cl), np.asarray(n_tri)
            mesh.remove_triangles_by_mask(n_tri[tri_cl] < self.cfg["cleanup_min_triangles"])
            mesh.remove_unreferenced_vertices()
        except Exception as e:  # noqa: BLE001
            log("(mesh cleanup skipped: %s)" % e)
        poses = np.array(self.keyframe_poses)
        if self.cfg.get("export_y_up", True):
            flip = np.diag([1.0, -1.0, -1.0, 1.0])
            mesh.transform(flip)
            pcd.transform(flip)
            poses = np.array([flip @ p for p in poses])
        base = os.path.join(out_dir, "scan_" + stamp)
        o3d.io.write_triangle_mesh(base + "_mesh.ply", mesh)
        o3d.io.write_point_cloud(base + "_points.ply", pcd)
        np.save(base + "_trajectory.npy", poses)
        log("Saved: %s_mesh.ply / _points.ply / _trajectory.npy" % base)
        return base
