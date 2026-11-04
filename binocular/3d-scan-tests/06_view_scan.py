"""View a saved scan without starting the camera.

Examples:
    python 06_view_scan.py
    python 06_view_scan.py scans/scan_20261104_162212_points.ply
    python 06_view_scan.py scans/scan_20261104_162212 --geometry mesh

The newest point-cloud scan is selected when no path is supplied. Press F to fit the complete scan and Q
to quit. Saved scans use the export convention from config.json: Y up and the original camera looking -Z.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from common import SCAN_DIR


SUFFIXES = ("_points.ply", "_mesh.ply", "_trajectory.npy")


def scan_base(path):
    text = str(path)
    for suffix in SUFFIXES:
        if text.lower().endswith(suffix):
            return Path(text[:-len(suffix)])
    return Path(text)


def newest_scan(directory):
    candidates = list(Path(directory).glob("scan_*_points.ply"))
    if not candidates:
        raise FileNotFoundError("No *_points.ply scans found in %s" % directory)
    return scan_base(max(candidates, key=lambda p: p.stat().st_mtime))


def resolve_base(value):
    if value is None:
        return newest_scan(SCAN_DIR)
    path = Path(value).expanduser()
    if path.is_dir():
        return newest_scan(path)
    return scan_base(path)


def load_scan(base, mode):
    point_path = Path(str(base) + "_points.ply")
    mesh_path = Path(str(base) + "_mesh.ply")
    geometries = []
    loaded = []

    if mode in ("points", "both"):
        if not point_path.exists():
            raise FileNotFoundError(point_path)
        cloud = o3d.io.read_point_cloud(str(point_path))
        if len(cloud.points) == 0:
            raise ValueError("Point cloud is empty: %s" % point_path)
        geometries.append(cloud)
        loaded.append(("points", point_path, len(cloud.points)))

    if mode in ("mesh", "both"):
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh is empty: %s" % mesh_path)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        geometries.append(mesh)
        loaded.append(("mesh", mesh_path, len(mesh.triangles)))

    return geometries, loaded


def combined_bounds(geometries):
    lo = np.min([g.get_axis_aligned_bounding_box().min_bound for g in geometries], axis=0)
    hi = np.max([g.get_axis_aligned_bounding_box().max_bound for g in geometries], axis=0)
    return lo, hi


def trajectory_geometry(base):
    path = Path(str(base) + "_trajectory.npy")
    if not path.exists():
        return None, None
    poses = np.load(path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        return None, path
    points = poses[:, :3, 3]
    lines = np.column_stack([np.arange(len(points) - 1), np.arange(1, len(points))])
    traj = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points.astype(np.float64)),
        o3d.utility.Vector2iVector(lines.astype(np.int32)),
    )
    traj.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.25, 0.05], (len(lines), 1)))
    return traj, path


def fit_view(vis, geometries):
    lo, hi = combined_bounds(geometries)
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])       # exported scan lies in -Z
    ctr.set_up([0, 1, 0])          # exported scan is Y-up
    ctr.set_lookat(((lo + hi) * 0.5).tolist())
    ctr.set_zoom(0.72)


def main():
    ap = argparse.ArgumentParser(description="View a saved stereo scan (newest scan by default).")
    ap.add_argument("scan", nargs="?", help="scan base, *_points.ply, *_mesh.ply, or scans directory")
    ap.add_argument("--geometry", choices=["auto", "points", "mesh", "both"], default="auto",
                    help="geometry to load (default: infer from path, otherwise points)")
    ap.add_argument("--no-trajectory", action="store_true", help="hide the camera path")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--info", action="store_true", help="print scan bounds and exit without opening a window")
    args = ap.parse_args()

    try:
        base = resolve_base(args.scan)
        mode = args.geometry
        if mode == "auto":
            mode = "mesh" if args.scan and str(args.scan).lower().endswith("_mesh.ply") else "points"
        geometries, loaded = load_scan(base, mode)
    except (FileNotFoundError, ValueError) as exc:
        print("Error:", exc, file=sys.stderr)
        return 1

    lo, hi = combined_bounds(geometries)
    for kind, path, count in loaded:
        unit = "points" if kind == "points" else "triangles"
        print("Loaded %s: %s (%s %s)" % (kind, path, f"{count:,}", unit))
    print("Bounds: min %s | max %s | size %s metres" % (
        np.array2string(lo, precision=3), np.array2string(hi, precision=3),
        np.array2string(hi - lo, precision=3)))
    if args.info:
        return 0

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window("Saved scan  [F fit whole scan | Q quit]", 1280, 800):
        print("Error: could not create an Open3D window", file=sys.stderr)
        return 1
    for geometry in geometries:
        vis.add_geometry(geometry)

    if not args.no_trajectory:
        traj, traj_path = trajectory_geometry(base)
        if traj is not None:
            vis.add_geometry(traj)
            print("Loaded trajectory:", traj_path)

    extent = float(np.max(hi - lo))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.clip(extent * 0.06, 0.10, 0.50))
    vis.add_geometry(frame)
    opt = vis.get_render_option()
    opt.point_size = max(1.0, float(args.point_size))
    opt.background_color = np.array([0.04, 0.04, 0.06])

    def refit(_vis):
        fit_view(_vis, geometries)
        return False

    vis.register_key_callback(ord("F"), refit)
    fit_view(vis, geometries)
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
