"""Interactive Open3D display for generated scans."""

from __future__ import annotations

import numpy as np

from .models import ScanResult


def _id_colors(scan: ScanResult) -> np.ndarray:
    palette = np.asarray(
        [[0.62, 0.62, 0.62], [0.90, 0.30, 0.24], [0.25, 0.65, 0.90], [0.35, 0.78, 0.42], [0.95, 0.70, 0.20], [0.65, 0.40, 0.85]],
        dtype=np.float64,
    )
    colors = palette[np.mod(np.maximum(scan.object_ids, 0), len(palette))]
    colors[scan.object_ids < 0] = (1.0, 0.0, 1.0)
    return colors


def show_scan(scan: ScanResult, *, point_size: float = 2.0) -> None:
    """Open a viewer. C toggles natural and semantic point colors."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(scan.points)
    cloud.colors = o3d.utility.Vector3dVector(scan.colors)
    cloud.normals = o3d.utility.Vector3dVector(scan.normals)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Synthetic room scan — C colors, Q/Esc close", width=1280, height=800)
    vis.add_geometry(cloud)
    vis.add_geometry(axes)
    state = {"semantic": False}

    def toggle_colors(viewer):
        state["semantic"] = not state["semantic"]
        cloud.colors = o3d.utility.Vector3dVector(_id_colors(scan) if state["semantic"] else scan.colors)
        viewer.update_geometry(cloud)
        return False

    vis.register_key_callback(ord("C"), toggle_colors)
    options = vis.get_render_option()
    options.point_size = float(point_size)
    options.background_color = np.asarray([0.035, 0.04, 0.055])
    vis.run()
    vis.destroy_window()

