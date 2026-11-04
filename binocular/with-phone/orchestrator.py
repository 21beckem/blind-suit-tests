#!/usr/bin/env python3
"""
stereo_scene.py - orchestrator: two photos -> point cloud -> floor / walls / obstacles -> 3D view.

    python stereo_scene.py left.jpg right.jpg -c stereo_config.json

It only wires things together:
  1. stereo_3d.py      photos -> 3D point cloud (+ pixel info for each point)
  2. obstacle_scene.py the point-cloud algorithm (knows nothing about images; works on any cloud)
  3. this file         the image-based object segmentation (it needs the pixel grid), the
                       report / JSON / snapshot output and the interactive viewer

Keep the three files in the same folder.  Install: pip install opencv-python numpy scipy open3d
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obstacle_scene as obs  # noqa: E402
import stereo_3d as s3d  # noqa: E402


def image_segments(pix, disp, shape, cfg):
    """Segment the candidate points in the IMAGE: neighbouring pixels belong to the same object
    when their disparity agrees. A depth jump (basket in front of a chair) separates objects,
    while thin things (chair legs) stay connected along their length."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    h, w = shape
    M = len(pix)
    G = -np.ones((h, w), np.int64)
    G[pix[:, 0], pix[:, 1]] = np.arange(M)
    D = np.full((h, w), np.nan, np.float32)
    D[pix[:, 0], pix[:, 1]] = disp
    tau = cfg["segment_disp_px"]
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = 0, h - dy
        x0, x1 = (0, w - dx) if dx >= 0 else (-dx, w)
        A = G[y0:y1, x0:x1]
        Bn = G[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        DA = D[y0:y1, x0:x1]
        DB = D[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        ok = (A >= 0) & (Bn >= 0) & (np.abs(DA - DB) <= tau)
        rows.append(A[ok])
        cols.append(Bn[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(M, M))
    _, lab = connected_components(g, directed=False)
    return lab


def main():
    ap = argparse.ArgumentParser(description="Plane finder + obstacle boxes from a photo pair.")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("-c", "--config", default="stereo_config.json")
    ap.add_argument("--no-view", action="store_true", help="don't open the 3D window")
    ap.add_argument("--screenshot", metavar="PNG", help="save a headless snapshot of the scene")
    ap.add_argument("--snapshot-view", choices=["iso", "top", "front"], default="iso",
                    help="camera angle used for --screenshot")
    ap.add_argument("--save-json", metavar="FILE", help="save floor, walls, surfaces + obstacles as JSON")
    ap.add_argument("--debug-dir", help="save rectified pair, disparity and cluster previews here")
    ap.add_argument("--boxes-for-all", action="store_true",
                    help="also draw the oriented box around mesh-type obstacles")
    ap.add_argument("--solid-meshes", action="store_true", help="fill the low-poly meshes")
    ap.add_argument("--verbose", action="store_true", help="print extra detail while grouping furniture")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("This script needs Open3D:  pip install open3d")

    # ---- 1. images -> point cloud
    scfg = s3d.load_config(args.config)
    cfg = obs.default_config(scfg)
    img_l = s3d.load_resized(args.left, cfg["max_image_side_px"])
    img_r = s3d.load_resized(args.right, cfg["max_image_side_px"])
    if img_r.shape[:2] != img_l.shape[:2]:
        img_r = cv2.resize(img_r, (img_l.shape[1], img_l.shape[0]))
    h, w = img_l.shape[:2]
    K = s3d.make_K(w, h, cfg)
    pts_mm, cols, info = s3d.reconstruct(img_l, img_r, K, cfg, args.debug_dir)
    pts = s3d.to_view_coords(pts_mm)  # metres, x right, y up, camera looks down -z
    print(f"{len(pts)} points reconstructed")

    # ---- 2. the algorithm (stereo depth-noise model; objects split using the image grid)
    sensor = {"f": info["f_rect"], "B": cfg["baseline_mm"] / 1000.0}
    segment_fn = lambda cand: image_segments(info["pix"][cand], info["disp"][cand], info["shape"], cfg)  # noqa: E731
    try:
        res = obs.analyze_point_cloud(o3d, pts, cfg, segment_fn=segment_fn, sensor=sensor, verbose=args.verbose)
    except RuntimeError as exc:
        sys.exit(str(exc))

    # ---- 3. output
    obs.print_report(res)
    if args.debug_dir:
        obs.debug_views(os.path.join(args.debug_dir, "clusters.png"), pts, cols, res)
    if args.save_json:
        with open(args.save_json, "w") as fh:
            json.dump(obs.result_to_json(res), fh, indent=1)
        print(f"Saved {args.save_json}")

    groups = obs.scene_groups(o3d, pts, cols, res, cfg, args.solid_meshes, args.boxes_for_all)
    center = np.median(pts, axis=0)
    if args.screenshot:
        eye = {"iso": (0.9, 0.7, 1.0), "top": (0.0, 1.0, 0.0001), "front": (0.0, 0.25, 1.0)}[args.snapshot_view]
        obs.render_png(o3d, groups, args.screenshot, center, eye_dir=eye)
    if not args.no_view:
        print("\nViewer: X = points, G = planes, V = obstacles | ctrl+drag = pan")
        obs.show_scene(o3d, groups, cfg["point_size"], center)


if __name__ == "__main__":
    main()