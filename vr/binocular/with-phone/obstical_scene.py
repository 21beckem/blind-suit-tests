#!/usr/bin/env python3
"""
obstacle_scene.py - plane finder + obstacle avoidance view, built on stereo_3d.py.

    python obstacle_scene.py left.jpg right.jpg -c stereo_config.json

(keep this file in the same folder as stereo_3d.py)

Pipeline
  1. stereo_3d builds a metric point cloud from the two photos.
  2. Planes are found one after another with RANSAC. Each plane's inliers are
     rasterised on the plane, small gaps / holes are filled (e.g. floor hidden
     behind objects or textureless patches) and turned into a mesh.
  3. One plane is designated the FLOOR: a large plane that is roughly
     horizontal (relative to the camera) and has most of the scene above it.
     Big vertical planes are treated as walls.
  4. Everything above the floor (and not floor/wall) is clustered into obstacles.
  5. Box-like obstacles get an ORANGE box aligned to the floor and rotated
     to match the object's heading (min-area rectangle, not an axis-aligned box).
     Obscure shapes get an ORANGE low-poly mesh (decimated convex hull that
     reaches down to the floor).
  6. Everything is shown in an Open3D window you can orbit / pan / zoom.

Viewer keys:  X = points on/off   G = planes on/off   V = obstacles on/off
Mouse: left-drag orbit, ctrl+left-drag (or middle-drag) pan, wheel zoom, +/- point size.

Install:  pip install opencv-python numpy open3d
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stereo_3d as s3d  # noqa: E402

OBS_DEFAULTS = {
    "disparity_sigma_px": 0.6,        # expected matching noise, drives depth-dependent tolerances
    "plane_thresh_m": 0.03,           # RANSAC inlier distance
    "max_planes": 5,
    "min_plane_points": 3000,
    "min_plane_area_m2": 0.25,
    "plane_cell_m": 0.04,             # raster cell size used to build plane meshes
    "gap_fill_m": 0.30,               # gaps narrower than this are closed
    "max_hole_area_m2": 1.5,          # enclosed holes smaller than this are filled
    "plane_render_offset_m": 0.04,    # plane mesh is drawn this far behind the points
    "floor_max_tilt_deg": 45.0,       # floor normal must be within this of camera 'up'
    "wall_max_cos": 0.35,             # |n_plane . n_floor| below this => wall
    "obstacle_clearance_m": 0.05,     # ignore stuff lower than this above the floor
    "max_obstacle_height_m": 2.0,
    "voxel_m": 0.006,
    "cluster_eps_m": 0.05,
    "min_cluster_points": 200,
    "core_cell_m": 0.04,              # density filter: sparse streaks (stereo smear) are ignored
    "core_min_frac": 0.08,
    "box_fill_ratio": 0.85,           # top-face hull area / rectangle area to count as box (disc = 0.79)
    "ransac_seed": 0,
    "box_top_flat_fraction": 0.12,    # share of points near the top height (flat top)
    "mesh_triangles": 28,
    "edge_radius_m": 0.006,
}

ORANGE = (1.0, 0.5, 0.0)
PLANE_COLORS = {"floor": (0.25, 0.75, 0.35), "wall": (0.35, 0.55, 0.95), "surface": (0.70, 0.50, 0.90)}


# --------------------------------------------------------------------------- geometry helpers
def sigma_z(z, f, B_m, sd_px):
    """1-sigma depth noise (m) of a stereo rig at depth z (m)."""
    return (z * z) / (f * B_m) * sd_px


def orthonormal_basis(n):
    """e1, e2 with e1 x e2 = n."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def fit_plane(P):
    c = P.mean(0)
    Q = P - c
    w, v = np.linalg.eigh(Q.T @ Q)
    n = v[:, 0]
    return n, -float(n @ c)


# --------------------------------------------------------------------------- planes
def plane_patch(P, n, cfg):
    """Raster of the plane's inliers with gaps filled. None if absurdly large."""
    e1, e2 = orthonormal_basis(n)
    u, v = P @ e1, P @ e2
    cell = float(cfg["plane_cell_m"])
    u0, v0 = float(u.min()), float(v.min())
    iu = ((u - u0) / cell).astype(np.int64)
    iv = ((v - v0) / cell).astype(np.int64)
    W, H = int(iu.max()) + 1, int(iv.max()) + 1
    if W * H > 2_000_000:
        return None
    cnt = np.bincount(iv * W + iu, minlength=W * H).reshape(H, W)
    mask = (cnt >= 3).astype(np.uint8)

    k = max(3, int(round(cfg["gap_fill_m"] / cell)) | 1)
    m = np.pad(mask, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inv = (m == 0).astype(np.uint8)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    Hp, Wp = m.shape
    max_cells = cfg["max_hole_area_m2"] / cell ** 2
    fill = np.zeros(nlab, bool)
    for i in range(1, nlab):
        x, y, w, h, a = stats[i]
        if x > 0 and y > 0 and x + w < Wp and y + h < Hp and a <= max_cells:
            fill[i] = True
    m[fill[lab]] = 1
    m = m[k:-k, k:-k]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return {"mask": m, "u0": u0, "v0": v0, "cell": cell, "e1": e1, "e2": e2,
            "area": float(m.sum()) * cell * cell}


def patch_mesh(o3d, patch, n, d, offset, color):
    mask, cell = patch["mask"], patch["cell"]
    H, W = mask.shape
    cy, cx = np.nonzero(mask)
    if len(cy) == 0:
        return None
    c00 = cy * (W + 1) + cx
    c10, c01 = c00 + 1, c00 + (W + 1)
    c11 = c01 + 1
    tris = np.concatenate([np.stack([c00, c10, c11], 1), np.stack([c00, c11, c01], 1)])
    used = np.unique(tris)
    remap = -np.ones((H + 1) * (W + 1), np.int64)
    remap[used] = np.arange(len(used))
    tris = remap[tris]
    vy, vx = np.divmod(used, W + 1)
    uu, vv = patch["u0"] + vx * cell, patch["v0"] + vy * cell
    verts = np.outer(uu, patch["e1"]) + np.outer(vv, patch["e2"]) + (-d - offset) * n
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(tris))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def find_planes(o3d, pts, cfg, f, B):
    o3d.utility.random.seed(int(cfg["ransac_seed"]))
    N = len(pts)
    z = -pts[:, 2]
    thr_pt = np.maximum(cfg["plane_thresh_m"], 2.0 * sigma_z(z, f, B, cfg["disparity_sigma_px"]))
    remaining = np.ones(N, bool)
    rng = np.random.default_rng(0)
    planes, tries = [], 0
    min_pts = int(cfg["min_plane_points"])
    while len(planes) < cfg["max_planes"] and tries < cfg["max_planes"] + 4:
        tries += 1
        idx = np.nonzero(remaining)[0]
        if len(idx) < min_pts:
            break
        samp = idx if len(idx) <= 200_000 else rng.choice(idx, 200_000, replace=False)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts[samp])
        model, inl = pc.segment_plane(cfg["plane_thresh_m"], 3, 2000)
        if len(inl) < min_pts // 3:
            break
        n = np.array(model[:3])
        s = np.linalg.norm(n)
        n, d = n / s, float(model[3]) / s
        P = pts[idx]
        for _ in range(2):  # grow to all remaining points, then refit
            member = np.abs(P @ n + d) < thr_pt[idx]
            if member.sum() < 10:
                break
            n, d = fit_plane(P[member])
        member = np.abs(P @ n + d) < thr_pt[idx]
        mi = idx[member]
        remaining[mi] = False
        if len(mi) < min_pts:
            continue
        if d < 0:  # face the camera (camera is at the origin)
            n, d = -n, -d
        patch = plane_patch(pts[mi], n, cfg)
        if patch is None or patch["area"] < cfg["min_plane_area_m2"]:
            continue
        planes.append({"n": n, "d": d, "idx": mi, "patch": patch, "area": patch["area"]})
    return planes


def pick_floor(pts, planes, cfg):
    up = np.array([0.0, 1.0, 0.0])
    cos_min = math.cos(math.radians(cfg["floor_max_tilt_deg"]))
    best, best_score = None, 0.0
    for p in planes:
        c = float(p["n"] @ up)
        others = np.ones(len(pts), bool)
        others[p["idx"]] = False
        frac = float(((pts[others] @ p["n"] + p["d"]) > -0.05).mean()) if others.any() else 1.0
        score = len(p["idx"]) * c * frac if c >= cos_min else 0.0
        if score > best_score:
            best, best_score = p, score
    if best is None and planes:
        print("WARNING: no roughly horizontal plane found; using the largest plane as floor.",
              file=sys.stderr)
        best = max(planes, key=lambda p: len(p["idx"]))
    return best


# --------------------------------------------------------------------------- obstacles
def uv_to_xyz(u, v, h, e1, e2, n, d):
    return u * e1 + v * e2 + (h - d) * n


def cylinder_between(o3d, a, b, radius):
    v = b - a
    L = float(np.linalg.norm(v))
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, max(L, 1e-6), resolution=8)
    z = np.array([0, 0, 1.0])
    dirn = v / max(L, 1e-9)
    axis = np.cross(z, dirn)
    s, c = float(np.linalg.norm(axis)), float(z @ dirn)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * math.atan2(s, c))
    cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((a + b) / 2)
    return cyl


def edges_mesh(o3d, V, E, radius, color):
    out = o3d.geometry.TriangleMesh()
    for a, b in E:
        out += cylinder_between(o3d, V[a], V[b], radius)
    out.paint_uniform_color(color)
    out.compute_vertex_normals()
    return out


def lowpoly_hull(o3d, P, floor, bottom, cfg):
    n, d = floor["n"], floor["d"]
    allp = P
    if bottom == 0.0:  # object rests on the floor: extend the hull down to it
        h = P @ n + d
        allp = np.vstack([P, P - np.outer(h, n)])
    if len(allp) > 6000:
        allp = allp[np.random.default_rng(0).choice(len(allp), 6000, replace=False)]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(allp)
    hull, _ = pc.compute_convex_hull()
    hull = hull.simplify_quadric_decimation(int(cfg["mesh_triangles"]))
    hull.remove_degenerate_triangles()
    hull.remove_unreferenced_vertices()
    return hull


def density_core(P, e1, e2, cfg):
    """Mask of points lying in dense top-down cells (drops thin smear/flying-pixel streaks)."""
    cell = cfg["core_cell_m"]
    u, v = P @ e1, P @ e2
    iu = ((u - u.min()) / cell).astype(np.int64)
    iv = ((v - v.min()) / cell).astype(np.int64)
    W, H = int(iu.max()) + 1, int(iv.max()) + 1
    cnt = np.bincount(iv * W + iu, minlength=W * H).reshape(H, W)
    thr = max(2.0, cfg["core_min_frac"] * float(np.percentile(cnt[cnt > 0], 90)))
    keep = cnt[iv, iu] >= thr
    return keep if keep.mean() >= 0.3 else np.ones(len(P), bool)


def fit_obstacle(o3d, P, floor, cfg, force_box=False):
    n, d = floor["n"], floor["d"]
    e1, e2 = orthonormal_basis(n)
    P = P[density_core(P, e1, e2, cfg)]
    h = P @ n + d
    xy = np.stack([P @ e1, P @ e2], 1).astype(np.float32)

    def rect_fit(pts2):
        rd = cv2.minAreaRect(pts2)
        area = max(rd[1][0] * rd[1][1], 1e-9)
        return rd, float(cv2.contourArea(cv2.convexHull(pts2))) / area

    rect_def, ratio = rect_fit(xy)  # whole footprint (sides may be occluded / smeared)
    bottom = float(h.min())
    top = float(np.percentile(h, 98))
    if bottom < 0.15:
        bottom = 0.0

    def top_face_mask():
        """Points of a roughly horizontal top face, found by fitting its own plane so a
        slightly tilted floor estimate doesn't slice the face diagonally."""
        sel = h > top - 0.10
        m = h > top - 0.05
        for _ in range(3):
            if m.sum() < 30:
                return None
            pn, pd = fit_plane(P[m])
            if pn @ n < 0:
                pn, pd = -pn, -pd
            if pn @ n < 0.9:
                return None
            m = sel & (np.abs(P @ pn + pd) < 0.025)
        return m

    topmask = top_face_mask()
    flat = float(topmask.mean()) if topmask is not None else 0.0
    boxy = False
    if topmask is not None and flat >= cfg["box_top_flat_fraction"] and topmask.sum() >= 40:
        # A flat top face is seen in full from above: a rectangle fills its min-area
        # rectangle (~1.0), a disc only pi/4 = 0.79. Use it for the test and the box size.
        top_def, top_ratio = rect_fit(xy[topmask])
        full_area = max(rect_def[1][0] * rect_def[1][1], 1e-9)
        top_area = top_def[1][0] * top_def[1][1]
        # the flat top must be most of the object's footprint (a sphere's small cap is not)
        if top_ratio >= cfg["box_fill_ratio"] and top_area >= 0.5 * full_area:
            boxy, rect_def, ratio = True, top_def, top_ratio
    elif ratio >= cfg["box_fill_ratio"]:
        boxy = True  # top not visible (tall object): fall back to the whole footprint
    (cu, cv_), (rw, rh), _ = rect_def
    rect = cv2.boxPoints(rect_def).astype(np.float64)  # 4x2, ordered around the rectangle

    corners = np.array([uv_to_xyz(u, v, hh, e1, e2, n, d) for hh in (bottom, top) for u, v in rect])
    box_edges = [(i, (i + 1) % 4) for i in range(4)] + [(4 + i, 4 + (i + 1) % 4) for i in range(4)] \
        + [(i, i + 4) for i in range(4)]

    ea, eb = rect[1] - rect[0], rect[2] - rect[1]
    long_uv = ea if np.linalg.norm(ea) >= np.linalg.norm(eb) else eb
    long3 = long_uv[0] * e1 + long_uv[1] * e2
    ref = np.array([1.0, 0, 0])
    ref = ref - (ref @ n) * n
    ref /= np.linalg.norm(ref)
    yaw = math.degrees(math.atan2(float(n @ np.cross(ref, long3)), float(ref @ long3)))
    yaw = (yaw + 90.0) % 180.0 - 90.0

    foot = -d * n  # camera's projection on the floor
    rng_m = max(0.0, -cv2.pointPolygonTest(rect.astype(np.float32).reshape(-1, 1, 2),
                                           (float(foot @ e1), float(foot @ e2)), True))
    centre = uv_to_xyz(cu, cv_, (bottom + top) / 2, e1, e2, n, d)
    info = {"size_m": [round(max(rw, rh), 3), round(min(rw, rh), 3), round(top - bottom, 3)],
            "yaw_deg": round(yaw, 1), "center_m": [round(float(x), 3) for x in centre],
            "range_m": round(float(rng_m), 2), "footprint_fill": round(ratio, 2),
            "n_points": int(len(P))}

    ob = {"info": info, "box_corners": corners, "box_edges": box_edges, "mesh": None}
    if boxy or force_box:
        ob["kind"] = "box"
        return ob
    try:
        hull = lowpoly_hull(o3d, P, floor, bottom, cfg)
        E = np.asarray(o3d.geometry.LineSet.create_from_triangle_mesh(hull).lines)
        if len(E) < 6:
            raise ValueError("degenerate hull")
        ob["kind"] = "mesh"
        ob["mesh"] = hull
        ob["mesh_edges"] = E
    except Exception as exc:  # fall back to the box
        print(f"  (low-poly mesh failed: {exc}; using box)", file=sys.stderr)
        ob["kind"] = "box"
    return ob


def find_obstacles(o3d, pts, floor, exclude, cfg, f, B, boxes_for_all=False):
    n, d = floor["n"], floor["d"]
    h = pts @ n + d
    z = -pts[:, 2]
    noise = 2.0 * sigma_z(z, f, B, cfg["disparity_sigma_px"])
    cand = (h > cfg["obstacle_clearance_m"] + noise) & (h < cfg["max_obstacle_height_m"]) & ~exclude
    if cand.sum() < cfg["min_cluster_points"]:
        return []
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts[cand])
    pc = pc.voxel_down_sample(cfg["voxel_m"])
    if len(pc.points) > 30:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    Pd = np.asarray(pc.points)
    labels = np.array(pc.cluster_dbscan(eps=cfg["cluster_eps_m"], min_points=8))
    obstacles = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        Pc = Pd[labels == lab]
        if len(Pc) < cfg["min_cluster_points"]:
            continue
        obstacles.append(fit_obstacle(o3d, Pc, floor, cfg, force_box=False))
    obstacles.sort(key=lambda o: o["info"]["range_m"])
    for i, o in enumerate(obstacles):
        o["id"] = i + 1
    return obstacles


# --------------------------------------------------------------------------- viewing
def build_groups(o3d, pts, cols, planes, floor, walls, obstacles, cfg, solid_meshes, boxes_for_all):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    groups = {"points": [pcd], "planes": [], "obstacles": [],
              "frame": [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]}
    for p in planes:
        if p["kind"] == "facet":  # small tilted patch on an object: not drawn as a plane
            continue
        m = patch_mesh(o3d, p["patch"], p["n"], p["d"], cfg["plane_render_offset_m"],
                       PLANE_COLORS[p["kind"]])
        if m is not None:
            groups["planes"].append(m)
    r = cfg["edge_radius_m"]
    for o in obstacles:
        if o["kind"] == "box" or boxes_for_all:
            groups["obstacles"].append(edges_mesh(o3d, o["box_corners"], o["box_edges"], r, ORANGE))
        if o["kind"] == "mesh":
            groups["obstacles"].append(edges_mesh(o3d, np.asarray(o["mesh"].vertices),
                                                  o["mesh_edges"], r * 0.7, ORANGE))
            if solid_meshes:
                solid = o3d.geometry.TriangleMesh(o["mesh"])
                solid.paint_uniform_color((1.0, 0.55, 0.1))
                solid.compute_vertex_normals()
                groups["obstacles"].append(solid)
    return groups


def setup_view(vis, groups, point_size, center, eye_dir=(0.35, 0.35, 1.0)):
    for geoms in groups.values():
        for g in geoms:
            vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True
    opt.background_color = np.array([0.08, 0.08, 0.10])
    ctr = vis.get_view_control()
    front = np.array(eye_dir, dtype=float)
    ctr.set_front((front / np.linalg.norm(front)).tolist())
    ctr.set_up([0.0, 1.0, 0.0])
    ctr.set_lookat(center.tolist())
    ctr.set_zoom(0.45)


def show_scene(o3d, groups, point_size, center):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Obstacle scene - X points | G planes | V obstacles", 1280, 800)
    setup_view(vis, groups, point_size, center)

    def toggler(name):
        state = {"on": True}

        def cb(v):
            for g in groups[name]:
                (v.remove_geometry if state["on"] else v.add_geometry)(g, reset_bounding_box=False)
            state["on"] = not state["on"]
            return False
        return cb

    for key, name in (("X", "points"), ("G", "planes"), ("V", "obstacles")):
        vis.register_key_callback(ord(key), toggler(name))
    vis.run()
    vis.destroy_window()


def render_png(o3d, groups, path, center, point_size=3.0, size=(1400, 900), eye_dir=(0.9, 0.7, 1.0)):
    """Snapshot through the same OpenGL path as the viewer (needs a display / GL context)."""
    vis = o3d.visualization.Visualizer()
    vis.create_window("snapshot", size[0], size[1], visible=False)
    setup_view(vis, groups, point_size, center, eye_dir)
    for _ in range(3):
        vis.poll_events()
        vis.update_renderer()
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()
    print(f"Saved snapshot {path}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Plane finder + obstacle boxes from a photo pair.")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("-c", "--config", default="stereo_config.json")
    ap.add_argument("--no-view", action="store_true", help="don't open the 3D window")
    ap.add_argument("--screenshot", metavar="PNG", help="save a headless snapshot of the scene")
    ap.add_argument("--save-json", metavar="FILE", help="save floor plane + obstacles as JSON")
    ap.add_argument("--snapshot-view", choices=["iso", "top"], default="iso",
                    help="camera angle used for --screenshot")
    ap.add_argument("--debug-dir", help="save rectified pair + disparity previews here")
    ap.add_argument("--boxes-for-all", action="store_true",
                    help="also draw the oriented box around mesh-type obstacles")
    ap.add_argument("--solid-meshes", action="store_true", help="fill the low-poly meshes")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("This script needs Open3D:  pip install open3d")

    cfg = s3d.load_config(args.config)
    for k, v in OBS_DEFAULTS.items():
        cfg.setdefault(k, v)

    img_l = s3d.load_resized(args.left, cfg["max_image_side_px"])
    img_r = s3d.load_resized(args.right, cfg["max_image_side_px"])
    if img_r.shape[:2] != img_l.shape[:2]:
        img_r = cv2.resize(img_r, (img_l.shape[1], img_l.shape[0]))
    h, w = img_l.shape[:2]
    K = s3d.make_K(w, h, cfg)
    pts_mm, cols, info = s3d.reconstruct(img_l, img_r, K, cfg, args.debug_dir)
    pts = s3d.to_view_coords(pts_mm)  # metres, x right, y up, camera looks down -z
    f, B = info["f_rect"], cfg["baseline_mm"] / 1000.0
    print(f"{len(pts)} points reconstructed")

    planes = find_planes(o3d, pts, cfg, f, B)
    if not planes:
        sys.exit("No planes found. Try a lower min_plane_area_m2 / min_plane_points in the config.")
    floor = pick_floor(pts, planes, cfg)
    nf = floor["n"]
    for p in planes:
        c = abs(float(p["n"] @ nf))
        if p is floor:
            p["kind"] = "floor"
        elif c < cfg["wall_max_cos"]:
            p["kind"] = "wall"       # vertical relative to the floor
        elif p["area"] >= 0.6:
            p["kind"] = "surface"    # table top / shelf / ramp
        else:
            p["kind"] = "facet"      # small tilted patch of an object (stays an obstacle)
    walls = [p for p in planes if p["kind"] == "wall"]

    print("\nPLANES")
    for p in planes:
        print(f"  {p['kind'].upper():7s} normal=({p['n'][0]:+.2f},{p['n'][1]:+.2f},{p['n'][2]:+.2f}) "
              f"area={p['area']:.2f} m2  points={len(p['idx'])}")
    print(f"  camera height above floor: {floor['d']:.2f} m, "
          f"floor tilt vs camera up: {math.degrees(math.acos(min(1, float(nf[1])))):.1f} deg")

    exclude = np.zeros(len(pts), bool)
    exclude[floor["idx"]] = True
    for p in walls:
        exclude[p["idx"]] = True
    obstacles = find_obstacles(o3d, pts, floor, exclude, cfg, f, B, args.boxes_for_all)

    print("\nOBSTACLES (nearest first)  size = long x short x height, yaw vs camera-right axis")
    if not obstacles:
        print("  none found")
    for o in obstacles:
        i = o["info"]
        s = i["size_m"]
        print(f"  #{o['id']} {o['kind']:4s} size={s[0]:.2f}x{s[1]:.2f}x{s[2]:.2f} m  yaw={i['yaw_deg']:+.0f} deg  "
              f"range={i['range_m']:.2f} m  center=({i['center_m'][0]:+.2f},{i['center_m'][1]:+.2f},"
              f"{i['center_m'][2]:+.2f})  fill={i['footprint_fill']}")

    if args.save_json:
        out = {"floor": {"normal": nf.tolist(), "d": floor["d"]},
               "obstacles": [{"id": o["id"], "kind": o["kind"], **o["info"],
                              "box_corners": o["box_corners"].tolist(),
                              **({"mesh_vertices": np.asarray(o["mesh"].vertices).tolist(),
                                  "mesh_triangles": np.asarray(o["mesh"].triangles).tolist()}
                                 if o["mesh"] is not None else {})} for o in obstacles]}
        with open(args.save_json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"Saved {args.save_json}")

    groups = build_groups(o3d, pts, cols, planes, floor, walls, obstacles, cfg,
                          args.solid_meshes, args.boxes_for_all)
    center = np.median(pts, axis=0)
    if args.screenshot:
        render_png(o3d, groups, args.screenshot, center,
                   eye_dir=(0.9, 0.7, 1.0) if args.snapshot_view == "iso" else (0.0, 1.0, 0.0001))
    if not args.no_view:
        print("\nViewer: X = points, G = planes, V = obstacles | ctrl+drag = pan")
        show_scene(o3d, groups, cfg["point_size"], center)


if __name__ == "__main__":
    main()