#!/usr/bin/env python3
"""
obstacle_scene.py - floor / wall / surface / obstacle finder for ANY point cloud.

This file is a library: it knows nothing about cameras or image files. Feed it points and
it returns the planes and obstacles. The scripts that produce the points (for example
stereo_scene.py, which builds them from two photos) call it:

    import open3d as o3d, obstacle_scene as obs
    cfg = obs.default_config({...your overrides...})
    result = obs.analyze_point_cloud(o3d, points_xyz, cfg)
    obs.print_report(result)
    groups = obs.scene_groups(o3d, points_xyz, colors, result, cfg)
    obs.show_scene(o3d, groups, cfg["point_size"], np.median(points_xyz, axis=0))

Point cloud conventions (metres)
  +y is UP, the SENSOR is at the origin (it is used as "the near side": planes are oriented
  to face it, and anything beyond the outermost wall is background). For a camera, x is
  right and the camera looks down -z. Other frames: rotate/translate your cloud first.
  Noise model: pass sensor={"f": focal_px, "B": baseline_m} for a stereo rig (depth noise grows
  with range^2), otherwise a constant cfg["noise_sigma_m"] is used.
  Optional: segment_fn(cand_mask) -> integer label per candidate point, to split objects
  using extra knowledge (the stereo script uses the image grid). Without it, objects are
  separated in 3D.

What it builds (all in metres)
  FLOOR      green   one big plane (found by RANSAC)
  WALLS      blue    only LARGE vertical planes (>= 1 m wide and tall, reaching the floor).
                     Everything behind a wall is discarded, so smear beyond a wall
                     never becomes an "obstacle".
  SURFACES   purple  horizontal supporting surfaces (table top, chair seats), found by scanning
                     heights, then filled with their convex hull so a glass top is closed.
  OBSTACLES  orange  everything else above the floor, clustered into objects:
                       box-like object -> oriented box (aligned with the floor, turned to
                                          match the object, not an axis-aligned box)
                       anything else   -> low-poly mesh (decimated convex hull)

Planes never cross: walls are cut where they meet the floor or another wall, and surfaces
are cut at walls. Oblique planes (neither ~horizontal nor ~vertical) are not drawn: those
are faces of objects, which are represented by their box / mesh instead.

Viewer keys:  X = points on/off   G = planes on/off   V = obstacles on/off
Mouse: left-drag orbit, ctrl+left-drag (or middle-drag) pan, wheel zoom, +/- point size.

Install:  pip install numpy opencv-python scipy open3d
"""
import json
import math
import os
import sys

import cv2
import numpy as np

OBS_DEFAULTS = {
    # --- noise model / RANSAC (floor search is unchanged from v1)
    "disparity_sigma_px": 0.6,        # stereo only: expected matching noise (with sensor={"f","B"})
    "noise_sigma_m": 0.02,            # any other sensor: constant position noise
    "point_size": 2.0,                # viewer
    "plane_thresh_m": 0.03,           # RANSAC inlier distance
    "max_planes": 5,
    "min_plane_points": 3000,
    "min_plane_area_m2": 0.25,
    "plane_cell_m": 0.04,             # raster cell size used to build plane meshes
    "gap_fill_m": 0.30,               # gaps narrower than this are closed
    "max_hole_area_m2": 1.5,          # enclosed holes smaller than this are filled
    "plane_render_offset_m": 0.04,    # plane mesh is drawn this far behind the points
    "floor_max_tilt_deg": 45.0,       # floor normal must be within this of camera 'up'
    "box_plane_min_share": 0.10,      # planes covering at least this share of the points are checked ...
    "box_plane_level_deg": 30.0,      # ... each must be within this of level or upright
    "box_parallel_min_share": 0.18,   # (a thin smear strip next to a face doesn't count)
    "box_parallel_deg": 20.0,         # ... and two big upright ones may not be this parallel
    "ransac_seed": 0,
    # --- walls
    "wall_tol_deg": 12.0,             # must be within this of vertical (90 deg to the floor)
    "wall_min_width_m": 1.0,
    "wall_min_height_m": 1.0,
    "wall_max_base_m": 0.6,           # lowest part of a wall must come down near the floor
    "wall_merge_deg": 10.0,           # near-parallel walls closer than this are one wall
    "wall_merge_dist_m": 0.25,
    "wall_band_m": 0.08,              # points this close to a wall (or behind it) are wall/background
    "wall_footprint_margin_m": 0.30,
    "behind_range_margin_m": 0.20,    # nothing is accepted farther than the farthest wall + this
    # --- horizontal surfaces (table top / seats)
    "max_supports": 10,
    "support_min_height_m": 0.15,
    "support_max_height_m": 1.30,
    "support_tol_m": 0.025,           # slab half-thickness (grows with depth noise)
    "support_tol_sigma": 1.5,
    "support_normal_cos": 0.9,        # surface normal within ~25 deg of vertical
    "support_min_points": 150,
    "support_min_area_m2": 0.06,
    "support_min_short_m": 0.15,      # thin things (chair-back posts, wall edges) are not surfaces
    "support_close_m": 0.06,
    "support_render_offset_m": 0.01,
    # --- obstacles
    "obstacle_clearance_m": 0.05,     # ignore stuff lower than this above the floor
    "max_obstacle_height_m": 2.0,
    "voxel_m": 0.01,
    "segment_disp_px": 0.5,           # neighbouring pixels join one object if disparity differs less than this
    "grounded_max_m": 0.25,           # a segment whose lowest points are above this is "floating"
    "float_merge_m": 0.15,            # floating segments touching a grounded one (top-down) join it
    "normal_radius_m": 0.08,
    "min_cluster_points": 150,        # (after voxel downsampling) smaller clusters are fragments
    "fragment_min_points": 25,
    "adopt_dist_m": 0.12,             # fragments this close to a real cluster join it, else are dropped
    "anchor_min_size_m": 0.9,         # a segment longer than this (or with several surfaces) is rebuilt around its surfaces
    "anchor_group_gap_m": 0.12,       # surfaces closer than this, at similar height, form one top (a table)
    "anchor_group_dh_m": 0.18,
    "anchor_sparse_points": 450,      # a surface seen with fewer points than this (items on a glass top) ...
    "anchor_sparse_gap_m": 0.50,      # ... join a top over this longer gap
    "anchor_min_top_area_m2": 0.10,   # a lone surface smaller than this is not furniture
    "anchor_radius_m": 0.07,          # points this close (top-down) to a top belong to that piece of furniture
    "anchor_height_m": 0.55,          # ... up to this far above it (chair back, things on a table)
    "min_obstacle_height_m": 0.08,
    "min_obstacle_size_m": 0.08,
    "core_cell_m": 0.04,              # density filter: sparse streaks (stereo smear) are ignored
    "core_min_frac": 0.08,
    "box_fill_ratio": 0.85,           # flat-top hull area / rectangle area to count as box (disc = 0.79)
    "box_footprint_fill": 0.87,       # same test on the whole footprint when there is no flat top
    "box_planar_coverage": 0.60,      # 3 biggest planes explain this share of points => box-like
    "box_planar_fill": 0.60,
    "core_open_m": 0.08,              # thin streaks narrower than this are trimmed before fitting
    "box_top_flat_fraction": 0.12,    # share of points near the top height (flat top)
    "segment_link_m": 0.05,           # no segment_fn: points closer than ~this are one object
    "mesh_triangles": 28,
    "edge_radius_m": 0.006,
}

ORANGE = (1.0, 0.5, 0.0)
PLANE_COLORS = {"floor": (0.25, 0.75, 0.35), "wall": (0.35, 0.55, 0.95), "surface": (0.70, 0.50, 0.90)}


# --------------------------------------------------------------------------- geometry helpers
def sigma_z(z, f, B_m, sd_px):
    """1-sigma depth noise (m) of a stereo rig at depth z (m)."""
    return (z * z) / (f * B_m) * sd_px


def noise_sigma(z, cfg):
    """1-sigma position noise (m) at range z (m): stereo model if the sensor is a stereo rig
    (cfg['_f'], cfg['_B'] set by analyze_point_cloud), else a constant."""
    if cfg.get("_f") and cfg.get("_B"):
        return sigma_z(z, cfg["_f"], cfg["_B"], cfg["disparity_sigma_px"])
    return np.full(np.shape(z), float(cfg["noise_sigma_m"]))


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


def odd(k):
    k = int(round(k))
    return max(3, k | 1)


# --------------------------------------------------------------------------- planes (floor code = v1)
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


def find_planes(o3d, pts, cfg):
    """Sequential RANSAC (v1). Only used to locate the FLOOR (the first plane) plus wall candidates."""
    o3d.utility.random.seed(int(cfg["ransac_seed"]))
    N = len(pts)
    z = -pts[:, 2]
    thr_pt = np.maximum(cfg["plane_thresh_m"], 2.0 * noise_sigma(z, cfg))
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


# --------------------------------------------------------------------------- walls
def largest_patch(P, n, cfg):
    """Biggest connected, gap-filled patch among a plane's inliers -> (point mask, patch)."""
    patch = plane_patch(P, n, cfg)
    if patch is None:
        return None, None
    cell = patch["cell"]
    iu = ((P @ patch["e1"] - patch["u0"]) / cell).astype(np.int64)
    iv = ((P @ patch["e2"] - patch["v0"]) / cell).astype(np.int64)
    nlab, lab = cv2.connectedComponents(patch["mask"], connectivity=8)
    pl = lab[iv, iu]
    counts = np.bincount(pl, minlength=nlab)
    counts[0] = 0
    if counts.max() == 0:
        return None, None
    return pl == counts.argmax(), patch


def inside_patch(patch, X, margin):
    """True for points whose in-plane position lies on the patch (dilated by margin)."""
    k = 2 * max(1, int(round(margin / patch["cell"]))) + 1
    m = cv2.dilate(patch["mask"], np.ones((k, k), np.uint8))
    iu = np.floor((X @ patch["e1"] - patch["u0"]) / patch["cell"]).astype(np.int64)
    iv = np.floor((X @ patch["e2"] - patch["v0"]) / patch["cell"]).astype(np.int64)
    ok = (iu >= 0) & (iu < m.shape[1]) & (iv >= 0) & (iv < m.shape[0])
    out = np.zeros(len(X), bool)
    out[ok] = m[iv[ok], iu[ok]] > 0
    return out


def find_walls(pts, planes, floor, cfg):
    """Keep only large vertical planes reaching the floor; merge duplicates."""
    nf, df = floor["n"], floor["d"]
    cands, notes = [], []
    for p in planes:
        if p is floor:
            continue
        ang = math.degrees(math.acos(min(1.0, abs(float(p["n"] @ nf)))))
        tag = f"normal=({p['n'][0]:+.2f},{p['n'][1]:+.2f},{p['n'][2]:+.2f}) {ang:.0f} deg to floor"
        if abs(ang - 90.0) > cfg["wall_tol_deg"]:
            notes.append(f"  not drawn (oblique, {tag}): treated as object faces")
            continue
        sel, patch = largest_patch(pts[p["idx"]], p["n"], cfg)
        if sel is None:
            continue
        gi = p["idx"][sel]
        Pc = pts[gi]
        n, d = fit_plane(Pc)
        if d < 0:
            n, d = -n, -d
        t = np.cross(n, nf)
        t /= max(np.linalg.norm(t), 1e-9)
        s = Pc @ t
        h = Pc @ nf + df
        width = float(np.percentile(s, 98) - np.percentile(s, 2))
        height = float(np.percentile(h, 98) - np.percentile(h, 2))
        base = float(np.percentile(h, 2))
        if width < cfg["wall_min_width_m"] or height < cfg["wall_min_height_m"] or base > cfg["wall_max_base_m"]:
            notes.append(f"  not drawn (vertical but only {width:.2f} m wide x {height:.2f} m tall: an object face)")
            continue
        patch2 = plane_patch(Pc, n, cfg)
        if patch2 is None:
            continue
        cands.append({"n": n, "d": d, "idx": gi, "patch": patch2, "area": patch2["area"],
                      "width": width, "height": height, "kind": "wall"})
    cands.sort(key=lambda w: -len(w["idx"]))
    walls = []
    for c in cands:
        dup = False
        for w in walls:
            a = math.degrees(math.acos(min(1.0, abs(float(c["n"] @ w["n"])))))
            if a < cfg["wall_merge_deg"] and abs(c["d"] - w["d"]) < cfg["wall_merge_dist_m"]:
                dup = True
                notes.append("  merged a duplicate of an existing wall")
                break
        if not dup:
            walls.append(c)
    return walls, notes


# --------------------------------------------------------------------------- horizontal surfaces
def find_supports(P, N, hd, sd, floor, cfg):
    """Horizontal supporting surfaces (table top, chair seats, tops of objects).
    Only points whose surface normal is ~vertical count; a height scan then finds slabs
    with big, thick top-down coverage."""
    nf, df = floor["n"], floor["d"]
    e1, e2 = orthonormal_basis(nf)
    horiz = np.abs(N @ nf) > cfg["support_normal_cos"]
    idx_h = np.nonzero(horiz)[0]
    if len(idx_h) < cfg["support_min_points"]:
        return []
    Ph, hh, sh = P[idx_h], hd[idx_h], sd[idx_h]
    u, v = Ph @ e1, Ph @ e2
    cell = 0.03
    u0, v0 = float(u.min()), float(v.min())
    iu = ((u - u0) / cell).astype(np.int64)
    iv = ((v - v0) / cell).astype(np.int64)
    W, H = int(iu.max()) + 1, int(iv.max()) + 1
    tol = np.maximum(cfg["support_tol_m"], cfg["support_tol_sigma"] * sh)
    avail = np.ones(len(Ph), bool)
    Hs = np.arange(cfg["support_min_height_m"], cfg["support_max_height_m"], 0.01)
    out = []
    k = odd(cfg["support_close_m"] / cell)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for _ in range(int(cfg["max_supports"])):
        scores = np.array([(avail & (np.abs(hh - h0) < tol)).sum() for h0 in Hs])
        if scores.max() < cfg["support_min_points"]:
            break
        h0 = float(Hs[scores.argmax()])
        inl = avail & (np.abs(hh - h0) < tol)
        h0 = float(np.median(hh[inl]))
        inl = avail & (np.abs(hh - h0) < tol)
        mask = (np.bincount(iv[inl] * W + iu[inl], minlength=W * H).reshape(H, W) > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
        nlab, lab = cv2.connectedComponents(mask, connectivity=8)
        pl = lab[iv, iu]
        for L in range(1, nlab):
            sel = inl & (pl == L)
            if sel.sum() < cfg["support_min_points"] // 2:
                continue
            uv = np.stack([u[sel], v[sel]], 1).astype(np.float32)
            (_, _), (rw, rh), _ = cv2.minAreaRect(uv)
            if min(rw, rh) < cfg["support_min_short_m"]:
                continue
            hull = cv2.convexHull(uv)[:, 0, :]
            if cv2.contourArea(hull) < cfg["support_min_area_m2"]:
                continue
            hull_area = cv2.contourArea(hull)
            fill = float((lab == L).sum()) * cell * cell / max(hull_area, 1e-9)  # solid seat ~1, sparse items << 1
            out.append({"H": float(np.median(hh[sel])), "hull_uv": hull, "n_points": int(sel.sum()),
                        "fill": min(fill, 1.0), "src": idx_h[np.nonzero(sel)[0]], "kind": "surface"})
        avail &= ~inl
    return out


def support_patch(hull_uv, e1, e2, cell=0.03):
    u0, v0 = float(hull_uv[:, 0].min() - cell), float(hull_uv[:, 1].min() - cell)
    W = int(math.ceil((hull_uv[:, 0].max() - u0) / cell)) + 2
    H = int(math.ceil((hull_uv[:, 1].max() - v0) / cell)) + 2
    poly = np.round((hull_uv - np.array([u0, v0])) / cell).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, poly, 1)
    return {"mask": m, "u0": u0, "v0": v0, "cell": cell, "e1": e1, "e2": e2,
            "area": float(m.sum()) * cell * cell}


# --------------------------------------------------------------------------- no-crossing clip
def cell_centers(patch, n, d):
    cy, cx = np.nonzero(patch["mask"])
    c = patch["cell"]
    uu = patch["u0"] + (cx + 0.5) * c
    vv = patch["v0"] + (cy + 0.5) * c
    X = np.outer(uu, patch["e1"]) + np.outer(vv, patch["e2"]) + (-d) * n
    return cy, cx, X


def clip_patch(patch, n, d, floor, walls, self_wall=None, margin=0.3, clip_floor=True):
    """Remove cells that would poke through the floor or through a wall (planes may only
    meet along an edge, never cross)."""
    cy, cx, X = cell_centers(patch, n, d)
    if len(cy) == 0:
        return
    bad = np.zeros(len(cy), bool)
    if clip_floor:
        bad |= (X @ floor["n"] + floor["d"]) < -0.03
    for w in walls:
        if w is self_wall:
            continue
        behind = (X @ w["n"] + w["d"]) < -0.05
        if behind.any():
            bad |= behind & inside_patch(w["patch"], X, margin)
    patch["mask"][cy[bad], cx[bad]] = 0
    patch["area"] = float(patch["mask"].sum()) * patch["cell"] ** 2


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


def footprint_core(P, e1, e2, cfg):
    """Keep the thick part of the top-down footprint: thin streaks (stereo smear, flying
    pixels) are removed by a morphological opening."""
    cell = 0.02
    u, v = P @ e1, P @ e2
    iu = ((u - u.min()) / cell).astype(np.int64)
    iv = ((v - v.min()) / cell).astype(np.int64)
    W, H = int(iu.max()) + 1, int(iv.max()) + 1
    m = (np.bincount(iv * W + iu, minlength=W * H).reshape(H, W) > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k = odd(cfg["core_open_m"] / cell)
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(k // 2 + 1):  # grow the core back into the original footprint: corners return,
        core = cv2.dilate(core, cross) & m  # but streaks only keep a stub
    keep = core[iv, iu] > 0
    return keep if keep.mean() >= 0.25 else np.ones(len(P), bool)


def planar_coverage(o3d, P, cfg, up, max_planes=3):
    """(coverage, boxlike).  coverage: share of the points explained by the few biggest
    planes (~1 for a box, low for round / spiky / thin things).  boxlike: those planes are
    consistent with the faces of an upright box seen from one side - each one level or
    upright (a ball's tilted facets are not), and no two big upright ones parallel (a
    cylinder's flank splits into near-parallel strips; opposite box faces are never both seen)."""
    if len(P) < 60:
        return 0.0, False
    sig = float(np.median(noise_sigma(-P[:, 2], cfg)))
    thr = float(min(0.04, max(0.015, sig)))
    rem, covered, planes = P, 0, []
    o3d.utility.random.seed(int(cfg["ransac_seed"]) + 1)  # same points -> same planes, every run
    for _ in range(max_planes):
        if len(rem) < 40:
            break
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(rem)
        model, inl = pc.segment_plane(thr, 3, 300)
        covered += len(inl)
        nn = np.array(model[:3], dtype=float)
        nn /= np.linalg.norm(nn)
        planes.append((len(inl) / len(P), nn, float(np.degrees(np.arccos(min(1.0, abs(nn @ up)))))))
        keep = np.ones(len(rem), bool)
        keep[inl] = False
        rem = rem[keep]
    boxlike = True
    big = [pl for pl in planes if pl[0] >= cfg["box_plane_min_share"]]
    for share, nn, tilt in big:
        if cfg["box_plane_level_deg"] < tilt < 90.0 - cfg["box_plane_level_deg"]:
            boxlike = False  # a tilted facet: curved surface
    upright = [pl for pl in planes if pl[2] >= 90.0 - cfg["box_plane_level_deg"]
               and pl[0] >= cfg["box_parallel_min_share"]]
    for i in range(len(upright)):
        for j in range(i + 1, len(upright)):
            ang = math.degrees(math.acos(min(1.0, abs(float(upright[i][1] @ upright[j][1])))))
            if ang < cfg["box_parallel_deg"]:
                boxlike = False  # two big parallel upright strips: a cylinder's flank
    if cfg.get("_verbose"):
        print("      [planes] " + " ".join(f"{sh:.2f}@{t:.0f}deg" for sh, _, t in planes) + f" boxlike={boxlike}")
    return covered / len(P), boxlike


def fit_obstacle(o3d, P, floor, cfg, extra=None, ground=False):
    n, d = floor["n"], floor["d"]
    e1, e2 = orthonormal_basis(n)
    P = P[density_core(P, e1, e2, cfg)]
    P = P[footprint_core(P, e1, e2, cfg)]
    coverage, boxlike = planar_coverage(o3d, P, cfg, n)
    if extra is not None and len(extra):
        P = np.vstack([P, extra])
    h = P @ n + d
    xy = np.stack([P @ e1, P @ e2], 1).astype(np.float32)

    def rect_fit(pts2):
        rd = cv2.minAreaRect(pts2)
        area = max(rd[1][0] * rd[1][1], 1e-9)
        return rd, float(cv2.contourArea(cv2.convexHull(pts2))) / area

    full_def, full_ratio = rect_fit(xy)  # whole footprint (sides may be occluded / smeared)
    rect_def, ratio = full_def, full_ratio
    bottom = float(h.min())
    top = float(np.percentile(h, 98))
    low = bottom
    if bottom < 0.15 or ground:
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
        full_area = max(full_def[1][0] * full_def[1][1], 1e-9)
        top_area = top_def[1][0] * top_def[1][1]
        # the flat top must be most of the object's footprint (a sphere's small cap is not)
        if top_ratio >= cfg["box_fill_ratio"] and top_area >= 0.5 * full_area:
            boxy, rect_def, ratio = True, top_def, top_ratio
    if not boxy and coverage >= cfg["box_planar_coverage"] and full_ratio >= cfg["box_planar_fill"] \
            and (boxlike or ground):
        boxy, rect_def, ratio = True, full_def, full_ratio  # a few big flat faces: box (even if leaning)
    if not boxy and full_ratio >= cfg["box_footprint_fill"]:
        boxy, rect_def, ratio = True, full_def, full_ratio  # tilted / uneven top, rectangular footprint
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
            "planar_coverage": round(float(coverage), 2),
            "bottom_m": round(float(low), 2), "top_m": round(float(top), 2),
            "n_points": int(len(P))}

    ob = {"info": info, "box_corners": corners, "box_edges": box_edges, "mesh": None}
    if boxy:
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


def prepare_points(o3d, Pc, labc, floor, cfg):
    """Thin the candidates to one point per voxel and segment (keeping labels), drop outliers,
    estimate normals."""
    key = np.floor(Pc / cfg["voxel_m"]).astype(np.int64)
    key = np.concatenate([key, labc[:, None]], 1)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    P, lab = Pc[first], labc[first]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(P)
    if len(P) > 30:
        pc, ind = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
        ind = np.asarray(ind)
        P, lab = P[ind], lab[ind]
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg["normal_radius_m"], max_nn=40))
    N = np.asarray(pc.normals)
    sd = noise_sigma(-P[:, 2], cfg)
    hd = P @ floor["n"] + floor["d"]
    return P, N, hd, sd, lab


def tidy_segments(P, lab, cfg):
    """Keep real segments; small fragments join a nearby real one (3D) or are dropped."""
    from scipy.spatial import cKDTree
    sizes = np.bincount(lab)
    big = np.nonzero(sizes >= cfg["min_cluster_points"])[0]
    out = -np.ones(len(lab), int)
    for new, L in enumerate(big):
        out[lab == L] = new
    if len(big):
        tr = cKDTree(P[out >= 0])
        owner = out[out >= 0]
        for L in np.nonzero((sizes >= cfg["fragment_min_points"]) & (sizes < cfg["min_cluster_points"]))[0]:
            mk = lab == L
            dd, j = tr.query(P[mk])
            near = dd < cfg["adopt_dist_m"]
            if near.any():
                out[mk] = np.bincount(owner[j[near]]).argmax()
    return out


def hull_ring(hull_uv, H, e1, e2, n, d, step=0.05):
    ring = []
    for a, b in zip(hull_uv, np.roll(hull_uv, -1, axis=0)):
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / step)), endpoint=False):
            ring.append(a + (b - a) * t)
    return np.array([uv_to_xyz(p[0], p[1], H, e1, e2, n, d) for p in ring])


def footprint_gap(Pa, Pb, e1, e2):
    """Smallest top-down distance between two point sets (convex hulls)."""
    from scipy.spatial import cKDTree
    A = np.stack([Pa @ e1, Pa @ e2], 1)
    Bp = np.stack([Pb @ e1, Pb @ e2], 1)
    if len(A) > 3000:
        A = A[::max(1, len(A) // 3000)]
    return float(cKDTree(Bp).query(A)[0].min())


def merge_floating(P, hd, lab, floor, cfg):
    """Segments that do not touch the floor (things on a table top, a coat on a chair) join the
    nearest segment that does, if they touch it in the top-down view."""
    e1, e2 = orthonormal_basis(floor["n"])
    ids = [L for L in range(lab.max() + 1) if (lab == L).any()]
    low = {L: float(np.percentile(hd[lab == L], 3)) for L in ids}
    grounded = [L for L in ids if low[L] < cfg["grounded_max_m"]]
    for L in ids:
        if L in grounded:
            continue
        best, bd = None, cfg["float_merge_m"]
        for G in grounded:
            gdist = footprint_gap(P[lab == L], P[lab == G], e1, e2)
            if gdist < bd:
                best, bd = G, gdist
        if best is not None:
            lab[lab == L] = best
    return lab


def seat_distance(uv, hull):
    """Signed top-down distance from points to a convex polygon (negative inside)."""
    hull = hull if cv2.contourArea(hull.astype(np.float32), True) >= 0 else hull[::-1]
    best = np.full(len(uv), -1e9)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-9:
            continue
        nrm = np.array([e[1], -e[0]]) / L  # outward for counter-clockwise polygons
        best = np.maximum(best, (uv - a) @ nrm)
    return best


def group_supports(sups, cfg):
    """Surfaces that touch (top-down) at a similar height are one top, e.g. the patches seen
    on a glass table."""
    parent = list(range(len(sups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sups)):
        for j in range(i + 1, len(sups)):
            if abs(sups[i]["H"] - sups[j]["H"]) > cfg["anchor_group_dh_m"]:
                continue
            A, Bh = sups[i]["hull_uv"], sups[j]["hull_uv"]
            gap = min(float(seat_distance(A, Bh).min()), float(seat_distance(Bh, A).min()))
            sparse = min(sups[i]["n_points"], sups[j]["n_points"]) < cfg["anchor_sparse_points"]
            if gap < (cfg["anchor_sparse_gap_m"] if sparse else cfg["anchor_group_gap_m"]):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(sups)):
        groups.setdefault(find(i), []).append(sups[i])
    return list(groups.values())


def anchored_pieces(Pk, sup, floor, cfg):
    """Rebuild a messy furniture segment around its horizontal surfaces: one piece per top
    (seat, table top), made of the points just around / below / above it."""
    n, d = floor["n"], floor["d"]
    e1, e2 = orthonormal_basis(n)
    uv = np.stack([Pk @ e1, Pk @ e2], 1)
    hk = Pk @ n + d
    pieces = []
    tops = []
    for g in group_supports(sup, cfg):
        pts_uv = np.concatenate([s["hull_uv"] for s in g]).astype(np.float32)
        hull = cv2.convexHull(pts_uv)[:, 0, :]
        if cv2.contourArea(hull) < cfg["anchor_min_top_area_m2"]:
            continue
        tops.append({"hull": hull, "H": float(np.median([s["H"] for s in g])), "sups": g})
    if not tops:
        return []
    if cfg.get("_verbose"):
        for t in tops:
            c = t["hull"].mean(0)
            print(f"    top at {t['H']:.2f} m, {len(t['sups'])} patch(es), area "
                  f"{cv2.contourArea(t['hull'].astype(np.float32)):.2f} m2, fill "
                  f"{[round(x['fill'], 2) for x in t['sups']]}, centre u,v=({c[0]:+.2f},{c[1]:+.2f})")
    dist = np.stack([seat_distance(uv, t["hull"]) for t in tops], 1)
    near = dist.argmin(1)
    for k, t in enumerate(tops):
        sel = (near == k) & (dist[:, k] < cfg["anchor_radius_m"]) & (hk < t["H"] + cfg["anchor_height_m"])
        if sel.sum() < cfg["fragment_min_points"]:
            continue
        pieces.append((Pk[sel], t))
    return pieces


def find_obstacles(P, N, hd, sd, lab, supports, floor, o3d, cfg):
    n, d = floor["n"], floor["d"]
    e1, e2 = orthonormal_basis(n)
    lab = tidy_segments(P, lab, cfg)
    if len(lab) == 0 or lab.max() < 0:
        return [], lab
    lab = merge_floating(P, hd, lab, floor, cfg)
    rs = np.random.default_rng(0)
    for s in supports:  # which segment owns each surface
        src = s["src"]
        src = src[rs.choice(len(src), min(len(src), 300), replace=False)]
        own = lab[src]
        own = own[own >= 0]
        s["cluster"] = int(np.bincount(own).argmax()) if len(own) else -1
    # One table can be seen as several image segments (glass top): segments whose surfaces
    # group into the same top become a single segment.
    remap = {}
    for g in group_supports([s for s in supports if s["cluster"] >= 0], cfg):
        cl = sorted({s["cluster"] for s in g})
        for c in cl[1:]:
            remap[c] = cl[0]
    if remap:
        def root(c):
            while c in remap:
                c = remap[c]
            return c
        for c in list(remap):
            lab[lab == c] = root(c)
        for s in supports:
            if s["cluster"] in remap:
                s["cluster"] = root(s["cluster"])

    obstacles = []

    def add(Pk, ex, L, anchored, used):
        ob = fit_obstacle(o3d, Pk, floor, cfg, ex, ground=anchored)
        sz = ob["info"]["size_m"]
        if sz[2] < cfg["min_obstacle_height_m"] or sz[0] < cfg["min_obstacle_size_m"]:
            return
        if not anchored and ob["info"]["bottom_m"] > cfg["grounded_max_m"]:
            return  # floating leftover (something on a wall, a hanging item): not an obstacle
        ob["cluster"], ob["anchored"], ob["used"] = L, anchored, [id(x) for x in used]
        obstacles.append(ob)

    for L in range(lab.max() + 1):
        Pk = P[lab == L]
        if len(Pk) < cfg["fragment_min_points"]:
            continue
        sup = [s for s in supports if s["cluster"] == L]
        long_side = 0.0
        if len(Pk) > 20:
            core = Pk[footprint_core(Pk, e1, e2, cfg)]
            (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([core @ e1, core @ e2], 1).astype(np.float32))
            long_side = max(rw, rh)
        if sup and (len(sup) >= 2 or long_side > cfg["anchor_min_size_m"]):
            for Ppiece, top in anchored_pieces(Pk, sup, floor, cfg):
                add(Ppiece, hull_ring(top["hull"], top["H"], e1, e2, n, d), L, True, top["sups"])
            continue
        ex = [hull_ring(s["hull_uv"], s["H"], e1, e2, n, d) for s in sup]
        add(Pk, np.vstack(ex) if ex else None, L, False, sup)
    obstacles.sort(key=lambda o: o["info"]["range_m"])
    for i, o in enumerate(obstacles):
        o["id"] = i + 1
    return obstacles, lab


# --------------------------------------------------------------------------- viewing
def build_groups(o3d, pts, cols, floor, walls, supports, obstacles, cfg, solid_meshes, boxes_for_all):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    groups = {"points": [pcd], "planes": [], "obstacles": [],
              "frame": [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]}
    nf, df = floor["n"], floor["d"]
    m = patch_mesh(o3d, floor["patch"], nf, df, cfg["plane_render_offset_m"], PLANE_COLORS["floor"])
    if m is not None:
        groups["planes"].append(m)
    for w in walls:
        m = patch_mesh(o3d, w["patch"], w["n"], w["d"], cfg["plane_render_offset_m"], PLANE_COLORS["wall"])
        if m is not None:
            groups["planes"].append(m)
    for s in supports:
        m = patch_mesh(o3d, s["patch"], nf, df - s["H"], cfg["support_render_offset_m"], PLANE_COLORS["surface"])
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


def save_debug_views(path, pts, cols, Pd, lab, walls, supports, floor, obstacles=None):
    """Camera view + top-down view of the clusters, surfaces and fitted obstacles (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    nf, df = floor["n"], floor["d"]
    e1, e2 = orthonormal_basis(nf)
    fig, ax = plt.subplots(1, 3, figsize=(27, 10))
    x, y = pts[::6, 0] / -pts[::6, 2], pts[::6, 1] / -pts[::6, 2]
    ax[0].scatter(x, y, c=cols[::6], s=1)
    ax[0].set_title("scene")
    if Pd is not None and len(Pd):
        rng = np.random.default_rng(3)
        nl = (lab.max() + 1) if len(lab) and lab.max() >= 0 else 0
        pal = rng.uniform(0.2, 1, (nl + 1, 3))
        c = np.where(lab[:, None] >= 0, pal[lab] if nl else 0.5, 0.8)
        px, py = Pd[:, 0] / -Pd[:, 2], Pd[:, 1] / -Pd[:, 2]
        ax[1].scatter(x, y, c="0.92", s=1)
        ax[1].scatter(px, py, c=c, s=3)
        for L in range(nl):
            m = lab == L
            if m.any():
                ax[1].text(np.median(px[m]), np.median(py[m]), str(L), fontsize=14, weight="bold")
        ax[2].scatter(Pd @ e1, Pd @ e2, c=c, s=3)
    for i, sp in enumerate(supports):
        hull = np.vstack([sp["hull_uv"], sp["hull_uv"][:1]])
        ax[2].plot(hull[:, 0], hull[:, 1], "k-", lw=1.5)
        ax[2].text(hull[:, 0].mean(), hull[:, 1].mean(), f"S{i}\n{sp['H']:.2f}m", fontsize=9, ha="center")
        X = np.array([uv_to_xyz(p[0], p[1], sp["H"], e1, e2, nf, df) for p in hull])
        ax[1].plot(X[:, 0] / -X[:, 2], X[:, 1] / -X[:, 2], "k-", lw=1)
    for o in obstacles or []:
        cn = o["box_corners"][:4]
        q = np.vstack([cn, cn[:1]])
        ax[2].plot(q @ e1, q @ e2, "r-" if o["kind"] == "box" else "m--", lw=2)
        ax[2].text((cn @ e1).mean(), (cn @ e2).mean(), f"#{o['id']}", color="r", fontsize=12, weight="bold")
    ax[1].set_title("segments (labels) + surfaces")
    ax[2].set_title("top-down: segments, surfaces (black), obstacle footprints")
    fx, fy = (-df * nf) @ e1, (-df * nf) @ e2
    ax[2].plot([fx], [fy], "k^", ms=10)
    for a_ in ax:
        a_.set_aspect("equal")
    plt.savefig(path, dpi=50)
    plt.close(fig)


# --------------------------------------------------------------------------- entry point
def default_config(user=None):
    """Algorithm defaults, overridden by the keys in `user` (e.g. the stereo config dict)."""
    cfg = dict(OBS_DEFAULTS)
    if user:
        cfg.update({k: v for k, v in user.items() if not str(k).startswith("_")})
    return cfg


def segment_points_3d(P, cfg):
    """Fallback object segmentation for a bare point cloud (no image grid): points closer than
    cfg['segment_link_m'] are one object (single linkage on a voxel-thinned cloud)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    key = np.floor(P / cfg["segment_link_m"]).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    Q = P[first]
    pairs = cKDTree(Q).query_pairs(cfg["segment_link_m"] * 1.8, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q)))
    _, lab = connected_components(g, directed=False)
    return lab[inv]


def analyze_point_cloud(o3d, pts, cfg, segment_fn=None, sensor=None, verbose=False):
    """Find the floor, walls, horizontal surfaces and obstacles in a point cloud.

    pts        (N,3) metres, +y up, sensor at the origin (see module docstring)
    cfg        from default_config()
    segment_fn optional: segment_fn(cand_mask) -> int labels for pts[cand_mask]
    sensor     optional {"f": focal_px, "B": baseline_m} for the stereo depth-noise model
    Returns a dict: floor, walls, supports, obstacles, notes, Pd (analysed points), lab.
    """
    pts = np.asarray(pts, dtype=np.float64)
    cfg["_f"], cfg["_B"] = (sensor["f"], sensor["B"]) if sensor else (None, None)
    cfg["_verbose"] = verbose

    # ---- floor + walls
    planes = find_planes(o3d, pts, cfg)
    if not planes:
        raise RuntimeError("No planes found. Try a lower min_plane_area_m2 / min_plane_points.")
    floor = pick_floor(pts, planes, cfg)
    nf, df = floor["n"], floor["d"]
    walls, notes = find_walls(pts, planes, floor, cfg)

    # ---- points that are neither floor nor wall nor behind a wall
    hgt = pts @ nf + df
    z = -pts[:, 2]
    sig = noise_sigma(z, cfg)
    excl = np.zeros(len(pts), bool)
    excl[floor["idx"]] = True
    for wl in walls:
        excl[wl["idx"]] = True
        wd = pts @ wl["n"] + wl["d"]
        on_or_behind = wd < np.maximum(cfg["wall_band_m"], 2 * sig)
        excl |= on_or_behind & inside_patch(wl["patch"], pts, cfg["wall_footprint_margin_m"])
    if walls:
        z_far = max(float(np.percentile(-pts[wl["idx"], 2], 98)) for wl in walls)
        excl |= z > z_far + cfg["behind_range_margin_m"]
    cand = (hgt > cfg["obstacle_clearance_m"] + 2 * sig) & (hgt < cfg["max_obstacle_height_m"]) & ~excl
    Pc = pts[cand]

    # ---- horizontal surfaces + obstacles
    Pd = lab = None
    supports, obstacles = [], []
    if len(Pc) > cfg["min_cluster_points"]:
        labc = segment_fn(cand) if segment_fn else segment_points_3d(Pc, cfg)
        Pd, Nd, hd, sd, labd = prepare_points(o3d, Pc, labc, floor, cfg)
        supports = find_supports(Pd, Nd, hd, sd, floor, cfg)
        obstacles, lab = find_obstacles(Pd, Nd, hd, sd, labd, supports, floor, o3d, cfg)

    used = {i for o in obstacles for i in o.get("used", [])}
    supports = [s_ for s_ in supports if id(s_) in used]

    # ---- make the planes consistent (no crossing)
    e1, e2 = orthonormal_basis(nf)
    for s_ in supports:
        s_["patch"] = support_patch(s_["hull_uv"], e1, e2)
    for wl in walls:
        clip_patch(wl["patch"], wl["n"], wl["d"], floor, walls, self_wall=wl)
    for s_ in supports:
        clip_patch(s_["patch"], nf, df - s_["H"], floor, walls, clip_floor=False)
    walls = [wl for wl in walls if wl["patch"]["area"] >= cfg["min_plane_area_m2"]]
    supports = [s_ for s_ in supports if s_["patch"]["area"] >= cfg["support_min_area_m2"]]
    return {"floor": floor, "walls": walls, "supports": supports, "obstacles": obstacles,
            "notes": notes, "Pd": Pd, "lab": lab}


# --------------------------------------------------------------------------- report / export
def print_report(res):
    floor, nf, df = res["floor"], res["floor"]["n"], res["floor"]["d"]
    print("\nPLANES")
    print(f"  FLOOR   normal=({nf[0]:+.2f},{nf[1]:+.2f},{nf[2]:+.2f}) area={floor['area']:.2f} m2 "
          f"| sensor height {df:.2f} m, tilt vs sensor up {math.degrees(math.acos(min(1, float(nf[1])))):.1f} deg")
    for wl in res["walls"]:
        n_ = wl["n"]
        print(f"  WALL    normal=({n_[0]:+.2f},{n_[1]:+.2f},{n_[2]:+.2f}) {wl['width']:.1f} m wide x "
              f"{wl['height']:.1f} m tall, {wl['d']:.2f} m away")
    for s_ in res["supports"]:
        print(f"  SURFACE horizontal at {s_['H']:.2f} m above floor, area={s_['patch']['area']:.2f} m2")
    for t in res["notes"]:
        print(t)
    print("\nOBSTACLES (nearest first)  size = long x short x height, yaw vs sensor-right axis")
    if not res["obstacles"]:
        print("  none found")
    for o in res["obstacles"]:
        i = o["info"]
        sz = i["size_m"]
        print(f"  #{o['id']} {o['kind']:4s} size={sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f} m  yaw={i['yaw_deg']:+.0f} deg  "
              f"range={i['range_m']:.2f} m  center=({i['center_m'][0]:+.2f},{i['center_m'][1]:+.2f},"
              f"{i['center_m'][2]:+.2f})  fill={i['footprint_fill']} planar={i['planar_coverage']} "
              f"floor-to-top={i['top_m']:.2f} m{' (furniture, anchored on its top)' if o.get('anchored') else ''}")


def result_to_json(res):
    nf, df = res["floor"]["n"], res["floor"]["d"]
    return {"floor": {"normal": nf.tolist(), "d": df},
            "walls": [{"normal": wl["n"].tolist(), "d": wl["d"], "width_m": wl["width"],
                       "height_m": wl["height"]} for wl in res["walls"]],
            "surfaces": [{"height_m": s_["H"], "area_m2": s_["patch"]["area"]} for s_ in res["supports"]],
            "obstacles": [{"id": o["id"], "kind": o["kind"], **o["info"],
                           "box_corners": o["box_corners"].tolist(),
                           **({"mesh_vertices": np.asarray(o["mesh"].vertices).tolist(),
                               "mesh_triangles": np.asarray(o["mesh"].triangles).tolist()}
                              if o["mesh"] is not None else {})} for o in res["obstacles"]]}


def scene_groups(o3d, pts, cols, res, cfg, solid_meshes=False, boxes_for_all=False):
    return build_groups(o3d, pts, cols, res["floor"], res["walls"], res["supports"], res["obstacles"],
                        cfg, solid_meshes, boxes_for_all)


def debug_views(path, pts, cols, res):
    save_debug_views(path, pts, cols, res["Pd"], res["lab"], res["walls"], res["supports"],
                     res["floor"], res["obstacles"])