#!/usr/bin/env python3
"""
floor_path_planner.py -- draw a recommended walking path on a photo of a room.

Pipeline
    1. Segment the walkable floor       (SegFormer / ADE20K  -or-  MobileSAM + YOLO fallback)
    2. Build a cost map                 (obstacle clearance that grows toward the camera,
                                         so near objects get a bigger safety margin)
    3. Pick start and goal              (start = bottom-centre; goal = --goal or the
                                         most open spot at the far end of the floor)
    4. A* / Dijkstra over the cost map  (skimage.graph.route_through_array)
    5. Smooth with a spline, check it still lies in free space, draw it on the image.

Usage
    python floor_path_planner.py room.jpg
    python floor_path_planner.py room.jpg --goal 320,1185 --debug
    python floor_path_planner.py room.jpg --backend segformer --model nvidia/segformer-b5-finetuned-ade-640-640

Outputs (next to --out, default <image>_path.png)
    <stem>_path.png    the photo with the path drawn on it
    <stem>_path.json   waypoints in original-image pixels (x, y), start, goal, timings
    <stem>_debug.png   (with --debug) floor mask / traversable area / path side by side

Install
    pip install numpy scipy opencv-python-headless pillow scikit-image torch transformers
    pip install ultralytics          # only needed for the fallback backend

Backends
    segformer  Hugging Face SegFormer trained on ADE20K; floor = classes whose label contains
               "floor", "rug" or "carpet". Needs to download the model from huggingface.co.
    sam        Fallback for machines that cannot reach huggingface.co. Detects objects with
               YOLO11-seg and grows the floor from seed points at the bottom-centre of the
               frame with MobileSAM. ASSUMES the bottom-centre of the photo is floor.
    auto       (default) try segformer, fall back to sam if the model cannot be loaded.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array

FLOOR_WORDS = ("floor", "rug", "carpet")


# --------------------------------------------------------------------------- segmentation
def segment_segformer(rgb, model_name, floor_words=FLOOR_WORDS):
    """Return (floor_mask, extra_obstacle_mask, info) at full image resolution."""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    inputs = proc(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy()
    ids = [int(i) for i, n in model.config.id2label.items()
           if any(w in str(n).lower() for w in floor_words)]
    if not ids:
        raise RuntimeError("no floor-like class in id2label of " + model_name)
    floor = np.isin(pred, ids)
    return floor, np.zeros_like(floor), {"floor_class_ids": ids}


def segment_sam(rgb, yolo_weights="yolo11s-seg.pt", sam_weights="mobile_sam.pt", conf=0.15):
    """Fallback: YOLO-seg obstacles + MobileSAM floor grown from bottom-centre seeds."""
    from ultralytics import YOLO, SAM

    H, W = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1) obstacles: every detected object (chairs, tables, bags, people, laptops ...)
    det = YOLO(yolo_weights).predict(bgr, conf=conf, verbose=False)[0]
    obst = np.zeros((H, W), np.uint8)
    if det.masks is not None:
        for poly in det.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(obst, [poly.astype(np.int32)], 1)
    obst = obst.astype(bool)

    # 2) floor: SAM prompted with seeds along the bottom of the frame
    near_obst = cv2.dilate(obst.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    seeds = [(int(W * fx), int(H * fy)) for fy in (0.97, 0.90, 0.80, 0.70)
             for fx in (0.35, 0.5, 0.65)]
    seeds = [s for s in seeds if not near_obst[s[1], s[0]]]
    if not seeds:
        raise RuntimeError("no floor seed found at the bottom of the image")
    sam = SAM(sam_weights)
    masks = []
    for x, y in seeds:
        r = sam(bgr, points=[[x, y]], labels=[1], verbose=False)[0]
        if r.masks is not None:
            m = r.masks.data.cpu().numpy()[0] > 0.5
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            masks.append(m)
    if not masks:
        raise RuntimeError("SAM returned no masks")
    # keep the masks that agree with each other (drops a seed that landed on a bag, etc.)
    areas = np.array([m.sum() for m in masks], float)
    agree = [sum((m & o).sum() / max((m | o).sum(), 1) for o in masks) for m in masks]
    ref = masks[int(np.argmax(agree))]
    keep = [m for m in masks if (m & ref).sum() / max((m | ref).sum(), 1) >= 0.3]
    floor = np.any(keep, axis=0) & ~obst
    return floor, obst, {"seeds": seeds, "masks_kept": len(keep), "masks_total": len(masks)}


def segment(rgb, backend, model_name):
    if backend in ("auto", "segformer"):
        try:
            f, o, info = segment_segformer(rgb, model_name)
            info["backend"] = "segformer"
            return f, o, info
        except Exception as e:  # noqa: BLE001
            if backend == "segformer":
                raise
            print(f"[warn] SegFormer unavailable ({type(e).__name__}: {str(e)[:120]}); "
                  f"falling back to MobileSAM + YOLO", file=sys.stderr)
    f, o, info = segment_sam(rgb)
    info["backend"] = "sam"
    return f, o, info


# --------------------------------------------------------------------------- planning helpers
def largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool), lab
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k, lab


def nearest_true(mask, x, y):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[i]), int(ys[i])


def parse_xy(s, W, H):
    if s is None or s == "auto":
        return None
    a, b = [float(v) for v in s.split(",")]
    if a <= 1.0 and b <= 1.0 and "." in s:          # fractions of width / height
        a, b = a * W, b * H
    return a, b


def smooth_path(pts_xy, ok_fn, tol_candidates=(6, 3, 1.5)):
    """Spline-smooth an (N,2) polyline; keep the smoothest version that stays in free space."""
    pts = np.asarray(pts_xy, float)
    if len(pts) < 5:
        return pts
    sub = pts[:: max(1, len(pts) // 60)]
    if np.linalg.norm(sub[-1] - pts[-1]) > 0:
        sub = np.vstack([sub, pts[-1]])
    for tol in tol_candidates:
        try:
            tck, _ = splprep([sub[:, 0], sub[:, 1]], s=len(sub) * tol ** 2, k=min(3, len(sub) - 1))
            u = np.linspace(0, 1, 400)
            x, y = splev(u, tck)
            sm = np.stack([x, y], 1)
            if ok_fn(sm):
                return sm
        except Exception:  # noqa: BLE001
            continue
    return pts


# --------------------------------------------------------------------------- main planner
def plan_path(rgb, backend="auto", model_name="nvidia/segformer-b2-finetuned-ade-512-512",
              start=None, goal=None, clearance=0.05, horizon=0.40, max_side=512):
    t0 = time.time()
    H, W = rgb.shape[:2]
    floor, extra_obst, info = segment(rgb, backend, model_name)
    t_seg = time.time() - t0

    # ---- work on a down-scaled grid for speed
    s = min(1.0, max_side / max(H, W))
    pw, ph = max(8, int(round(W * s))), max(8, int(round(H * s)))

    def down(m):
        return cv2.resize(m.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA) > 0.5

    free = down(floor) & ~down(extra_obst)
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    # safety radius grows linearly from the horizon line to the bottom of the image
    yy = np.arange(ph, dtype=np.float32)[:, None]
    hy = horizon * ph
    frac = np.clip((yy - hy) / max(ph - hy, 1), 0, 1)
    r_map = np.maximum(2.0, clearance * pw * frac) * np.ones((1, pw), np.float32)

    # ---- traversable space (relax the margin if the scene is very cramped)
    relax = 1.0
    for _ in range(4):
        trav = free & (dist >= r_map * relax)
        main, _ = largest_component(trav)
        if main.sum() > 0.01 * pw * ph:
            break
        relax *= 0.5
    if main.sum() == 0:
        raise RuntimeError("no traversable floor found")

    # ---- start and goal
    st = parse_xy(start, W, H) or (0.5 * W, 0.99 * H)
    sx, sy = nearest_true(main, st[0] * s, st[1] * s)
    reach = main
    g = parse_xy(goal, W, H)
    if g is not None:
        gx, gy = nearest_true(reach, g[0] * s, g[1] * s)
    else:
        open_ok = reach & (dist >= 1.5 * r_map * relax)
        cand = open_ok if open_ok.sum() > 20 else reach
        ys_, xs_ = np.nonzero(cand)
        far = ys_ <= ys_.min() + 0.04 * ph             # the far end of the floor ...
        i = np.argmax(np.where(far, dist[ys_, xs_], -1))  # ... at its most open point
        gx, gy = int(xs_[i]), int(ys_[i])

    # ---- cost map + route
    cost = 1.0 + 3.0 * np.clip(1.0 - (dist - r_map * relax) / (4.0 * r_map * relax), 0, 1)
    cost[~main] = np.inf
    route, _ = route_through_array(cost, (sy, sx), (gy, gx), fully_connected=True, geometric=True)
    raw = np.array([(c, r) for r, c in route], float)           # (x, y) in grid px

    def ok_fn(pts):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, pw - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, ph - 1)
        return bool(np.all(free[yi, xi] & (dist[yi, xi] >= 0.6 * r_map[yi, xi] * relax)))

    smooth = smooth_path(raw, ok_fn)
    path_full = smooth / s                                       # back to original pixels
    return {
        "path": path_full, "start": (sx / s, sy / s), "goal": (gx / s, gy / s),
        "floor": floor, "extra_obstacles": extra_obst, "info": info,
        "grid": {"scale": s, "free": free, "traversable": main, "dist": dist, "r_map": r_map},
        "seg_seconds": t_seg, "total_seconds": time.time() - t0, "clearance_relax": relax,
    }


# --------------------------------------------------------------------------- drawing
def draw_path(rgb, res, markers=True):
    H, W = rgb.shape[:2]
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    th = max(6, int(0.012 * W))
    pts = np.round(res["path"]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], False, (30, 30, 230), th, cv2.LINE_AA)       # red (BGR)
    if markers:
        cv2.circle(out, tuple(int(v) for v in res["start"]), int(th * 1.6), (60, 200, 60), -1, cv2.LINE_AA)
        cv2.circle(out, tuple(int(v) for v in res["goal"]), int(th * 1.6), (230, 120, 20), -1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def debug_panel(rgb, res):
    H, W = rgb.shape[:2]
    s = res["grid"]["scale"]

    def up(m):
        return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    def tint(base, mask, color, a=0.55):
        o = base.copy()
        o[mask] = ((1 - a) * o[mask] + a * np.array(color)).astype(np.uint8)
        return o

    p1 = tint(rgb, res["floor"], (0, 255, 0))
    p2 = tint(rgb, up(res["grid"]["traversable"]), (0, 200, 255))
    p3 = draw_path(rgb, res)
    k = 600.0 / W
    panels = [cv2.resize(p, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) for p in (p1, p2, p3)]
    return np.hstack(panels)


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", help="output image (default <image>_path.png)")
    ap.add_argument("--nojson", action="store_true", help="also write <image>_path.json")
    ap.add_argument("--backend", choices=["auto", "segformer", "sam"], default="auto")
    ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-ade-512-512",
                    help="SegFormer model id or local folder")
    ap.add_argument("--start", default=None, help="x,y in pixels (or fractions); default bottom-centre")
    ap.add_argument("--goal", default=None, help="x,y in pixels (or fractions); default: far, open floor")
    ap.add_argument("--clearance", type=float, default=0.05,
                    help="safety margin at the bottom of the image, as a fraction of image width")
    ap.add_argument("--horizon", type=float, default=0.40,
                    help="row (fraction of height) where the margin shrinks to its minimum")
    ap.add_argument("--max-side", type=int, default=512, help="planning-grid size (pixels)")
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--debug", action="store_true", help="also write a debug panel")
    a = ap.parse_args()

    rgb = np.array(ImageOps.exif_transpose(Image.open(a.image)).convert("RGB"))
    res = plan_path(rgb, a.backend, a.model, a.start, a.goal, a.clearance, a.horizon, a.max_side)

    stem = a.out or os.path.splitext(a.image)[0] + "_path.png"
    base = os.path.splitext(stem)[0]
    Image.fromarray(draw_path(rgb, res, not a.no_markers)).save(stem)
    if not a.nojson:
        with open(base + ".json", "w") as f:
            json.dump({
                "image": os.path.basename(a.image), "size": [rgb.shape[1], rgb.shape[0]],
                "backend": res["info"].get("backend"), "start": [round(v, 1) for v in res["start"]],
                "goal": [round(v, 1) for v in res["goal"]],
                "waypoints_xy": [[round(float(x), 1), round(float(y), 1)] for x, y in res["path"][::8]],
            "segmentation_seconds": round(res["seg_seconds"], 2),
            "total_seconds": round(res["total_seconds"], 2),
            "clearance_relax": res["clearance_relax"],
        }, f, indent=1)
    if a.debug:
        Image.fromarray(debug_panel(rgb, res)).save(base.replace("_path", "") + "_debug.png")
        np.savez_compressed(base + "_maps.npz", floor=res["floor"], path=res["path"],
                            traversable=res["grid"]["traversable"], scale=res["grid"]["scale"])
    print(f"backend={res['info'].get('backend')}  start={tuple(int(v) for v in res['start'])}  "
          f"goal={tuple(int(v) for v in res['goal'])}  time={res['total_seconds']:.1f}s -> {stem}")


if __name__ == "__main__":
    main()