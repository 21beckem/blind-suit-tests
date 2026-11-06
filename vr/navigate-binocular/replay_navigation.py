"""Inspect a captured depth frame without a camera or phone.

Usage: python replay_navigation.py captures/depth-....npz [--show]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from config import DEPTH_NEAR_METERS, MAX_AHEAD_METERS, V_ON, V_OFF
from grid import generate_grid
from navigation import Navigator
from stereo_obstacles import StereoObstacles
from tuning_controls import TuningValues, load_settings


def replay(path, show=False):
    with np.load(path) as sample:
        depth = sample["depth"]
        fx, fy, cx, cy = (float(sample[name]) for name in ("fx", "fy", "cx", "cy"))
        settings = TuningValues(**json.loads(str(sample["settings"]))) if "settings" in sample else load_settings()
    if depth.ndim == 2:
        depth = depth[None, ...]
    observer = StereoObstacles(
        fx=fx, fy=fy, cx=cx, cy=cy, width=depth.shape[2], height=depth.shape[1],
        near=DEPTH_NEAR_METERS, far=MAX_AHEAD_METERS,
        min_hazard_points=settings.hazard_points,
    )
    navigator = Navigator(path_width=settings.path_width,
                          stop_distance=settings.stop_distance,
                          near_limit=settings.near_distance,
                          ahead_limit=settings.ahead_distance,
                          min_observed_fraction=settings.min_coverage,
                          switch_frames=settings.switch_frames)
    for number, frame in enumerate(depth):
        scene = observer.observe(frame)
        decision = navigator.decide(scene)
        grid = generate_grid(decision)
        print(f"frame {number}: {decision.action} {decision.direction if decision.direction is not None else ''} (on={V_ON:g}, off={V_OFF:g})")
        for row in grid:
            print("".join("1" if abs(value - V_ON) < abs(value - V_OFF) else "0" for value in row))
        if not show:
            continue
        scale = np.clip(frame / MAX_AHEAD_METERS, 0, 1)
        depth_image = cv2.applyColorMap(np.uint8(scale * 255), cv2.COLORMAP_TURBO)
        depth_image[frame <= 0] = (0, 0, 0)
        local = np.zeros((*scene.blocked.shape, 3), np.uint8)
        local[scene.observed] = (0, 180, 0)
        local[scene.blocked] = (0, 0, 255)
        local = cv2.resize(local, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        hud = cv2.resize(np.uint8(grid * 255), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        hud = cv2.cvtColor(hud, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Depth | local map | HUD (any key: next, q: quit)", np.hstack((depth_image, local, hud)))
        if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
            break
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    replay(args.capture, args.show)
