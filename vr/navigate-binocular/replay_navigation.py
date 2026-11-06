"""Inspect a captured depth frame without a camera or phone.

Usage: python replay_navigation.py captures/depth-....npz [--show]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import DEPTH_NEAR_METERS, DEPTH_FAR_METERS, PATH_WIDTH_METERS
from config import NEAR_LIMIT_METERS, MIN_OBSERVED_FRACTION, SWITCH_FRAMES
from grid import generate_grid
from navigation import Navigator
from stereo_obstacles import StereoObstacles


def replay(path, show=False):
    with np.load(path) as sample:
        depth = sample["depth"]
        fx, fy, cx, cy = (float(sample[name]) for name in ("fx", "fy", "cx", "cy"))
    if depth.ndim == 2:
        depth = depth[None, ...]
    observer = StereoObstacles(
        fx=fx, fy=fy, cx=cx, cy=cy, width=depth.shape[2], height=depth.shape[1],
        near=DEPTH_NEAR_METERS, far=DEPTH_FAR_METERS,
    )
    navigator = Navigator(path_width=PATH_WIDTH_METERS, near_limit=NEAR_LIMIT_METERS,
                          min_observed_fraction=MIN_OBSERVED_FRACTION,
                          switch_frames=SWITCH_FRAMES)
    for number, frame in enumerate(depth):
        scene = observer.observe(frame)
        decision = navigator.decide(scene)
        grid = generate_grid(decision)
        print(f"frame {number}: {decision.action} {decision.direction if decision.direction is not None else ''}")
        for row in grid.astype(int):
            print("".join(map(str, row)))
        if not show:
            continue
        scale = np.clip(frame / DEPTH_FAR_METERS, 0, 1)
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
