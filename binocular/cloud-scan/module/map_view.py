"""Cheap top-down diagnostic rendering, separate from map integration."""

import math

import cv2
import numpy as np

from .image_tools import ImageTools
from .obstacle_map import ObstacleMap


def render_map(grid, pixels_per_cell=3, fov_deg=46.0):
    """Return a BGR top-down image; forward world Z points toward image top."""
    states = grid.state_grid()
    palette = np.array([[50, 50, 50], [65, 120, 55],
                        [40, 45, 230], [180, 40, 180]], np.uint8)
    image = np.flipud(palette[states])
    image = cv2.resize(image, None, fx=pixels_per_cell, fy=pixels_per_cell,
                       interpolation=cv2.INTER_NEAREST)
    if grid.pose is not None:
        x = int(round((grid.pose.x_m - grid.origin_x_m) / grid.cell_m * pixels_per_cell))
        y = int(round((grid.side - (grid.pose.z_m - grid.origin_z_m) / grid.cell_m)
                      * pixels_per_cell))
        length = int(round(0.7 / grid.cell_m * pixels_per_cell))
        color = (255, 255, 255) if grid.tracking_ok else (0, 180, 255)
        cv2.circle(image, (x, y), max(3, pixels_per_cell), color, -1)
        for offset in (-fov_deg / 2, 0, fov_deg / 2):
            angle = grid.pose.yaw_rad + math.radians(offset)
            end = (int(round(x + length * math.sin(angle))),
                   int(round(y - length * math.cos(angle))))
            cv2.line(image, (x, y), end, color, 1, cv2.LINE_AA)
    return image


class MapWindow:
    """Companion OpenCV window; does not alter DepthWindow's API."""

    def __init__(self, name="Obstacle map"):
        self.name = name
        self.is_open = False

    def __enter__(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.is_open = True
        return self

    def update(self, result, grid):
        if not self.is_open:
            return
        depth_image = ImageTools.depth_image(result)
        fov_deg = math.degrees(2 * math.atan(result.gray.shape[1] / (2 * result.Q[2, 3])))
        map_image = render_map(grid, pixels_per_cell=2, fov_deg=fov_deg)
        map_image = cv2.resize(map_image, (depth_image.shape[0], depth_image.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.name, np.hstack((depth_image, map_image)))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()
            return
        try:
            visible = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self.close()

    def close(self):
        if self.is_open:
            self.is_open = False
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass

    def __exit__(self, *_):
        self.close()
