"""Live desktop diagnostics for the navigation pipeline."""

import math

import cv2
import numpy as np

from config import V_ON, V_OFF


PANEL_W = 600
PANEL_H = 360
HEADER_H = 28
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(image, value, x, y, scale=0.55, color=(235, 235, 235)):
    cv2.putText(image, str(value), (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _panel(title, content):
    panel = np.full((PANEL_H, PANEL_W, 3), (28, 28, 28), np.uint8)
    _text(panel, title, 12, 20, 0.60)
    body_h = PANEL_H - HEADER_H
    panel[HEADER_H:] = cv2.resize(content, (PANEL_W, body_h), interpolation=cv2.INTER_NEAREST)
    return panel


def _camera_panel(camera, labels):
    image = cv2.cvtColor(camera, cv2.COLOR_GRAY2BGR) if camera.ndim == 2 else camera.copy()
    tint = image.copy()
    tint[labels == 1] = (40, 180, 40)     # floor
    tint[labels == 2] = (40, 40, 235)     # hazard
    mask = (labels == 1) | (labels == 2)
    image[mask] = cv2.addWeighted(image, 0.35, tint, 0.65, 0)[mask]
    return _panel("CAMERA  green=floor  red=hazard", image)


def _depth_panel(depth, debug, near, far, settings):
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    value = np.zeros(depth.shape, dtype=np.uint8)
    value[valid] = np.uint8(np.clip((far - depth[valid]) / (far - near), 0, 1) * 255)
    image = cv2.applyColorMap(value, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    panel = _panel("METRIC DEPTH  warm=near  black=invalid", image)
    details = [f"valid: {debug['valid_fraction']:.0%}"]
    if debug["floor_found"]:
        details.append(f"floor fit: {debug['floor_inliers']}/{debug['sample_count']} samples")
        details.append(f"camera height: {debug['camera_height']:.2f} m")
        details.append(f"floor tilt: {debug['floor_tilt_deg']:.0f} deg")
    else:
        details.append(f"NO FLOOR FIT: {debug['floor_status']}")
        details.append(f"support: {debug['floor_inliers']}/{debug['sample_count']} samples")
    for i, line in enumerate(details):
        y = 56 + i * 27
        cv2.rectangle(panel, (4, y - 19), (345, y + 6), (0, 0, 0), -1)
        _text(panel, line, 12, y, 0.55, (255, 255, 255))
    if settings is not None:
        cv2.rectangle(panel, (4, 258), (596, 355), (0, 0, 0), -1)
        _text(panel, f"STOP {settings.stop_distance:.2f}m   NEAR {settings.near_distance:.2f}m   AHEAD {settings.ahead_distance:.2f}m",
              12, 283, 0.57)
        _text(panel, f"Width {settings.path_width:.2f}m   Coverage {settings.min_coverage:.0%}",
              12, 312, 0.55)
        _text(panel, f"Obstacle points {settings.hazard_points}   Switch frames {settings.switch_frames}",
              12, 340, 0.55)
    return panel


def _map_panel(scene, decision):
    map_image = np.full((*scene.blocked.shape, 3), (40, 40, 40), np.uint8)
    map_image[scene.observed] = (35, 150, 35)
    map_image[scene.blocked] = (35, 35, 230)
    map_image = cv2.flip(map_image, 0)
    panel = _panel("LOCAL MAP  bottom=near  green=observed  red=blocked", map_image)
    x0, x1 = scene.x_edges[0], scene.x_edges[-1]
    z0, z1 = scene.z_edges[0], scene.z_edges[-1]
    for column, degrees in enumerate((-12, -6, 0, 6, 12)):
        points = []
        for z in np.linspace(max(z0, 0.3), z1, 30):
            x = math.tan(math.radians(degrees)) * z
            px = int((x - x0) / (x1 - x0) * (PANEL_W - 1))
            py = HEADER_H + int((z1 - z) / (z1 - z0) * (PANEL_H - HEADER_H - 1))
            points.append((px, py))
        selected = decision.action == "direction" and decision.direction == column
        cv2.polylines(panel, [np.array(points, np.int32)], False,
                      (0, 255, 255) if selected else (200, 200, 200), 3 if selected else 1)
    return panel


def _hud_panel(grid, decision, debug, stats):
    panel = np.full((PANEL_H, PANEL_W, 3), (20, 20, 20), np.uint8)
    _text(panel, f"EXACT 5x4 HUD  on={V_ON:g} off={V_OFF:g}", 12, 20, 0.60)
    size, gap, left, top = 40, 8, 20, 54
    for row in range(4):
        for col in range(5):
            x = left + col * (size + gap)
            y = top + row * (size + gap)
            value = int(np.clip(grid[row, col], 0, 1) * 255)
            color = (value, value, value)
            cv2.rectangle(panel, (x, y), (x + size, y + size), color, -1)
    names = ("ACTION", "NEAR", "AHEAD", "UNKNOWN")
    for row, name in enumerate(names):
        _text(panel, name, 265, top + 27 + row * (size + gap), 0.47)
    action = f"{decision.action.upper()}" + (f"  col {decision.direction}" if decision.direction is not None else "")
    _text(panel, action, 20, 302, 0.63, (0, 255, 255) if decision.action == "direction" else (40, 120, 255))
    _text(panel, decision.reason, 20, 332, 0.52)
    near_cov = " ".join(map(str, np.round(decision.near_coverage * 100).astype(int)))
    far_cov = " ".join(map(str, np.round(decision.ahead_coverage * 100).astype(int)))
    _text(panel, f"near cov %: {near_cov}", 350, 65, 0.40)
    _text(panel, f"far cov %:  {far_cov}", 350, 87, 0.40)
    _text(panel, f"floor cells: {debug['floor_cells']}", 350, 111, 0.44)
    _text(panel, f"hazard cells: {debug['hazard_cells']}", 350, 132, 0.44)
    _text(panel, f"camera: {stats['camera_fps']:.0f} fps", 350, 162, 0.48)
    _text(panel, f"frame age: {stats['frame_age_ms']:.0f} ms", 350, 185, 0.48)
    _text(panel, f"stereo: {stats['stereo_ms']:.0f} ms", 350, 208, 0.48)
    _text(panel, f"observe: {stats['observe_ms']:.0f} ms", 350, 231, 0.48)
    _text(panel, f"pipeline: {stats['total_ms']:.0f} ms", 350, 254, 0.48)
    gap = stats["message_gap_ms"]
    _text(panel, f"message gap: {gap:.0f} ms", 350, 277, 0.48,
          (40, 120, 255) if gap > 750 else (235, 235, 235))
    return panel


def render_dashboard(camera, depth, scene, decision, grid, debug, stats, *, near, far, settings=None):
    """Compose four views of the exact frame used to make the phone message."""
    top = np.hstack((_camera_panel(camera, debug["labels"]),
                     _depth_panel(depth, debug, near, far, settings)))
    bottom = np.hstack((_map_panel(scene, decision),
                        _hud_panel(grid, decision, debug, stats)))
    return np.vstack((top, bottom))
